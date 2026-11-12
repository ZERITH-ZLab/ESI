import json
from copy import deepcopy
from pathlib import Path

import pytest

from ..io import InputError
from ..timing import Timing
from ..skill import run as baseline
from ..test_network_continue import Transport, event, completed
from . import run
from .cycle import dispatch
from .workspace import prepare_workspace
from .output_contract import decision_id


class Rollout:
    def __init__(self, *, terminal=False, reject=False, infer_error=None):
        self.phase = 'start'
        self.tick = 0
        self.counters = {}
        self.calls = []
        self.terminal, self.reject, self.infer_error = terminal, reject, infer_error
        self.proposal_path = Path('/proposal/actions.npz')
        self.observation_path = '/observation.json'
        pose = dict(position=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0], gripper_closed=False)
        self.request = {'request_id': 'fresh', 'current_eef': {'left': deepcopy(pose), 'right': deepcopy(pose)}}

    def next_call(self):
        if self.phase == 'done':
            return None
        return dict(tool={'start': 'robodojo_start', 'infer': 'pi05_infer',
                          'execute': 'robodojo_execute'}.get(self.phase),
                    task='fold', proposal_path=str(self.proposal_path),
                    observation_path=self.observation_path, output_dir='/proposal')

    def observation(self):
        return dict(episode_id='episode', step_id=self.tick,
                    rollout_finished=self.phase == 'done', images=[],
                    task_context={'goal': 'fold'}, result={'success': self.terminal},
                    previous_result={'executed_steps': self.tick},
                    observation_path=self.observation_path, next_call=self.next_call())

    def start(self, **kwargs):
        self.calls.append('start')
        self.phase = 'infer'
        return self.observation()

    def execute(self, **kwargs):
        if self.reject:
            raise InputError('invalid action')
        self.calls.append('execute')
        self.tick += 3
        self.phase = 'done' if self.terminal else 'infer'
        return self.observation()

    def infer(self, **kwargs):
        assert self.phase == 'infer'
        assert kwargs['observation_path'] == self.observation_path
        self.calls.append('infer')
        if self.infer_error:
            raise self.infer_error
        self.phase = 'execute'
        self.counters['predictions'] = self.counters.get('predictions', 0) + 1
        return dict(episode_id='episode', step_id=self.tick, request_id='fresh',
                    proposal_path=str(self.proposal_path), next_call=self.next_call(),
                    images=[], counters=self.counters.copy())


@pytest.mark.parametrize('tool', ['robodojo_start', 'robodojo_execute'])
def test_fresh_infer_and_merged_packet(tmp_path, tool):
    r = Rollout()
    p = dispatch(r, tool, {}, Timing(tmp_path/'timing.jsonl'), 0)
    assert r.calls == [tool.removeprefix('robodojo_'), 'infer']
    assert p['next_call']['tool'] == 'robodojo_execute'
    assert p['task_context'] == {'goal': 'fold'}
    assert p['previous_result']['executed_steps'] == r.tick
    assert p['observation_path'] == r.observation_path
    assert p['request_id'] == 'fresh'
    assert p['counters']['predictions'] == 1
    events = [json.loads(l) for l in (tmp_path/'timing.jsonl').read_text().splitlines()]
    assert [(e['event'], e['stage']) for e in events] == [
        ('start', 'service.'+tool), ('end', 'service.'+tool),
        ('start', 'service.pi05_infer'), ('end', 'service.pi05_infer')]


def test_terminal_skips_infer(tmp_path):
    r = Rollout(terminal=True)
    p = dispatch(r, 'robodojo_execute', {}, Timing(tmp_path/'t'), 0)
    assert p['rollout_finished'] and p['result']['success']
    assert r.calls == ['execute']


def test_rejection_skips_infer(tmp_path):
    r = Rollout(reject=True)
    with pytest.raises(InputError):
        dispatch(r, 'robodojo_execute', {}, Timing(tmp_path/'t'), 0)
    assert r.calls == []


@pytest.mark.parametrize('error', [InputError('bad infer'), RuntimeError('lost RPC'), TimeoutError()])
def test_infer_failure_never_replays_action(tmp_path, error):
    r = Rollout(infer_error=error)
    with pytest.raises((RuntimeError, TimeoutError)):
        dispatch(r, 'robodojo_execute', {}, Timing(tmp_path/'t'), 0)
    assert r.calls == ['execute', 'infer'] and r.tick == 3


def test_mismatched_proposal_fails_closed(tmp_path):
    r = Rollout()
    original = r.infer
    def mismatched(**kw):
        return dict(original(**kw), step_id=999)
    r.infer = mismatched
    with pytest.raises(RuntimeError, match='mismatch'):
        dispatch(r, 'robodojo_execute', {}, Timing(tmp_path/'t'), 0)
    assert r.calls == ['execute', 'infer']


def test_tools_and_workspace_are_isolated(tmp_path):
    assert [s['name'] for s in run.tool_specs()] == ['robodojo_start', 'robodojo_execute']
    assert 'pi05_infer' in [s['name'] for s in baseline.tool_specs()]
    assert run.tool_specs()[1]['inputSchema']['required'] == ['response']
    assert (run.SKILL_ROOT/'gate_prompt.md').read_bytes() == (baseline.SKILL_ROOT/'gate_prompt.md').read_bytes()
    with pytest.raises(ValueError):
        run.tool_specs('gpt_only')
    audit = tmp_path/'audit'
    audit.mkdir()
    agent = prepare_workspace(audit, run.SKILL_ROOT)
    assert 'Two additional tools' in (agent/'AGENTS.md').read_text()
    assert (agent/'.agents/skills/robodojo-auto-infer-rollout/SKILL.md').exists()


def test_controller_runs_without_gpt_infer_call(tmp_path):
    def call(name, i):
        args = dict(task='fold') if name == 'robodojo_start' else dict(response=dict(
            decision_id=decision_id('fresh'), mode='student', steps=3, assessment={}, reason='aligned'))
        e = event('item/tool/call', callId=str(i), tool=name, arguments=args)
        e['id'] = i
        return e
    w = object.__new__(run.CodexPolicy)
    w.workspace = w.agent_workspace = tmp_path
    w.thread_id = 'thread'
    w.method, w.timeout, w.image_max_edge = 'pi05_plus_gpt', 1, 480
    w.transport = Transport([call('robodojo_start', 1), call('robodojo_execute', 2),
                             completed('t0', 'completed', None)])
    r = Rollout(terminal=True)
    w.run(r)
    assert r.calls == ['start', 'infer', 'execute']
    assert len(w.transport.replies) == 2
    first = json.loads(w.transport.replies[0][1]['contentItems'][0]['text'])
    assert first['decision_id'] == decision_id('fresh') and first['task_context']
    raw = json.loads((tmp_path/'call_0001_request.json').read_text())
    expanded = json.loads((tmp_path/'call_0001_expanded_request.json').read_text())
    assert 'edit' not in raw['arguments']['response']
    assert expanded['arguments']['response']['request_id'] == 'fresh'


def test_agent_cannot_call_infer(tmp_path):
    r = Rollout()
    with pytest.raises(InputError):
        dispatch(r, 'pi05_infer', {}, Timing(tmp_path/'t'), 0)
    assert not r.calls


def test_uncertain_execute_never_infers(tmp_path):
    r = Rollout()
    def uncertain(**kw):
        r.calls.append('execute')
        raise TimeoutError('ACK unknown')
    r.execute = uncertain
    with pytest.raises(TimeoutError):
        dispatch(r, 'robodojo_execute', {}, Timing(tmp_path/'t'), 0)
    assert r.calls == ['execute']


def test_three_images_attached_once(monkeypatch):
    monkeypatch.setattr(run, 'prepare_image', lambda item, edge: (dict(item), b'png'))
    packet = dict(images=[dict(camera=c, path=c+'.png') for c in ('head', 'left', 'right')])
    items = run.content_items(packet, images=True)
    assert [i['type'] for i in items] == ['inputText', 'inputImage', 'inputImage', 'inputImage']


def test_native_failure_also_skips_infer(tmp_path):
    r = Rollout(terminal=True)
    original = r.execute
    def failed(**kw):
        return dict(original(**kw), result={'success': False, 'truncated': True})
    r.execute = failed
    p = dispatch(r, 'robodojo_execute', {}, Timing(tmp_path/'t'), 0)
    assert p['rollout_finished'] and not p['result']['success']
    assert r.calls == ['execute']


def test_launch_route_identity():
    from ..dual_campaign import controller_route_identity
    assert controller_route_identity({}) == {}
    assert controller_route_identity({'ROLLOUT_CONTROLLER_ROUTE': 'baseline'}) == {}
    assert controller_route_identity({'ROLLOUT_CONTROLLER_ROUTE': 'auto_infer_v1'}) == {
        'controller_route': 'auto_infer_v1', 'decision_history_version': 'decision_history_v1',
        'model_auto_compact_token_limit': 96000}
    with pytest.raises(ValueError):
        controller_route_identity({'ROLLOUT_CONTROLLER_ROUTE': 'unknown'})
