"""Real host state machine with synthetic simulator/model; no network or GPU."""
import json
from types import SimpleNamespace

import pytest

from ..io import InputError
from ..robodojo_server.client import RoboDojoTools
from ..test_contract import Sim, Student, response
from . import run
from .cycle import dispatch
from .feedforward import FeedforwardMixin, ROUTE, DECISION_POLICY_VERSION, adapt_prompt
from .output_contract import next_call
from .subtask_instruction import SKILLS, render_instruction, format_prompt


def instruction(skill='grasp', **changes):
    item = dict(skill_type=skill, arm='left', object_ref='digit_5', destination_ref=None,
        target_relation='none', target_orientation_ref=None, custom_description=None,
        giver_arm='unknown', receiver_arm='unknown')
    if skill in ('place', 'transport', 'push'):
        item.update(destination_ref='pad_3', target_relation='on')
    if skill == 'rotate':
        item['target_orientation_ref'] = 'upright'
    if skill == 'custom':
        item['custom_description'] = 'Transport and rotate digit 5'
    if skill == 'handover':
        item.update(arm='both', giver_arm='left', receiver_arm='right')
    if skill == 'return_home':
        item['object_ref'] = None
    return dict(item, **changes)


@pytest.mark.parametrize('skill', SKILLS)
def test_training_renderer_parity(skill):
    from data_processing.subtasks.contract import canonical
    from data_processing.openpi_loader.conditioning import format_prompt as training_prompt
    for arm in ('left', 'right', 'both') if skill != 'handover' else ('both',):
        item = instruction(skill, arm=arm)
        assert render_instruction(item) == canonical(item)
        assert format_prompt('Original task.', render_instruction(item)) == training_prompt('Original task.', canonical(item))
    assert format_prompt('Original task.\n', None) == training_prompt('Original task.\n', None)
    assert render_instruction(instruction('grasp', target_orientation_ref='upright')) == canonical(instruction('grasp', target_orientation_ref='upright'))


class RecordingStudent(Student):
    def __init__(self):
        self.prompts = []
        self.error = None

    def infer(self, obs, output):
        self.prompts.append(obs['instruction'])
        if self.error:
            raise self.error
        return super().infer(obs, output)


class GuidedTools(FeedforwardMixin, RoboDojoTools):
    pass


def setup(tmp_path, cls=GuidedTools, sim=Sim):
    (tmp_path/'controller').mkdir()
    r = cls(tmp_path/'controller', 'arrange_largest_number', RecordingStudent(), rpc_factory=sim, max_decisions=0)
    call = r.next_call()
    packet = dispatch(r, call['tool'], {k:v for k,v in call.items() if k != 'tool'}, r.timing, 0)
    assert r.student.prompts == []
    assert packet['next_call']['tool'] == 'pi05_infer'
    return r, packet


def predict(r, subtask=None, intent='Acquire digit 5 with the left arm; verify retention before carrying.'):
    return dispatch(r, 'pi05_infer', dict(observation_id=r.observation_id, subtask=subtask, current_intent=intent), r.timing, 1)


def execute(r, mode='student', steps=1):
    args = {k:v for k,v in r.next_call().items() if k != 'tool'}
    args['response'] = dict(response(r.request, mode), steps=steps)
    if r.tick:
        args['response']['assessment']['execution_status'] = 'progressing'
    return dispatch(r, 'robodojo_execute', args, r.timing, 2)


def test_full_cycle_preserves_task_null_and_history(tmp_path):
    r, initial = setup(tmp_path)
    assert next_call(r)['observation_id'] == initial['observation_id']
    p = predict(r, instruction())
    assert r.obs['instruction'] == 'Arrange the digits.'
    assert r.student.prompts == ['Arrange the digits.\nCurrent subtask: Left arm: Grasp digit 5.']
    assert p['conditioning']['prediction_id'] == r.prediction['prediction_id']
    assert 'conditioning.current_intent' in r.request['gate_instruction']
    assert r.run['decision_policy_version'] == DECISION_POLICY_VERSION
    with pytest.raises(InputError, match='already consumed'):
        predict(r)
    packet = execute(r)
    assert len(r.student.prompts) == 1  # Never auto infer after executing.
    assert packet['observation_id'] != initial['observation_id']
    assert r.history[0]['conditioning']['subtask_text'] == 'Left arm: Grasp digit 5.'
    saved = json.loads(r.observation_path.read_text())
    assert saved['previous_result']['conditioning'] == r.history[0]['conditioning']
    p = predict(r, None)
    assert r.student.prompts[-1] == 'Arrange the digits.'
    assert p['conditioning']['subtask'] is None
    assert p['conditioning']['current_intent'] == r.history[0]['conditioning']['current_intent']
    assert p['conditioning']['decision_policy_version'] == DECISION_POLICY_VERSION
    from .input_view import teacher_view
    from .memory import render_history
    assert teacher_view(p)['conditioning']['current_intent'] == p['conditioning']['current_intent']
    assert p['conditioning']['current_intent'] in render_history(r.history)
    assert 'current_intent' not in r.student.prompts[-1]
    assert json.loads((r.proposal_path.parent/'conditioning.json').read_text()) == p['conditioning']
    terminal = execute(r)
    assert terminal['rollout_finished'] and terminal['next_call'] is None
    assert len(r.student.prompts) == 2


def test_bad_fields_and_stale_ids_do_not_infer(tmp_path):
    r, _ = setup(tmp_path)
    for args in (dict(observation_id='stale', subtask=None, current_intent='Grasp digit 5.'),
                 dict(observation_id=r.observation_id, subtask=instruction(arm='unknown'), current_intent='Grasp digit 5.'),
                 dict(observation_id=r.observation_id, subtask={}, current_intent='Grasp digit 5.'),
                 dict(observation_id=r.observation_id, subtask=None, current_intent='Grasp digit 5.', extra=1)):
        with pytest.raises(InputError):
            dispatch(r, 'pi05_infer', args, r.timing, 1)
    assert r.student.prompts == [] and r.phase == 'infer'
    predict(r, None)  # Validation rejection can be corrected before inference.


def test_inference_failure_is_fatal_not_resampled(tmp_path):
    r, _ = setup(tmp_path)
    r.student.error = OSError('lost connection')
    with pytest.raises(RuntimeError, match='abort without replay'):
        predict(r, instruction())
    assert r.phase == 'failed'
    with pytest.raises(InputError):
        predict(r, None)
    assert len(r.student.prompts) == 1
    assert (r.observation_path.parent/'conditioning_selection.json').exists()


def test_zero_step_correction_still_requires_new_plan(tmp_path):
    class RefusalSim(Sim):
        def request(self, op, **args):
            if op == 'eef_short_motion':
                return dict(step_id=self.tick, steps=[], correction_result={'status':'planning_failed'})
            return super().request(op, **args)
    r, first = setup(tmp_path, sim=RefusalSim)
    predict(r, instruction())
    packet = execute(r, 'eef')
    assert r.tick == 0 and len(r.history) == 1
    assert packet['observation_id'] != first['observation_id']
    assert r.phase == 'infer' and len(r.student.prompts) == 1
    predict(r, None)
    assert len(r.student.prompts) == 2


def test_motion_completion_uses_same_no_auto_infer_dispatch(tmp_path, monkeypatch):
    from .motion import MotionMixin
    cls = type('GuidedMotion', (FeedforwardMixin, MotionMixin, RoboDojoTools), {})
    r, _ = setup(tmp_path, cls)
    predict(r, instruction())
    # Isolate existing motion geometry/validation, exercise its actual dispatch path.
    args = {k:v for k,v in r.next_call().items() if k != 'tool'}
    args['response'] = dict(response(r.request), steps=1)
    (r.output/'motion').mkdir()
    r._motion_plan = dict(status='done', initial=args['response']['target'])
    monkeypatch.setattr(r, 'prepare_motion', lambda *a: args)
    r.execute_motion('robodojo_move', {}, r.timing, 2)
    assert r.phase == 'infer' and len(r.student.prompts) == 1


def test_runtime_prompt_schema_eval_isolation(tmp_path, monkeypatch):
    from ..regression_eval import required_env
    captured = []
    for name, value in required_env(ROUTE).items():
        monkeypatch.setenv(name, value)
    class Transport:
        def __init__(self, *args): self.process = SimpleNamespace(pid=0)
        def request(self, method, params, timeout):
            if method == 'initialize': return {}
            captured.append(params)
            return dict(model=run.MODEL, reasoningEffort=run.EFFORT, thread={'id':'new-thread'})
        def notify(self, *args): pass
        def close(self): pass
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='offline'))
    w = run.CodexPolicy(tmp_path/'audit', 'unused', task='arrange_largest_number', transport_factory=Transport)
    p = captured[0]
    assert not w.scene_query_enabled and not w.abandon_enabled and w.feedforward
    specs = {s['name']:s for s in p['dynamicTools']}
    assert set(specs) == {'robodojo_start','pi05_infer','robodojo_execute','robodojo_move','robodojo_grasp','robodojo_history'}
    assert {'type':'null'} in specs['pi05_infer']['inputSchema']['properties']['subtask']['anyOf']
    for text in [p['developerInstructions'], *(f.read_text() for f in w.agent_workspace.rglob('*.md'))]:
        for forbidden in ('Never manually call infer', 'proposal automatically;', 'not your subtask text', 'There is no separate callable pi05_infer'):
            assert forbidden not in text
    assert 'Use null' in p['developerInstructions'] and 'object/acting arm cannot be determined' in p['developerInstructions']
    assert json.loads((w.agent_workspace/'workspace.json').read_text())['controller_route'] == ROUTE
    injected = p['developerInstructions']
    assert 'conditioning.current_intent' in injected
    assert 'current_intent' in specs['pi05_infer']['inputSchema']['required']
    assert 'Reissuing a subtask is neither correction nor execution progress.' in injected
    gate = (run.SKILL_ROOT/'feedforward_gate_prompt.md').read_text()
    assert injected.count(gate) == 1
    assert (w.workspace/'gate_prompt.md').read_text() == gate
    assert adapt_prompt(injected) == injected
    for text in [injected, *(f.read_text() for f in w.agent_workspace.rglob('*.md')),
                 *(s['description'] for s in specs.values())]:
        for old in ('apply the unchanged', 'appropriate subgoal?',
                    'student remains preferred when its proposal is aligned',
                    'prefer aligned student proposals', 'hand back to student as soon as appropriate'):
            assert old not in text
    assert json.loads((w.workspace.parent/'controller_protocol.json').read_text())['decision_policy_version'] == DECISION_POLICY_VERSION
    w.close()


def test_codex_tool_loop_guidance_rejection_then_execute(tmp_path):
    from ..test_network_continue import Transport, event, completed
    from .output_contract import decision_id
    r, _ = setup(tmp_path)
    def call(i, name, args):
        return dict(event('item/tool/call', callId=str(i), tool=name, arguments=args), id=i)
    def execute_args():
        value = response(r.request)
        return dict(response=dict(decision_id=decision_id(r.request['request_id']),
            **{k:value[k] for k in ('mode','steps','assessment','reason')}))
    class DynamicTransport(Transport):
        def next_message(self, timeout):
            if callable(self.events[0]):
                self.events[0] = self.events[0]()
            return super().next_message(timeout)
    w = object.__new__(run.CodexPolicy)
    w.workspace = w.agent_workspace = tmp_path/'audit'
    w.workspace.mkdir()
    w.thread_id, w.method = 'thread', 'pi05_plus_gpt'
    w.timeout, w.image_max_edge, w.feedforward = 1, 480, True
    w.transport = DynamicTransport([
        lambda: call(0, 'pi05_infer', dict(observation_id=r.observation_id, subtask=instruction(), current_intent='Grasp digit 5.')),
        lambda: call(1, 'pi05_infer', dict(observation_id=r.observation_id, subtask=None, current_intent='Verify retention of digit 5.')),
        lambda: call(2, 'robodojo_execute', execute_args()),
        completed('t0', 'completed', None)])
    w.run(r)
    assert r.phase == 'done' and len(r.student.prompts) == 1 and len(r.history) == 1
    assert [reply['success'] for _,reply in w.transport.replies] == [True,False,True]
    rejected = json.loads(w.transport.replies[1][1]['contentItems'][0]['text'])
    assert rejected['next_call']['tool'] == 'robodojo_execute'


def test_eval_campaign_selects_route_without_changing_old_defaults(tmp_path):
    from ..test_regression_eval import setup as eval_setup
    from ..dual_campaign import configure, controller_route_identity
    from ..eval_sequence import campaign_env
    _, _, env = eval_setup(tmp_path)
    cfg = dict(workers=4, account_pool_file='/account_pool.json')
    base = dict(experiment='test', checkpoint='/checkpoint', policy_input='task_only')
    old = campaign_env(env, cfg, base, env['ROLLOUT_EVAL_MANIFEST'])
    assert old['ROLLOUT_CONTROLLER_ROUTE'] == 'auto_infer_v1'
    new = campaign_env(env, cfg, dict(base, policy_input='task_subtask'), env['ROLLOUT_EVAL_MANIFEST'])
    assert new['ROLLOUT_CONTROLLER_ROUTE'] == ROUTE
    # Single-seed fixture: queue generation is independently covered by eval_sequence tests.
    new.update(ROLLOUT_POLICY_RNG_SEED='0', ROLLOUT_COUNT='5')
    _, cases = configure(new)
    assert len(cases) == 5
    assert controller_route_identity(new)['conditioning_version'] == ROUTE
    assert 'conditioning_version' not in controller_route_identity(old)


@pytest.mark.parametrize('intent', [None, '', '   ', 42, 'a' * 601, 'goal\nnext', 'goal\n'])
def test_invalid_intent_does_not_consume_inference(tmp_path, intent):
    r, _ = setup(tmp_path)
    with pytest.raises(InputError, match='current_intent'):
        predict(r, intent=intent)
    assert not r.student.prompts and r.phase == 'infer'
    predict(r, intent='Verify the retained object before carrying.')
    assert r.student.prompts == ['Arrange the digits.']


def test_missing_intent_rejected_before_inference(tmp_path):
    r, _ = setup(tmp_path)
    with pytest.raises(InputError, match='current_intent'):
        r.infer_subtask(dict(observation_id=r.observation_id, subtask=None))
    assert not r.student.prompts


def test_legacy_prompt_and_tools_remain_unadapted(tmp_path):
    from .motion import PROMPT as motion_prompt, allow_motion
    original = allow_motion(run.build_prompt(run.SKILL_ROOT, tmp_path)) + motion_prompt
    guided = adapt_prompt(original)
    assert 'appropriate subgoal?' in original
    assert 'student remains preferred when its proposal is aligned' in original
    assert 'conditioning.current_intent' not in original
    assert 'conditioning.current_intent' in guided
    assert [s['name'] for s in run.tool_specs()] == ['robodojo_start', 'robodojo_execute']
    assert 'current_intent' not in json.dumps(run.tool_specs())
