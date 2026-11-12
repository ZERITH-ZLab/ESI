from copy import deepcopy

import pytest

from .memory import compact_limit, render_history, save_history
from . import run
from .test_cycle import Rollout
from ..test_network_continue import Transport, event


def row():
    return dict(decision=0, start_tick=0, end_tick=15, executed_steps=15, terminal=False,
        response=dict(mode='student', steps=15, reason='approach', assessment=dict(
            current_subgoal='Pick up 8', execution_status='not_started',
            execution_evidence='No previous action', predicted_next_intent='Approach 8',
            intent_status='aligned', intent_evidence='Open gripper approaches',
            task_progress=dict(currently_attempting='Approach 8', verified_completed=[],
                               remaining=['Place 8', 'Return arms']))))


def test_history_is_compact_idempotent_and_searchable(tmp_path):
    first = row()
    second = deepcopy(first)
    second.update(decision=1, start_tick=15, end_tick=18, executed_steps=3)
    second['response'].update(mode='eef', steps=5)
    second['response']['assessment'].update(execution_status='failed',
                                           execution_evidence='Object remained on table\n## Not a heading')
    second['motion_result'] = dict(tool='robodojo_move', arm='left', status='joint_tracking_error',
                                   tracking=['LARGE_TENSOR'])
    second['privileged_binding'] = {'coordinates': 'SECRET_COORDINATES'}
    second['privileged_history_exposed'] = True
    for _ in range(2):
        save_history(tmp_path, [first, second])
    text = (tmp_path/'memory/decisions.md').read_text()
    assert text.count('## Decision 001') == 1
    assert 'step 15 → 18' in text and 'requested=5, executed=3' in text
    assert 'unchanged from Decision 000' in text
    assert 'joint_tracking_error' in text
    assert 'LARGE_TENSOR' not in text and 'SECRET_COORDINATES' not in text
    assert '\n## Not a heading' not in text
    assert not (tmp_path/'memory/current.md').exists()


def test_changed_progress_and_empty_history():
    first, second = row(), row()
    second['decision'] = 1
    second['response']['assessment']['task_progress']['verified_completed'] = ['8 placed']
    assert '8 placed' in render_history([first, second])
    assert '## Decision' not in render_history([]).split('Search headings')[0]


def test_compact_config_and_identity(monkeypatch, tmp_path):
    from ..dual_campaign import controller_route_identity
    monkeypatch.delenv('ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT', raising=False)
    assert run.agent_config(tmp_path, tmp_path)['model_auto_compact_token_limit'] == 96000
    monkeypatch.setenv('ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT', '128000')
    assert run.agent_config(tmp_path, tmp_path)['model_auto_compact_token_limit'] == 128000
    assert 'model_context_window' not in run.agent_config(tmp_path, tmp_path)
    env = {'ROLLOUT_CONTROLLER_ROUTE': 'auto_infer_v1', 'ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT': '128000'}
    assert controller_route_identity(env)['model_auto_compact_token_limit'] == 128000
    assert controller_route_identity(dict(env, ROLLOUT_CONTROLLER_ROUTE='baseline')) == {}
    for invalid in ['0', '-1', 'abc']:
        with pytest.raises(ValueError):
            compact_limit({'ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT': invalid})


def test_executed_history_saved_when_auto_infer_fails(tmp_path):
    r = Rollout(infer_error=RuntimeError('inference failed'))
    r.phase = 'execute'
    r.history = []
    def execute(**kwargs):
        r.history.append(row())
        r.tick = 15
        r.phase = 'infer'
        return r.observation()
    r.execute = execute
    args = dict(response=dict(decision_id='fresh', mode='student', steps=15, assessment={}, reason='go'))
    from .output_contract import decision_id
    args['response']['decision_id'] = decision_id('fresh')
    e = event('item/tool/call', callId='1', tool='robodojo_execute', arguments=args)
    e['id'] = 1
    w = object.__new__(run.CodexPolicy)
    w.workspace = w.agent_workspace = tmp_path/'codex_workspace'
    w.workspace.mkdir()
    w.thread_id = 'thread'
    w.method, w.timeout, w.image_max_edge = 'pi05_plus_gpt', 1, 480
    w.transport = Transport([e])
    with pytest.raises(RuntimeError, match='inference failed'):
        w.run(r)
    assert 'step 0 → 15' in (tmp_path/'memory/decisions.md').read_text()
    assert len(r.history) == 1
