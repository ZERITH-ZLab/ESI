"""Prompt-only metadata and future-only migration, no model/cluster/physics."""
import json

import pytest

from . import campaign
from .io import write_json
from .prompt_context import CONTEXT_VERSION
from .robodojo_server.client import RoboDojoTools
from .test_contract import BenchmarkSim, Student, invoke, response


def test_context_budget_tracks_native_steps_without_new_queries(tmp_path):
    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=BenchmarkSim, max_decisions=0)
    obs = invoke(tools)
    assert (obs['step_id'], obs['max_episode_steps'], obs['remaining_steps'], obs['rollout_finished']) == (0, 2, 2, False)
    assert obs['task_context']['requires_arm_return'] is True
    assert 'digits' in obs['task_context']['process']
    proposal = invoke(tools)
    assert (proposal['step_id'], proposal['max_episode_steps'], proposal['remaining_steps'], proposal['rollout_finished']) == (0, 2, 2, False)
    decision = response(tools.request); decision['steps'] = 1
    obs = invoke(tools, response=decision)
    assert 'task_context' not in obs  # First observation only; no repetitive context cost.
    assert (obs['step_id'], obs['max_episode_steps'], obs['remaining_steps'], obs['rollout_finished']) == (1, 2, 1, False)
    invoke(tools)
    decision = response(tools.request); decision['steps'] = 1
    decision['assessment']['execution_status'] = 'progressing'
    final = invoke(tools, response=decision)
    assert (final['step_id'], final['max_episode_steps'], final['remaining_steps'], final['rollout_finished']) == (2, 2, 0, True)
    assert final['result']['complete'] and not final['result']['success']
    assert json.loads((tmp_path/'run.json').read_text())['context_version'] == CONTEXT_VERSION == 'v3'
    assert not any('score' in op or 'reward' in op for op in tools.sim.calls)
    assert tools.sim.calls.count('reset') == 1


def test_task_notes_home_exception_and_variants():
    from .prompt_context import task_context, TASK_NOTES
    from .evaluation import TASKS
    assert set(TASKS) <= set(TASK_NOTES)  # Extra supported tasks need not be in this old panel.
    for task in TASK_NOTES:
        note = task_context(task)
        assert note == task_context(task+'_random')
        assert note['requires_arm_return'] == (task != 'make_kong')
    assert task_context('unknown_task') == {}








def test_case_binding_follows_case_not_slot_and_rejects_wrong_context(tmp_path, monkeypatch):
    plan = dict(queue=[dict(case={'case_id': 'old'}), dict(case={'case_id':'new'})],
        source_sha256='new', context_version='v2', cluster='c', workspace='w', worker_spec='sku', quota='reserved',
        case_runtime_overrides={'0':dict(source_sha256='old', context_version='v1', dispatcher_source='/old')})
    path = tmp_path/'submission.json'
    monkeypatch.setattr(campaign, 'submission_path', lambda p, s:path)
    for index, source, context in [(0, 'old', 'v1'), (1, 'new', 'v2')]:
        sub = dict(source_sha256=source, cluster='c', workspace='w', worker_spec='sku', quota='reserved',
            environment={'ROLLOUT_AUTH_PROFILE':'codex_b'}, evaluation_cases=[plan['queue'][index]['case']])
        if context == 'v2': sub['context_version'] = context
        write_json(path, sub)
        slot = dict(id=3, index=index, auth_profile='codex_b')
        assert campaign.ensure_submission(plan, slot) == path
        sub['context_version'] = 'v2' if context == 'v1' else 'v1'
        write_json(path, sub)
        with pytest.raises(ValueError, match='differs'):
            campaign.ensure_submission(plan, slot)
