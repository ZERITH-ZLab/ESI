"""Offline eval isolation, pairing and collection compatibility checks."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import regression_eval as e
from .deployment import verify_policy
from .dual_campaign import configure
from .evaluation import read_panel, verify_assets
from .io import InputError, write_json
from .robodojo_server.step_budget import apply
from .test_collection import setup as collection_setup


def setup(tmp_path, seed=0):
    source, _, old_env = collection_setup(tmp_path)
    panel = e.freeze(source, seed)
    path = tmp_path/'eval.json'
    write_json(path, panel)
    env = dict(old_env, **e.REQUIRED_ENV)
    env.update(ROLLOUT_EVAL_MANIFEST=str(path), ROLLOUT_POLICY_RNG_SEED=str(seed), ROLLOUT_COUNT='5')
    return source, panel, env


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_pairing_and_native_budget(tmp_path, seed):
    source, panel, env = setup(tmp_path, seed)
    verify_assets(read_panel(env['ROLLOUT_EVAL_MANIFEST']), source)
    _, cases = configure(env)
    assert [(r['variant'], r['layout_id']) for r in cases] == list(e.CASES)
    assert {r['policy_rng_seed'] for r in cases} == {seed}
    native = SimpleNamespace(step_lim=1050)
    budget = apply(native, 0, panel)
    assert native.step_lim == 1050 and not budget['step_limit_overridden']
    for limit, override in ((1500, 0), (1050, 1500), (1000, 0)):
        with pytest.raises(ValueError):
            apply(SimpleNamespace(step_lim=limit), override, panel)


@pytest.mark.parametrize('key,value', [
    ('ROLLOUT_ENABLE_SCENE_QUERY', '1'), ('ROLLOUT_ENABLE_TASK_MEMORY', '1'),
    ('ROLLOUT_ENABLE_TEACHER_ABANDON', '1'), ('ROLLOUT_COLLECTION_STEP_LIMIT', '1500'),
    ('MAX_DECISIONS', '180'), ('ROLLOUT_STRICT_EVAL', '0'),
    ('ROLLOUT_POLICY_RNG_SEED', '2'), ('ROLLOUT_ALLOW_SOURCE_UPGRADE', '1'),
    ('ROLLOUT_WORKER_CONTROL_FILE', '/tmp/workers.json')])
def test_reject_mixed_protocol(tmp_path, key, value):
    _, _, env = setup(tmp_path)
    with pytest.raises(ValueError):
        configure(dict(env, **{key: value}))


def test_filter_preserves_actions_and_archive(monkeypatch):
    from .skill_auto_infer.run import content_items
    packet = dict(step_id=50, images=[], rollout_finished=False,
        task_context={'goal': 'original prompt'}, current_state=[1, 2],
        previous_result={'native_success': True, 'executed_steps': 15,
            'response': {'assessment': {'execution_status': 'success', 'execution_evidence': 'visually placed'}},
            'correction_result': {'status': 'tracking_failed', 'native_score': .5}},
        result={'success': True, 'native_score': 1, 'reason': 'native_completed'})
    original = deepcopy(packet)
    monkeypatch.setenv('ROLLOUT_STRICT_EVAL', '1')
    visible = json.loads(content_items(packet, images=False)[0]['text'])
    assert 'native_success' not in json.dumps(visible) and 'native_score' not in json.dumps(visible)
    assert 'result' not in visible
    assert visible['previous_result']['executed_steps'] == 15
    assert visible['previous_result']['correction_result']['status'] == 'tracking_failed'
    assert visible['previous_result']['response']['assessment']['execution_status'] == 'success'
    assert visible['task_context'] == original['task_context']
    assert packet == original
    monkeypatch.delenv('ROLLOUT_STRICT_EVAL')
    old = json.loads(content_items(packet, images=False)[0]['text'])
    assert old['previous_result']['native_success'] is True


def test_history_only_contains_this_episodes_visual_assessments():
    rows = [dict(decision=0, native_success=True, native_score=1, terminal=False,
        executed_steps=5, response={'assessment': {'execution_evidence': 'Object fell visually'}})]
    text = e.history_packet(rows, {})['decision_history']
    assert 'Object fell visually' in text and 'native_score' not in text
    assert 'Decision 000' not in e.history_packet(rows, {'from_decision': 1})['decision_history']
    with pytest.raises(InputError):
        e.history_packet(rows, {'path': '/sim/summary.json'})


def test_eval_tools_cannot_read_raw_archives_and_prompt_unchanged(monkeypatch, tmp_path):
    from .skill_auto_infer import run
    before = run.build_prompt(run.SKILL_ROOT, tmp_path)
    monkeypatch.setenv('ROLLOUT_STRICT_EVAL', '1')
    config = run.agent_config(Path('/audit'), Path('/agent'))
    assert not config['features.shell_tool'] and not config['features.view_image']
    assert '/audit' not in config['permissions.rollout_agent.filesystem']
    assert run.build_prompt(run.SKILL_ROOT, tmp_path) == before
    monkeypatch.delenv('ROLLOUT_STRICT_EVAL')
    assert run.agent_config(Path('/audit'), Path('/agent'))['features.shell_tool']


def test_loaded_policy_seed_must_match_frozen_case(tmp_path):
    write_json(tmp_path/'resume_identity.json', {'checkpoint_sha256': 'abc', 'eval_protocol': e.VERSION})
    write_json(tmp_path/'evaluation_case.json', {'case': {'policy_rng_seed': 2}})
    write_json(tmp_path/'policy_identity.json', {'checkpoint_sha256': 'abc', 'policy_rng_seed': 0, 'inferences': 0})
    with pytest.raises(ValueError, match='paired policy seed'):
        verify_policy(tmp_path)
    write_json(tmp_path/'policy_identity.json', {'checkpoint_sha256': 'abc', 'policy_rng_seed': 2, 'inferences': 0})
    verify_policy(tmp_path)


def test_fresh_threads_and_exposed_tools(tmp_path, monkeypatch):
    from .skill_auto_infer import run
    captured = []
    for name, value in e.REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    class Transport:
        def __init__(self, *args):
            self.process = SimpleNamespace(pid=0)
        def request(self, method, params, timeout):
            if method == 'initialize':
                return {}
            assert method == 'thread/start'
            captured.append(params)
            return dict(model=run.MODEL, reasoningEffort=run.EFFORT,
                        thread={'id': f'fresh-{len(captured)}'})
        def notify(self, *args):
            pass
        def close(self):
            pass
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='offline'))
    for i in range(2):
        worker = run.CodexPolicy(tmp_path/f'episode{i}/codex_workspace', 'unused',
            task='arrange_largest_number', transport_factory=Transport)
        assert worker.thread_id == f'fresh-{i+1}'
        assert not worker.scene_query_enabled and not worker.abandon_enabled
        worker.close()
    for params in captured:
        assert params['ephemeral'] is True
        tools = {s['name'] for s in params['dynamicTools']}
        assert tools == {'robodojo_start', 'robodojo_execute', 'robodojo_move', 'robodojo_grasp', 'robodojo_history'}
        assert '# Native task success conditions: arrange_largest_number' in params['developerInstructions']


def test_history_tool_never_advances_physics(tmp_path, monkeypatch):
    from .skill_auto_infer import run
    from .skill_auto_infer.test_cycle import Rollout
    from .test_network_continue import Transport, event, completed
    from .skill_auto_infer.output_contract import decision_id
    monkeypatch.setenv('ROLLOUT_STRICT_EVAL', '1')
    events = []
    for i, (tool, args) in enumerate([
        ('robodojo_start', {'task': 'fold'}), ('robodojo_history', {}),
        ('robodojo_execute', dict(response=dict(decision_id=decision_id('fresh'),
             mode='student', steps=3, assessment={}, reason='aligned')))]):
        events.append(dict(event('item/tool/call', callId=str(i), tool=tool, arguments=args), id=i))
    worker = object.__new__(run.CodexPolicy)
    worker.workspace = worker.agent_workspace = tmp_path
    worker.thread_id, worker.method = 'thread', 'pi05_plus_gpt'
    worker.timeout, worker.image_max_edge, worker.strict_eval = 1, 480, True
    worker.transport = Transport(events + [completed('t0', 'completed', None)])
    rollout = Rollout(terminal=True)
    worker.run(rollout)
    assert rollout.calls == ['start', 'infer', 'execute']
    replies = [json.loads(r['contentItems'][0]['text']) for _, r in worker.transport.replies]
    assert 'decision_history' in replies[1]
    assert replies[2]['rollout_finished'] and 'result' not in replies[2]
    assert json.loads((tmp_path/'call_0002_result.json').read_text())['result']['success']


def test_summary_separates_native_failure_from_incomplete_and_other_policy(tmp_path):
    case = next(e.specs(0))
    row = dict(case_id=case['case_id'], archive='/archive', state='completed', artifact_status='verified',
        reset=dict(all_owned_processes_exited=True, ports_released=True),
        result=dict(complete=True, edited_steps=4, recovery_steps=6, student_steps=1040, step_id=1050),
        evaluation_outcome=dict(complete=True, valid_for_success_rate=True, native_success=False, native_score=.5))
    path = tmp_path/'results/base_pseed0/_replicas/replica_3/attempt_0/batch.json'
    path.parent.mkdir(parents=True)
    write_json(path, dict(episodes=[row]))
    summary = e.report(tmp_path, 'base')
    assert summary['completed'] == summary['failures'] == 1 and summary['successes'] == 0
    assert summary['incomplete_or_unstarted'] == 14
    assert summary['episodes'][0]['gpt_steps'] == 10
    assert e.report(tmp_path, 'trained')['completed'] == 0


@pytest.mark.parametrize('workers', [1, 4])
def test_resume_starts_missing_seed_campaigns_fresh(tmp_path, monkeypatch, workers):
    import sys
    source, _, _ = setup(tmp_path)
    shared = tmp_path/'shared'
    for seed in range(3):
        path = shared/f'eval_panels/test/pseed{seed}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, e.freeze(source, seed))
    prior = shared/'results/test_pseed0/_replicas/replica_3/attempt_0'
    prior.mkdir(parents=True)
    captured = []
    monkeypatch.setenv('ROBODOJO_SOURCE', str(source))
    monkeypatch.setenv('ROLLOUT_SHARED_ROOT', str(shared))
    monkeypatch.setenv('ROLLOUT_EXPERIMENT_ID', 'test')
    monkeypatch.setenv('ROLLOUT_ENABLE_SCENE_QUERY', '1')
    monkeypatch.setenv('ROLLOUT_WORKER_CONTROL_FILE', '/old-collection-control')
    monkeypatch.setenv('ROLLOUT_ACCOUNT_POOL_FILE', '/test-pool.json')
    monkeypatch.setattr(sys, 'argv', ['eval', 'resume', '--workers', str(workers)])
    def launch(argv, env):
        captured.append(env)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(e.subprocess, 'run', launch)
    with pytest.raises(SystemExit) as exit_info:
        e.main()
    assert exit_info.value.code == 0
    assert [v['ROLLOUT_RUN_MODE'] for v in captured] == ['resume', 'fresh', 'fresh']
    assert [v['ROLLOUT_POLICY_RNG_SEED'] for v in captured] == ['0', '1', '2']
    assert all(v['ROLLOUT_PARALLEL_WORKERS'] == str(workers) for v in captured)
    assert all(v['ROLLOUT_ENABLE_SCENE_QUERY'] == '0' and not v.get('ROLLOUT_WORKER_CONTROL_FILE') for v in captured)


@pytest.mark.parametrize('unified', [False, True])
def test_four_eval_workers_claim_cases_once(tmp_path, monkeypatch, unified):
    import os
    import time
    from . import parallel_collection as p
    from .test_parallel_collection import environment
    from .deployment import source_identity
    source, _, env = setup(tmp_path)
    if unified:
        write_json(env['ROLLOUT_EVAL_MANIFEST'], e.freeze_queue(source))
        env.pop('ROLLOUT_POLICY_RNG_SEED')
        env['ROLLOUT_COUNT'] = '15'
    env.update(environment(), ROLLOUT_PARALLEL_WORKERS='4')
    env, cases = configure(env)
    root = tmp_path/'code'
    (root/'hybrid_rollout/robodojo').mkdir(parents=True)
    (root/'hybrid_rollout/robodojo/sample.py').write_text('# fixture\n')
    env.update(CODE_ROOT=str(root), ROLLOUT_RESUME_IDENTITY=json.dumps(
        dict(source_sha256=source_identity(root))))
    environments = p.slots(env)
    assert len(environments) == 4
    assert [(s['ROLLOUT_SIM_SSH_PORT'], s['SIM_GPU']) for s in environments] == [
        ('4205', '0'), ('5727', '0'), ('4205', '1'), ('5727', '1')]
    assert len({s['POLICY_PORT'] for s in environments}) == 4
    assert not any(s.get('ROLLOUT_WORKER_CONTROL_FILE') for s in environments)
    monkeypatch.setattr(p, 'AccountPool', lambda env: SimpleNamespace())
    monkeypatch.setattr(p, 'SUPERVISOR_POLL_SECONDS', .01)
    def fake_run(env, cases, *, pool, job_source, allocate_attempt):
        rows = []
        for index, (repeat, case) in job_source():
            rows.append((index, allocate_attempt(), repeat, case['case_id']))
            time.sleep(.02)
        (tmp_path/f'claimed_{os.getpid()}.json').write_text(json.dumps(rows))
        return 0
    monkeypatch.setattr(p, 'run_pool', fake_run)
    assert p.run_parallel(environments, cases) == 0
    rows = [r for path in tmp_path.glob('claimed_*.json') for r in json.loads(path.read_text())]
    assert len(rows) == len({r[3] for r in rows}) == len({r[1] for r in rows}) == (15 if unified else 5)
    assert all(r[2] is None for r in rows)
    with pytest.raises(ValueError, match='account pool'):
        configure(dict(env, ROLLOUT_ACCOUNT_POOL_FILE=''))
