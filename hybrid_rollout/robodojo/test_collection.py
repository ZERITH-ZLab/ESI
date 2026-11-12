"""Offline protocol, budget and repeat/resume tests; no paid inference."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from . import collection as c
from .account_pool import run_pool
from .deployment import source_identity
from .dual_campaign import configure
from .evaluation import read_panel, verify_assets
from .io import write_json
from .robodojo_server.step_budget import apply
from .test_evaluation import fake_native


def setup(tmp_path):
    source = fake_native(tmp_path)
    for row in c.specs():
        path = source/'Assets/Eval_Layout/RoboDojo/arx_x5/0'/f'{row["runtime_task"]}_{row["layout_id"]}.json'
        if not path.exists():
            path.write_text('{}')
    panel = c.freeze(source)
    path = tmp_path/'collection.json'
    write_json(path, panel)
    env = dict(ROLLOUT_EVAL_MANIFEST=str(path), ROBODOJO_TASK='arrange_largest_number',
        ROLLOUT_COUNT='20', ROLLOUT_EXPERIMENT_ID='test', ROLLOUT_RUN_MODE='fresh',
        ROLLOUT_ATTEMPT='auto', ROLLOUT_SHARED_ROOT=str(tmp_path),
        ROLLOUT_COLLECTION_STEP_LIMIT='1500', ROLLOUT_MAX_SECONDS='10800',
        ROLLOUT_ACCOUNT_POOL_FILE=str(tmp_path/'pool.json'), ROLLOUT_AUTH_ROOT=str(tmp_path/'auth'),
        ROLLOUT_CONTROLLER_ROUTE='auto_infer_v1', ROLLOUT_EVALUATION_METHOD='pi05_plus_gpt',
        ROLLOUT_ENABLE_SCENE_QUERY='1', ROLLOUT_ENABLE_MOTION_TOOLS='1')
    return source, panel, env


def test_frozen_cases_and_native_budget(tmp_path):
    source, panel, env = setup(tmp_path)
    verify_assets(read_panel(env['ROLLOUT_EVAL_MANIFEST']), source)
    actual, cases = configure(env)
    assert actual['ROLLOUT_REPLICA_ID'] == '3'
    assert [(r['variant'], r['layout_id']) for r in cases] == (
        [('standard', i) for i in range(15)] + [('random', i) for i in range(5)])
    native = SimpleNamespace(step_lim=1050)
    assert apply(native)['effective_step_limit'] == 1050
    budget = apply(native, 1500, panel)
    assert native.step_lim == 1500 and budget['original_native_step_limit'] == 1050
    with pytest.raises(ValueError): apply(SimpleNamespace(step_lim=1050), 1500)
    with pytest.raises(ValueError): apply(SimpleNamespace(step_lim=1050), 0, panel)
    with pytest.raises(ValueError): apply(SimpleNamespace(step_lim=1000), 1500, panel)
    with pytest.raises(ValueError): c.runtime_settings(dict(ROLLOUT_COLLECTION_STEP_LIMIT='1500'), {})
    with pytest.raises(ValueError): configure(dict(env, ROLLOUT_COUNT='5'))


def test_three_repeat_protocol_is_independent(tmp_path):
    source, original, env = setup(tmp_path)
    panel = c.freeze(source, repeats=3)
    assert original['collection']['repeats'] == 5
    assert c.PROTOCOL['repeats'] == 5
    assert panel['panel_sha256'] != original['panel_sha256']
    assert panel['cases'] == original['cases']
    assert c.runtime_settings(env, panel)['collection_protocol']['repeats'] == 3
    write_json(tmp_path/'three.json', panel)
    assert read_panel(tmp_path/'three.json')['collection']['repeats'] == 3
    for invalid in (0, -1, True, 3.5):
        with pytest.raises(ValueError): c.protocol(invalid)


@pytest.mark.parametrize('abandon_one', [False, True])
def test_repeat_major_quota_retry_timeout_and_resume(tmp_path, abandon_one):
    _, panel, env = setup(tmp_path)
    env, cases = configure(env)
    root = tmp_path/'repo'
    (root/'hybrid_rollout/robodojo').mkdir(parents=True)
    (root/'hybrid_rollout/robodojo/sample.py').write_text('# frozen\n')
    runtime = dict(source_sha256=source_identity(root), collection_protocol=c.PROTOCOL)
    if abandon_one:
        from .skill_auto_infer.abandon import VERSION
        runtime['teacher_abandon'] = dict(version=VERSION)
    env.update(CODE_ROOT=str(root), ROLLOUT_RESUME_IDENTITY=json.dumps(runtime))
    class Pool:
        config_path = tmp_path/'pool.json'
        blocked = []
        @contextmanager
        def acquire(self, job, on_wait=None):
            yield dict(auth_profile='codex_a_3', account_fingerprint='fixture',
                       credential_revision='fixture', fds=(1, 2))
        def block(self, *args): self.blocked.append(args)
    pool = Pool()
    attempted = []
    local_attempts = []
    def runner(child, fds):
        repeat = int(child['ROLLOUT_COLLECTION_REPEAT_ID'])
        local_attempts.append(int(child['ROLLOUT_SAMPLE_ATTEMPT']))
        attempted.append((repeat, child['ROLLOUT_CASE_ID']))
        n = int(child['ROLLOUT_ATTEMPT'])
        archive = tmp_path/f'archive_{n}'
        (archive/'dual_machine').mkdir(parents=True)
        (archive/'dual_machine/remote_cleanup_status.txt').write_text('0')
        receipt = dict(all_owned_processes_exited=True, ports_released=True)
        reason = 'quota_interrupted' if n == 0 else 'wall_time_limit' if n == 2 else None
        row = dict(case_id=child['ROLLOUT_CASE_ID'], repeat_id=repeat,
                   archive=str(archive), reset=receipt, failure_kind=reason,
                   state='failed' if reason else 'completed', artifact_status='verified',
                   result=dict(complete=not reason), evaluation_outcome=dict(complete=not reason))
        if abandon_one and n == 3:
            row.update(failure_kind='teacher_abandon', result=dict(
                reason='teacher_abandon', termination_reason='teacher_abandon',
                complete=False, success=False, task_success=False, collection_complete=True,
                abandon_gate=dict(version=VERSION, unlocked=True, max_consecutive_correction_steps=120)),
                evaluation_outcome=dict(reason='teacher_abandon', complete=False,
                    status='teacher_abandoned', task_success=False))
        path = tmp_path/f'results/test/_replicas/replica_3/attempt_{n}'
        path.mkdir(parents=True)
        write_json(path/'batch.json', dict(final_reset=receipt, episodes=[row], resume_identity=runtime))
        return 1 if n == 0 else 124 if reason else 0
    assert run_pool(env, cases, pool=pool, runner=runner) == 124
    expected = [(r, case['case_id']) for r in range(1, 6) for case in cases]
    assert attempted == [expected[0]] + expected
    assert local_attempts == [0, 1] + [0] * 99
    from .evaluation import archive_path
    sample = archive_path(tmp_path, 'test', cases[0], 3, 0, 1).parent
    summary = json.loads((sample/'sample.json').read_text())
    assert summary['repeat_id'] == 1 and summary['state'] == 'completed'
    assert [a['sample_attempt'] for a in summary['attempts']] == [0, 1]
    assert [a['batch_attempt'] for a in summary['attempts']] == ['0', '1']
    assert len(pool.blocked) == 1
    attempted.clear()
    assert run_pool(dict(env, ROLLOUT_RUN_MODE='resume'), cases, pool=pool, runner=runner) == 124
    assert not attempted  # Includes the timeout, but never counts quota interruption.


def test_model_episode_retry_then_continue_and_resume(tmp_path):
    _, _, env = setup(tmp_path)
    env, cases = configure(env)
    root = tmp_path/'repo'
    (root/'hybrid_rollout/robodojo').mkdir(parents=True)
    (root/'hybrid_rollout/robodojo/sample.py').write_text('# frozen\n')
    runtime = dict(source_sha256=source_identity(root))
    env.update(CODE_ROOT=str(root), ROLLOUT_RESUME_IDENTITY=json.dumps(runtime))
    class Pool:
        config_path = tmp_path/'pool.json'
        @contextmanager
        def acquire(self, job, on_wait=None):
            yield dict(auth_profile='codex_a_3', account_fingerprint='fixture',
                       credential_revision='fixture', fds=(1, 2))
        def block(self, *args):
            pytest.fail('Model failure must not block account')
    attempted = []
    def runner(child, fds):
        n = int(child['ROLLOUT_ATTEMPT'])
        retry = int(child['ROLLOUT_MODEL_EPISODE_RETRY'])
        attempted.append((child['ROLLOUT_CASE_ID'], retry))
        archive = tmp_path/f'archive_{n}'
        (archive/'dual_machine').mkdir(parents=True)
        (archive/'dual_machine/remote_cleanup_status.txt').write_text('0')
        receipt = dict(all_owned_processes_exited=True, ports_released=True)
        failed = child['ROLLOUT_CASE_ID'] == cases[0]['case_id']
        row = dict(case_id=child['ROLLOUT_CASE_ID'], repeat_id=1, archive=str(archive),
            reset=receipt, failure_kind='model_recovery_exhausted' if failed else None,
            model_episode_retry=retry, state='failed' if failed else 'completed',
            artifact_status='invalid' if failed else 'verified',
            result=dict(complete=not failed), evaluation_outcome=dict(complete=not failed))
        path = tmp_path/f'results/test/_replicas/replica_3/attempt_{n}'
        path.mkdir(parents=True)
        write_json(path/'batch.json', dict(final_reset=receipt, episodes=[row], resume_identity=runtime))
        return 1 if failed else 0
    def jobs():
        yield 0, (1, cases[0])
        yield 1, (1, cases[1])
    assert run_pool(env, cases, pool=Pool(), runner=runner, job_source=jobs) == 0
    assert attempted == [(cases[0]['case_id'], 0), (cases[0]['case_id'], 1), (cases[1]['case_id'], 0)]
    attempted.clear()
    assert run_pool(dict(env, ROLLOUT_RUN_MODE='resume'), cases,
                    pool=Pool(), runner=runner, job_source=jobs) == 0
    assert not attempted


def test_session_reports_effective_budget_and_separates_official_success(tmp_path):
    from .test_native_evaluation import NativeLikeEnv
    from .robodojo_server.session import RoboDojoSession
    _, panel, _ = setup(tmp_path)
    env = NativeLikeEnv()
    budget = apply(env, 1500, panel)
    session = RoboDojoSession(env, tmp_path/'sim', 'arrange_largest_number', step_budget=budget)
    assert session.metadata['max_episode_steps'] == 1500
    session.episode_id = 'fixture'
    session.poisoned = False
    session.terminated = session.success = True
    for step in (900, 1300):
        session.step_id = step
        session._write_summary('terminal')
        outcome = json.loads((tmp_path/'sim/evaluation_outcome.json').read_text())
        assert outcome['native_success'] is True
        assert outcome['valid_for_official_success_rate'] is False
        assert outcome['success_within_original_limit'] == (step <= 1050)
        assert outcome['success_after_original_limit'] == (step > 1050)



def test_explicit_three_case_selection_keeps_full_panel(tmp_path):
    from .dual_campaign import collection_selection_identity
    source, original, env = setup(tmp_path)
    panel = c.freeze(source, repeats=1)
    path = tmp_path/'repeat1.json'
    write_json(path, panel)
    ids = [row['case_id'] for row in panel['cases'][:3]]
    env.update(ROLLOUT_EVAL_MANIFEST=str(path), ROLLOUT_CASE_IDS=','.join(ids))
    configured, cases = configure(env)
    assert len(read_panel(path)['cases']) == 20
    assert [row['case_id'] for row in cases] == ids
    assert configured['ROLLOUT_COUNT'] == '3'
    assert panel['collection']['repeats'] == 1
    assert original['collection']['repeats'] == 5
    assert collection_selection_identity(configured, panel) == {'selected_case_ids': ids}
    assert collection_selection_identity(dict(env, ROLLOUT_CASE_IDS=''), panel) == {}
    assert len(cases)*panel['collection']['repeats'] == 3
    with pytest.raises(ValueError, match='COUNT'):
        configure(dict(env, ROLLOUT_CASE_IDS='', ROLLOUT_COUNT='3'))
    for invalid in (ids[0]+','+ids[0], 'unknown', ids[0]+','):
        with pytest.raises(ValueError, match='Duplicate or unknown'):
            configure(dict(env, ROLLOUT_CASE_IDS=invalid))


@pytest.mark.parametrize('receipt_kind', ['batch', 'pool', 'parallel'])
def test_collection_resume_selection_cannot_expand_or_change(tmp_path, receipt_kind):
    _, panel, env = setup(tmp_path)
    ids = [row['case_id'] for row in panel['cases'][:3]]
    base = tmp_path/'results/test'
    paths = dict(batch=base/'_replicas/replica_3/attempt_0/batch.json',
                 pool=base/'_account_pool/run_0/pool_run.json',
                 parallel=base/'_parallel/run_0/status.json')
    identity = dict(model='gpt-6-astra', effort='xhigh', selected_case_ids=ids)
    paths[receipt_kind].parent.mkdir(parents=True, exist_ok=True)
    write_json(paths[receipt_kind], dict(resume_identity=identity))
    # Resume requires a prior allocated attempt even when the only identity is status.
    (base/'_replicas/replica_3/attempt_0').mkdir(parents=True, exist_ok=True)
    resume = dict(env, ROLLOUT_RUN_MODE='resume', ROLLOUT_CASE_IDS=','.join(ids))
    assert len(configure(resume)[1]) == 3
    for changed in ('', ','.join(ids[:2]+[panel['cases'][3]['case_id']])):
        with pytest.raises(ValueError, match='case selection changed'):
            configure(dict(resume, ROLLOUT_CASE_IDS=changed))
    # Legacy full-panel identity remains compatible with default full selection only.
    identity.pop('selected_case_ids')
    write_json(paths[receipt_kind], dict(resume_identity=identity))
    assert len(configure(dict(resume, ROLLOUT_CASE_IDS=''))[1]) == 20
    with pytest.raises(ValueError, match='case selection changed'):
        configure(resume)


def test_collection_fresh_accepts_only_control_but_refuses_prior_attempt(tmp_path):
    _, panel, env = setup(tmp_path)
    base = tmp_path/'results/test'
    base.mkdir(parents=True)
    write_json(base/'worker_control.json', dict(desired_workers=3))
    ids = ','.join(row['case_id'] for row in panel['cases'][:3])
    assert len(configure(dict(env, ROLLOUT_CASE_IDS=ids))[1]) == 3
    (base/'_replicas/replica_3/attempt_0').mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'):
        configure(dict(env, ROLLOUT_CASE_IDS=ids))
