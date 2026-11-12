import multiprocessing as mp
import os
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from .parallel_collection import Allocation, slots


def environment():
    return dict(ROLLOUT_PARALLEL_WORKERS='2', ROLLOUT_ACCOUNT_POOL_FILE='pool.json',
                ROLLOUT_SIM_HOST='host', ROLLOUT_SIM_SSH_PORT='4205', SIM_GPU='0',
                POLICY_GPU='0', POLICY_PORT='18830', SIM_PORT='19113', ROLLOUT_SIM_REMOTE_PORT='19113')


def test_independent_slots_and_no_mutation():
    env = environment()
    first, second = slots(env)
    assert second['ROLLOUT_SIM_SSH_PORT'] == '5727'
    assert second['POLICY_PORT'] == '18831' and second['SIM_PORT'] == '19114'
    assert first['POLICY_GPU'] == second['POLICY_GPU'] == '0'
    assert 'ROLLOUT_WORKER_ID' not in env
    assert slots(dict(env, ROLLOUT_PARALLEL_WORKERS='1')) == [dict(env, ROLLOUT_PARALLEL_WORKERS='1')]


@pytest.mark.parametrize('extra', [dict(ROLLOUT_SECOND_POLICY_PORT='18830'),
    dict(ROLLOUT_SECOND_SIM_PORT='18830'), dict(ROLLOUT_SECOND_SIM_SSH_PORT='4205'),
    dict(ROLLOUT_PARALLEL_WORKERS='8'), dict(ROLLOUT_ACCOUNT_POOL_FILE='')])
def test_reject_unsafe_configuration(extra):
    with pytest.raises(ValueError): slots(dict(environment(), **extra))


def consume(allocation, output):
    output.put([(i, allocation.allocate_attempt()) for i, job in allocation.source()])


def test_processes_claim_each_job_once_and_attempts_unique():
    ctx = mp.get_context('fork')
    allocator = Allocation(ctx, list(range(100)), 7)
    output = ctx.Queue()
    processes = [ctx.Process(target=consume, args=(allocator, output)) for _ in range(2)]
    for p in processes: p.start()
    rows = output.get(timeout=10) + output.get(timeout=10)
    for p in processes:
        p.join(timeout=10)
        assert p.exitcode == 0
    assert sorted(i for i, _ in rows) == list(range(100))
    assert sorted(a for _, a in rows) == list(range(7,107))
    allocator.stop.set()
    assert list(allocator.source()) == []


def test_dynamic_slots_and_control_validation(tmp_path):
    from .parallel_collection import read_desired
    env = dict(environment(), ROLLOUT_WORKER_CONTROL_FILE=str(tmp_path/'control.json'))
    configured = slots(env)
    assert len(configured) == 8
    assert [e['SIM_GPU'] for e in configured] == ['0','0','1','1','0','0','1','1']
    assert len({e['POLICY_PORT'] for e in configured}) == 8
    assert len({(e['ROLLOUT_SIM_SSH_PORT'],e['ROLLOUT_SIM_REMOTE_PORT']) for e in configured}) == 8
    path = tmp_path/'control.json'
    for n in (0,1,4,8,9):
        path.write_text(json.dumps(dict(desired_workers=n)))
        assert read_desired(path) == n
    for n in (-1,True,'4'):
        path.write_text(json.dumps(dict(desired_workers=n)))
        with pytest.raises(ValueError): read_desired(path)
    for value in (None, [], 4):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError): read_desired(path)
    for value in (None, [], 4):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError): read_desired(path)


def test_retirement_returns_unstarted_job_not_finished_one():
    ctx = mp.get_context('fork')
    allocator = Allocation(ctx, ['a','b'], 0)
    retire = ctx.Event()
    source = allocator.source(retire)
    assert next(source) == (0,'a')
    retire.set()
    source.close()  # Account acquisition cancelled, no completion ACK.
    assert list(allocator.states) == [0,0]
    retire.clear()
    source = allocator.source(retire)
    assert next(source) == (0,'a')
    retire.set()
    assert list(source) == []  # Current episode completed before shrinking.
    assert list(allocator.states) == [2,0]
    assert list(allocator.source()) == [(1,'b')]
    assert allocator.complete()


@pytest.mark.parametrize('dynamic', [False, True])
def test_parent_and_two_workers_finish_full_queue(tmp_path, monkeypatch, dynamic):
    from . import parallel_collection as p
    from .test_collection import setup
    from .dual_campaign import configure
    from .deployment import source_identity
    _, _, env = setup(tmp_path)
    env, cases = configure(env)
    root = tmp_path/'code'
    (root/'hybrid_rollout/robodojo').mkdir(parents=True)
    (root/'hybrid_rollout/robodojo/sample.py').write_text('# fixture\n')
    env.update(environment(), CODE_ROOT=str(root),
               ROLLOUT_RESUME_IDENTITY=json.dumps(dict(source_sha256=source_identity(root))))
    if dynamic:
        control = tmp_path/'control.json'
        control.write_text(json.dumps(dict(desired_workers=2)))
        env['ROLLOUT_WORKER_CONTROL_FILE'] = str(control)
        monkeypatch.setattr(p, 'CONFIG_POLL_SECONDS', .01)
        monkeypatch.setattr(p, 'SUPERVISOR_POLL_SECONDS', .01)
    monkeypatch.setattr(p, 'AccountPool', lambda env: SimpleNamespace())
    def fake_run(env, cases, *, pool, job_source, allocate_attempt):
        rows = []
        for index, (repeat, case) in job_source():
            rows.append((index, allocate_attempt(), repeat, case['case_id']))
            if dynamic and index in (5,25,50):
                from .io import write_json
                write_json(control, dict(desired_workers={5:8,25:1,50:4}[index]))
            time.sleep(.01 if dynamic else .001)
        Path(tmp_path/f'{env["ROLLOUT_WORKER_ID"]}_{os.getpid()}.json').write_text(json.dumps(rows))
        return 0
    monkeypatch.setattr(p, 'run_pool', fake_run)
    assert p.run_parallel(p.slots(env), cases) == 0
    rows = [row for path in tmp_path.glob('worker_*.json') for row in json.loads(path.read_text())]
    assert len(rows) == 100
    assert len({(r[2],r[3]) for r in rows}) == 100
    assert len({r[1] for r in rows}) == 100


@pytest.mark.parametrize('stopped,desired,active,expected', [
    (True,8,True,'draining_after_error'), (True,0,True,'draining_after_error'),
    (True,8,False,'failed'), (False,8,True,'running'),
    (False,0,True,'draining'), (False,0,False,'paused')])
def test_dispatch_status_never_hides_error(stopped,desired,active,expected):
    from .parallel_collection import dispatch_state
    assert dispatch_state(stopped,desired,active)==expected


def test_worker_error_reports_draining_while_peer_finishes(tmp_path, monkeypatch):
    from . import parallel_collection as p
    from .test_collection import setup
    from .dual_campaign import configure
    from .deployment import source_identity
    _, _, env = setup(tmp_path)
    env, cases = configure(env)
    root = tmp_path/'code'
    (root/'hybrid_rollout/robodojo').mkdir(parents=True)
    (root/'hybrid_rollout/robodojo/sample.py').write_text('# fixture\n')
    env.update(environment(), CODE_ROOT=str(root),
               ROLLOUT_RESUME_IDENTITY=json.dumps(dict(source_sha256=source_identity(root))))
    monkeypatch.setattr(p, 'SUPERVISOR_POLL_SECONDS', .01)
    monkeypatch.setattr(p, 'AccountPool', lambda env: SimpleNamespace())
    ready = mp.get_context('fork').Event()
    def fake_run(env,cases,*,pool,job_source,allocate_attempt):
        for index, job in job_source():
            if index==0:
                assert ready.wait(10)
                raise RuntimeError('Task startup rejected')
            assert index==1  # No further dispatch after the other worker fails.
            ready.set()
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                paths=list((tmp_path/'results/test/_parallel').glob('*/status.json'))
                if paths:
                    state=json.loads(paths[0].read_text())
                    if state.get('dispatch_stopped'):
                        assert state['state']=='draining_after_error'
                        (tmp_path/'peer_finished').write_text('normal completion')
                        break
                time.sleep(.02)
            else:raise AssertionError('Missing live failure status')
        return 0
    monkeypatch.setattr(p, 'run_pool', fake_run)
    assert p.run_parallel(p.slots(env), cases)==1
    assert (tmp_path/'peer_finished').exists()
    status=json.loads(next((tmp_path/'results/test/_parallel').glob('*/status.json')).read_text())
    assert status['state']=='failed' and status['dispatch_stopped'] is True
