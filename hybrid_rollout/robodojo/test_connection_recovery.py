"""Offline transport/retry/dispatch tests; no simulator, SSH or model calls."""
from contextlib import contextmanager
import json
import multiprocessing as mp

import pytest

from . import account_pool as ap
from . import connection_recovery as cr
from .deployment import source_identity
from .dual_campaign import configure
from .io import write_json
from .robodojo_server import protocol
from .test_collection import setup

ROUTING_FAILURE = "RuntimeError: Codex error: {'error': {'message': 'workspace routing discovery failed', 'codexErrorInfo': 'other'}, 'willRetry': False}"
COMPACTION_FAILURE = 'RuntimeError: Codex error: ' + repr(dict(
    error=dict(message='Error running remote compact task: {"detail":"Bad Request"}',
               codexErrorInfo='other', additionalDetails=None), willRetry=False))
SIM_CAUSE = 'hybrid_rollout.robodojo.robodojo_server.protocol.SimulatorConnectionError: Simulator RPC transport failure during fk_preview'
WRAPPER = 'RuntimeError: Feedforward inference failed; abort without replay or candidate resampling'
WRAPPED_SIM_FAILURE = ('Traceback (most recent call last):\n'+SIM_CAUSE+
    '\n\nThe above exception was the direct cause of the following exception:\n\n'
    'Traceback (most recent call last):\n'+WRAPPER)


@pytest.mark.parametrize('message,expected', [
    (ROUTING_FAILURE, 'codex_transport_failed'),
    (COMPACTION_FAILURE, 'codex_compaction_failed'),
    (COMPACTION_FAILURE.replace('False', 'True'), None),
    (COMPACTION_FAILURE.replace("'other'", "'invalidRequest'"), None),
    (COMPACTION_FAILURE.replace('Bad Request', 'Invalid schema'), None),
    (COMPACTION_FAILURE.replace('Error running remote compact task: ', ''), None),
    (COMPACTION_FAILURE.replace('{"detail":"Bad Request"}', 'Bad Request'), None),
    (COMPACTION_FAILURE+'\nValueError: invalid action', None),
    ('ValueError: '+COMPACTION_FAILURE, None),
    (WRAPPED_SIM_FAILURE, 'simulator_connection_failed'),
    (WRAPPED_SIM_FAILURE.replace(SIM_CAUSE, 'ValueError: invalid subtask'), None),
    (WRAPPED_SIM_FAILURE.replace('direct cause', 'unrelated text'), None),
    (SIM_CAUSE+'\nValueError: IK failed', None),
    (ROUTING_FAILURE.replace('workspace routing discovery failed','authentication failed'), 'auth_interrupted'),
    ('RuntimeError: workspace routing discovery failed', None),
    (ROUTING_FAILURE+'\nValueError: invalid action', None),
])
def test_observed_errors_and_negative_controls(message, expected):
    assert ap.failure_kind(message)==expected


def monitor(jobs=5):
    ctx = mp.get_context('fork')
    return cr.ConnectionRecovery(ctx, ctx.Event(), jobs)


def test_rolling_window_distinct_workers_and_consecutive_reset():
    health = monitor()
    health.failed(0, now=1000)
    health.failed(0, now=1100)
    assert not health.stop.is_set()  # Two attempts on one worker are one worker.
    health.completed(0)  # Native task success OR failure resets the streak.
    health.failed(0, now=1200)
    assert health.snapshot(1200)['consecutive_failures'][0] == 1
    health.failed(1, now=1300)
    health.failed(2, now=2101)  # Worker 0 has fallen outside the 15-minute window.
    assert not health.stop.is_set()
    result = health.failed(3, now=2102)
    assert health.stop.is_set()
    assert result['pause_reason'] == 'multiple_workers_connection_failures'
    health.completed(3)
    assert health.stop.is_set()  # Successful draining episodes must not reopen dispatch.


def fail_in_child(health, index):
    health.failed(index)


def test_circuit_shared_across_processes():
    health = monitor()
    ctx = mp.get_context('fork')
    for index in range(3):
        process = ctx.Process(target=fail_in_child, args=(health, index))
        process.start()
        process.join(timeout=5)
        assert process.exitcode == 0
    assert health.stop.is_set()
    assert health.snapshot()['pause_reason'] == 'multiple_workers_connection_failures'


def test_circuit_drains_claimed_episode_and_keeps_unstarted_jobs_pending():
    from .parallel_collection import Allocation
    ctx = mp.get_context('fork')
    allocation = Allocation(ctx, ['standard_case', 'random_case'], 0)
    active = allocation.source()
    assert next(active) == (0, 'standard_case')
    for worker in range(3):
        allocation.connection_recovery.failed(worker)
    # The claimed episode may finish; no next case is dispatched or lost.
    assert list(active) == []
    assert list(allocation.states) == [2, 0]
    assert list(allocation.source()) == []


def test_invalid_response_is_not_a_transport_failure(monkeypatch):
    class Socket:
        def close(self): pass
    client = protocol.RPCClient.__new__(protocol.RPCClient)
    client.sock = Socket()
    monkeypatch.setattr(protocol, 'send_packet', lambda *args: None)
    monkeypatch.setattr(protocol, 'receive_packet', lambda sock: dict(version=-1))
    with pytest.raises(ValueError, match='identity mismatch'):
        client.request('step')
    assert client.sock is None


@pytest.mark.parametrize('error', [EOFError('RPC connection closed'),
    ConnectionResetError('reset'), BrokenPipeError('pipe'), TimeoutError('timeout')])
def test_rpc_transport_failure_is_typed_and_never_replayed(monkeypatch, error):
    class Socket:
        closed = False
        def close(self): self.closed = True
    client = protocol.RPCClient.__new__(protocol.RPCClient)
    sock = client.sock = Socket()
    sent = []
    monkeypatch.setattr(protocol, 'send_packet', lambda *args: sent.append(args))
    def receive(sock): raise error
    monkeypatch.setattr(protocol, 'receive_packet', receive)
    with pytest.raises(protocol.SimulatorConnectionError) as failure:
        client.request('step')
    assert len(sent) == 1 and sock.closed and client.sock is None
    message = f'{type(failure.value).__module__}.{type(failure.value).__name__}: {failure.value}'
    assert ap.failure_kind(message) == 'simulator_connection_failed'


@pytest.mark.parametrize('message,expected', [
    ('EOFError: RPC connection closed', 'simulator_connection_failed'),
    ('EOFError: end of input', None), ('ConnectionResetError: peer reset', None),
    ('TimeoutError: model timed out', None), ('ValueError: RPC response identity mismatch', None),
    ('RuntimeError: Simulator RPC transport failure during step', None),
])
def test_narrow_failure_classifier(message, expected):
    assert ap.failure_kind(message) == expected


@pytest.mark.parametrize('mode', ['recover', 'defer', 'circuit', 'unclean', 'cancel_wait'])
@pytest.mark.parametrize('kind', ['collection_simulator','eval_simulator','eval_codex_legacy','eval_codex_timeout',
                                'eval_codex_routing','eval_codex_compaction','eval_simulator_wrapped'])
def test_collection_retry_defer_resume_and_cleanup(tmp_path, monkeypatch, mode, kind):
    is_eval = kind.startswith('eval_')
    if is_eval:
        from .test_regression_eval import setup as eval_setup
        _, _, env = eval_setup(tmp_path)
    else:
        _, _, env = setup(tmp_path)
    env, cases = configure(env)
    root = tmp_path/'repo'
    (root/'hybrid_rollout/robodojo').mkdir(parents=True)
    (root/'hybrid_rollout/robodojo/sample.py').write_text('# frozen\n')
    runtime = dict(source_sha256=source_identity(root))
    env.update(CODE_ROOT=str(root), ROLLOUT_RESUME_IDENTITY=json.dumps(runtime),
               ROLLOUT_WORKER_ID='worker_0')
    class Pool:
        config_path = tmp_path/'pool.json'
        connection_recovery = monitor()
        held = False
        @contextmanager
        def acquire(self, job, on_wait=None):
            self.held = True
            try:
                yield dict(auth_profile='codex_a_3', account_fingerprint='fixture',
                           credential_revision='fixture', fds=(1, 2))
            finally:
                self.held = False
        def block(self, *args): pytest.fail('RPC failures must not block GPT accounts')
    pool = Pool()
    waits = []
    def wait(cancelled):
        assert not pool.held
        waits.append(True)
        if mode == 'cancel_wait':
            pool.connection_recovery.stop.set()
    monkeypatch.setattr(cr, 'wait_before_retry', wait)
    attempted = []
    force_success = False
    def runner(child, fds):
        n = int(child['ROLLOUT_ATTEMPT'])
        retry = int(child['ROLLOUT_CONNECTION_EPISODE_RETRY'])
        case_id = child['ROLLOUT_CASE_ID']
        attempted.append((case_id, retry))
        failed = not force_success and (mode == 'circuit' or case_id == cases[0]['case_id'])
        if mode == 'recover' and retry == 1:
            failed = False
        archive = tmp_path/f'archive_{n}'
        (archive/'dual_machine').mkdir(parents=True)
        (archive/'dual_machine/remote_cleanup_status.txt').write_text('1' if mode == 'unclean' else '0')
        receipt = dict(all_owned_processes_exited=True, ports_released=True)
        reason = 'simulator_connection_failed' if failed else None
        if failed and (kind.startswith('eval_codex_') or kind=='eval_simulator_wrapped'):
            reason=None
            (archive/'controller').mkdir()
            message = LEGACY_CODEX_FAILURE if kind=='eval_codex_legacy' else 'TimeoutError: Timed out waiting for a Codex turn event'
            if kind=='eval_codex_routing':message=ROUTING_FAILURE
            if kind=='eval_codex_compaction':message=COMPACTION_FAILURE
            if kind=='eval_simulator_wrapped':message=WRAPPED_SIM_FAILURE
            write_json(archive/'controller/failure.json',dict(error=message))
        row = dict(case_id=case_id, repeat_id=None if is_eval else 1, archive=str(archive), reset=receipt,
                   failure_kind=reason,
                   state='failed' if failed else 'completed',
                   artifact_status='invalid' if failed else 'verified',
                   result=dict(complete=not failed), evaluation_outcome=dict(complete=not failed))
        path = tmp_path/f'results/test/_replicas/replica_3/attempt_{n}'
        path.mkdir(parents=True)
        write_json(path/'batch.json', dict(final_reset=receipt, episodes=[row], resume_identity=runtime))
        return int(failed)
    def jobs():
        yield 0, (None if is_eval else 1, cases[0])
        yield 1, (None if is_eval else 1, cases[1])
    if mode == 'unclean':
        with pytest.raises(RuntimeError, match='cleanup not confirmed'):
            ap.run_pool(env, cases, pool=pool, runner=runner, job_source=jobs)
        assert len(attempted) == 1 and not waits
        return
    assert ap.run_pool(env, cases, pool=pool, runner=runner, job_source=jobs) == 0
    if mode == 'cancel_wait':
        assert len(attempted) == 1
        assert pool.connection_recovery.snapshot()['pending_recollection_job_indices'] == [0]
        return
    assert attempted == [(cases[0]['case_id'], 0), (cases[0]['case_id'], 1), (cases[1]['case_id'], 0)]
    assert len(waits) == 1
    summary = ap.load_json(next((tmp_path/'results/test/_account_pool').glob('*/pool_run.json')))
    if mode == 'circuit':
        assert pool.connection_recovery.stop.is_set()
        assert summary['connection_recovery']['pause_reason'] == 'worker_consecutive_connection_failures'
        assert [j['state'] for j in summary['jobs']] == ['pending_recollection'] * 2
        return
    assert not pool.connection_recovery.stop.is_set()
    assert summary['jobs'][0]['state'] == ('completed' if mode == 'recover' else 'pending_recollection')
    if kind.startswith('eval_codex_'):
        expected_kind = 'codex_compaction_failed' if kind == 'eval_codex_compaction' else 'codex_transport_failed'
        assert summary['jobs'][0]['attempts'][0]['failure_kind']==expected_kind
        assert summary['jobs'][0]['attempts'][0]['failure_kind_source']=='dispatcher_controller_failure'
        if kind == 'eval_codex_compaction' and mode == 'defer':
            assert summary['jobs'][0]['pending_reason'] == 'codex_compaction_retry_exhausted'
        # Reclassification must not alter the frozen episode receipt.
        original = ap.load_json(tmp_path/'results/test/_replicas/replica_3/attempt_0/batch.json')
        assert original['episodes'][0]['failure_kind'] is None
    if is_eval:
        return  # Native archive verification is tested by the eval resume tests.
    attempted.clear()
    force_success = True
    # A new scheduler run retries deferred cases, while preserving completed cases.
    pool.connection_recovery = monitor()
    assert ap.run_pool(dict(env, ROLLOUT_RUN_MODE='resume'), cases, pool=pool,
                       runner=runner, job_source=jobs) == 0
    assert attempted == ([(cases[0]['case_id'], 0)] if mode == 'defer' else [])


LEGACY_CODEX_FAILURE = ("TimeoutError: Timed out waiting for a Codex turn event\n"
    "RuntimeError: Codex RPC thread/read failed: {'code': -32600, 'message': 'ephemeral threads do not support includeTurns'}")


def test_codex_failure_classification_is_narrow_and_legacy_compatible():
    assert ap.failure_kind(LEGACY_CODEX_FAILURE)=='codex_transport_failed'
    assert ap.failure_kind(LEGACY_CODEX_FAILURE.splitlines()[-1]) is None
    assert ap.failure_kind(LEGACY_CODEX_FAILURE+'\nValueError: Invalid action') is None
    assert ap.failure_kind('TimeoutError: Timed out waiting for a Codex turn event') == 'codex_transport_failed'
    from .skill.network_recovery import CodexTransportRecoveryError
    error=CodexTransportRecoveryError('Disconnected ephemeral Codex turn timed out; terminal state unavailable')
    terminal=f'{type(error).__module__}.{type(error).__name__}: {error}'
    assert ap.failure_kind(terminal)=='codex_transport_failed'
