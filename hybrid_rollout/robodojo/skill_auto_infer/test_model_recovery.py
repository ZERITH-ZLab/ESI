import json
import pytest
from . import run
from .model_recovery import DELAYS, eligible, ModelRecoveryExhausted
from .test_cycle import Rollout
from ..test_network_continue import Transport, event, completed, COMPACT_NET

BAD = dict(message='{"detail":"Bad Request"}', codexErrorInfo='other')


def worker(tmp_path, events, snapshot=None):
    w = object.__new__(run.CodexPolicy)
    w.workspace = w.agent_workspace = tmp_path
    w.thread_id = 'thread'
    w.method, w.timeout, w.image_max_edge = 'pi05_plus_gpt', 1, 480
    w.transport = Transport(events, snapshot)
    return w


@pytest.mark.parametrize('error', [BAD, COMPACT_NET])
def test_ten_retries_and_no_action_replay(tmp_path, monkeypatch, error):
    assert DELAYS == (60,) * 10
    sleeps = []
    monkeypatch.setattr(run.time, 'sleep', sleeps.append)
    w = worker(tmp_path, [completed(f't{i}', error=error) for i in range(11)])
    r = Rollout()
    with pytest.raises(ModelRecoveryExhausted):
        w.run(r)
    assert w.transport.turns == 11 and r.calls == []
    assert sum(sleeps) == 600
    records = list(tmp_path.glob('model_continue_*.json'))
    assert len(records) == 10
    assert all(json.loads(p.read_text())['physical_actions_replayed'] == 0 for p in records)


def test_active_turn_not_restarted(tmp_path):
    snapshot = dict(thread=dict(id='thread', status=dict(type='active'),
        turns=[dict(id='t0', status='failed', error=BAD)]))
    w = worker(tmp_path, [event('error', error=BAD, willRetry=False), TimeoutError()], snapshot)
    with pytest.raises(TimeoutError):
        w.run(Rollout())
    assert w.transport.turns == 1


@pytest.mark.parametrize('error', [dict(message='invalid schema', codexErrorInfo='other'),
    dict(message='usage limit exceeded', codexErrorInfo='other'),
    dict(message='unauthorized', codexErrorInfo='other'),
    dict(message='self collision', codexErrorInfo='other')])
def test_not_allowlisted(error):
    assert not eligible(error)


def test_failure_marker():
    from ..account_pool import failure_kind
    assert failure_kind('hybrid_rollout.robodojo.skill_auto_infer.model_recovery.'
        'ModelRecoveryExhausted: Ten same-thread model recovery attempts exhausted') == 'model_recovery_exhausted'
    assert failure_kind('RuntimeError: Bad Request') is None


def test_compact_connection_failure_resumes_after_ack_without_replay(tmp_path, monkeypatch):
    from .output_contract import decision_id
    monkeypatch.setattr(run.time, 'sleep', lambda _: None)
    def execute(turn, identifier):
        return dict(event('item/tool/call', turn, callId=str(identifier),
            tool='robodojo_execute', arguments=dict(response=dict(
                decision_id=decision_id('fresh'), mode='student', steps=3,
                assessment={}, reason='aligned'))), id=identifier)
    w = worker(tmp_path, [execute('t0', 1),
        event('error', error=COMPACT_NET, willRetry=True),
        event('error', error=COMPACT_NET, willRetry=False),
        completed(error=COMPACT_NET), execute('t1', 2), completed('t1', 'completed', None)])
    r = Rollout()
    r.phase = 'execute'
    original_execute = r.execute
    def finish_on_second(**kwargs):
        r.terminal = r.tick == 3
        return original_execute(**kwargs)
    r.execute = finish_on_second
    w.run(r)
    assert r.calls == ['execute', 'infer', 'execute'] and r.tick == 6
    assert w.transport.turns == 2
    record = json.loads((tmp_path/'model_continue_0000.json').read_text())
    assert record['step_id'] == 3 and record['physical_actions_replayed'] == 0
