import json
import pytest
from .skill import run
from .test_network_continue import worker, event, completed, call, Rollout

CAPACITY = dict(message='Selected model is at capacity', codexErrorInfo='serverOverloaded')


def test_capacity_waits_for_terminal_then_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'CAPACITY_RETRY_DELAYS', (0,) * 4)
    w = worker(tmp_path, monkeypatch, [event('error', error=CAPACITY, willRetry=False),
        completed(error=CAPACITY), call(), completed('t1', 'completed', None)])
    r = Rollout(); w.run(r)
    assert r.actions == 1 and w.transport.turns == 2
    assert json.loads((tmp_path/'capacity_continue_0000.json').read_text())['physical_actions_replayed'] == 0


def test_five_capacity_failures_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'CAPACITY_RETRY_DELAYS', (0,) * 4)
    w = worker(tmp_path, monkeypatch, [completed(f't{i}', error=CAPACITY) for i in range(5)])
    with pytest.raises(RuntimeError, match='five consecutive'):
        w.run(Rollout())
    assert w.transport.turns == 5


def test_capacity_after_ack_does_not_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'CAPACITY_RETRY_DELAYS', (0,) * 4)
    class TwoActions(Rollout):
        def execute(self, **kw):
            self.actions += 1; self.tick += 1
            self.phase = 'done' if self.actions == 2 else 'execute'
            return dict(step_id=self.tick)
    w = worker(tmp_path, monkeypatch, [call('t0'), completed(error=CAPACITY),
        call('t1', 10), completed('t1', 'completed', None)])
    r = TwoActions(); w.run(r)
    assert r.actions == 2 and r.tick == 47
    assert len(w.transport.replies) == 2


def test_capacity_done_needs_no_extra_call(tmp_path, monkeypatch):
    w = worker(tmp_path, monkeypatch, [call('t0'), event('error', error=CAPACITY, willRetry=False)])
    r = Rollout(); w.run(r)
    assert r.actions == 1 and w.transport.turns == 1


@pytest.mark.parametrize('state', ['idle', 'active'])
def test_capacity_probe_never_wakes_active_turn(tmp_path, monkeypatch, state):
    monkeypatch.setattr(run, 'CAPACITY_RETRY_DELAYS', (0,) * 4)
    snapshot = dict(thread=dict(id='thread', status=dict(type=state),
        turns=[dict(id='t0', status='failed', error=CAPACITY)]))
    events = [event('error', error=CAPACITY, willRetry=False), TimeoutError()]
    if state == 'idle':
        events += [call(), completed('t1', 'completed', None)]
    w = worker(tmp_path, monkeypatch, events, snapshot)
    if state == 'idle':
        w.run(Rollout()); assert w.transport.turns == 2
    else:
        with pytest.raises(TimeoutError): w.run(Rollout())
        assert w.transport.turns == 1
