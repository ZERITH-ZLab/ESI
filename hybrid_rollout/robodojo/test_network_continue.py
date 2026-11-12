import copy
import json

import pytest

from .io import InputError
from .skill import run
from .skill.network_recovery import closed_network_turn, is_network_error
from .skill.network_recovery import probe_failed_turn, CodexTransportRecoveryError

NET = {'message': 'stream disconnected before completion: network error: error decoding response body'}
COMPACT_NET = dict(message='Error running remote compact task: Connection failed: error sending request',
    codexErrorInfo={'httpConnectionFailed': {'httpStatusCode': None}}, additionalDetails=None,
    misalignment=None)


def event(method, turn_id='t0', **fields):
    return dict(method=method, params=dict(threadId='thread', turnId=turn_id, **fields))


def completed(turn='t0', status='failed', error=NET):
    return event('turn/completed', turn, turn=dict(id=turn, status=status, error=error))


class Transport:
    def __init__(self, events, snapshot=None):
        self.events = list(events); self.requests = []; self.replies = []; self.turns = 0
        self.snapshot = snapshot

    def request(self, method, params, timeout):
        self.requests.append((method, params))
        if method == 'thread/read': return self.snapshot
        assert method == 'turn/start'
        identity = f't{self.turns}'; self.turns += 1
        return dict(turn=dict(id=identity))

    def next_message(self, timeout):
        assert self.events, 'Unexpected wait or duplicate continue'
        e = self.events.pop(0)
        if isinstance(e, BaseException): raise e
        return e

    def reply(self, identifier, result): self.replies.append((identifier, result))


class Rollout:
    phase = 'execute'; tick = 45; counters = {}; observation_path = '/evidence/observation.json'

    def __init__(self, reject=False): self.actions = 0; self.reject = reject
    def next_call(self): return dict(tool='robodojo_execute', proposal_path='/evidence/actions.npz')
    def start(self, **kw): raise AssertionError('No simulator reset')
    def infer(self, **kw): raise AssertionError('No repeated inference')
    def execute(self, **kw):
        if self.reject: raise InputError('Recovery EEF target exceeds 5 cm')
        self.actions += 1; self.tick += 1; self.phase = 'done'
        return dict(step_id=self.tick, rollout_finished=True)


def call(turn='t1', identity=9):
    e = event('item/tool/call', turn, callId=str(identity), tool='robodojo_execute', arguments={})
    return dict(e, id=identity)


def worker(tmp_path, monkeypatch, events, snapshot=None):
    monkeypatch.setattr(run, 'NETWORK_CONTINUE_DELAYS', (0,) * 20)
    w = object.__new__(run.CodexPolicy)
    w.workspace = tmp_path; w.agent_workspace = tmp_path; w.thread_id = 'thread'
    w.method = 'pi05_plus_gpt'; w.timeout = 1; w.image_max_edge = 480
    w.transport = Transport(events, snapshot)
    return w


def test_strict_eval_timeout_never_reads_ephemeral_history_or_replays(tmp_path, monkeypatch):
    from .skill_auto_infer import run as auto_run
    monkeypatch.setattr(auto_run, 'next_call', lambda r: r.next_call())
    w = object.__new__(auto_run.CodexPolicy)
    w.workspace = tmp_path/'controller'/'workspace'
    w.workspace.mkdir(parents=True)
    w.agent_workspace = w.workspace
    w.thread_id, w.timeout, w.strict_eval = 'thread', 1, True
    w.transport = Transport([event('error', error=NET, willRetry=True),
                             TimeoutError('Timed out waiting for a Codex turn event')])
    rollout = Rollout()
    with pytest.raises(CodexTransportRecoveryError):
        w.run(rollout)
    assert rollout.actions == 0
    assert [method for method, _ in w.transport.requests] == ['turn/start']


def test_same_thread_continue_after_final_network_error_no_replay(tmp_path, monkeypatch):
    w = worker(tmp_path, monkeypatch, [event('error', error=NET, willRetry=False),
        completed(), event('error', error=NET, willRetry=False), call(),
        completed('t1', 'completed', None)])
    rollout = Rollout(); w.run(rollout)
    assert rollout.actions == 1 and rollout.tick == 46
    turns = [p for m,p in w.transport.requests if m == 'turn/start']
    assert len(turns) == 2 and all(p['threadId'] == 'thread' for p in turns)
    assert all(p['model'] == run.MODEL and p['effort'] == run.EFFORT for p in turns)
    assert turns[1]['input'][0]['text'].startswith('Continue the same rollout.')
    assert '/evidence/observation.json' in turns[1]['input'][0]['text']
    assert json.loads((tmp_path/'network_continue_0000.json').read_text())['physical_actions_replayed'] == 0
    assert len(list(tmp_path.glob('turn_*_completed.json'))) == 2


def test_internal_retry_that_recovers_is_not_interrupted(tmp_path, monkeypatch):
    w=worker(tmp_path,monkeypatch,[event('error',error=NET,willRetry=True),call('t0'),completed('t0','completed',None)])
    w.run(Rollout()); assert w.transport.turns == 1


@pytest.mark.parametrize('error', [dict(message='Usage limit reached'),
    dict(message='Authentication failed'), dict(message='Recovery EEF target exceeds 5 cm')])
def test_nonnetwork_failure_does_not_trigger_continue(tmp_path,monkeypatch,error):
    w=worker(tmp_path,monkeypatch,[event('error',error=error,willRetry=False)])
    with pytest.raises(RuntimeError): w.run(Rollout())
    assert w.transport.turns == 1


def test_operator_interruption_not_woken(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch,[event('error',error=NET,willRetry=True),completed(status='interrupted',error=None)])
    with pytest.raises(RuntimeError): w.run(Rollout())
    assert w.transport.turns == 1


@pytest.mark.parametrize('state', ['idle','active','systemError'])
def test_timeout_probe_only_wakes_confirmed_idle_failed_turn(tmp_path,monkeypatch,state):
    snapshot=dict(thread=dict(id='thread',status=dict(type=state),turns=[dict(id='t0',status='failed',error=NET)]))
    events=[event('error',error=NET,willRetry=False),TimeoutError('missing terminal notification')]
    if state=='idle':events += [call(),completed('t1','completed',None)]
    w=worker(tmp_path,monkeypatch,events,snapshot)
    if state=='idle':w.run(Rollout()); assert w.transport.turns==2
    else:
        with pytest.raises(TimeoutError):w.run(Rollout())
        assert w.transport.turns==1


def test_twenty_consecutive_wakeups_then_existing_failure_path(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch,[completed(f't{i}') for i in range(21)])
    with pytest.raises(RuntimeError,match='exhausted'):w.run(Rollout())
    assert w.transport.turns==21  # Original turn plus exactly twenty continues.
    assert len(list(tmp_path.glob('network_continue_*.json')))==20


def test_native_done_never_uses_an_extra_model_call(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch,[call('t0'),event('error',error=NET,willRetry=False)])
    rollout=Rollout();w.run(rollout)
    assert rollout.actions==1 and w.transport.turns==1


def test_rejections_do_not_restart_or_consume_network_continue_budget(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch,[call('t0',i) for i in range(12)] +
        [completed('t0')] + [completed(f't{i}') for i in range(1,21)])
    rollout=Rollout(reject=True)
    with pytest.raises(RuntimeError,match='network continue exhausted'):w.run(rollout)
    assert rollout.actions==0 and w.transport.turns==21
    assert len(w.transport.replies)==12
    assert len(list(tmp_path.glob('call_*_result.json')))==12


def test_network_classifier_and_snapshot_identity():
    assert is_network_error(dict(message='Error running remote compact task: stream disconnected before completion'))
    assert not is_network_error(dict(message='network error: quota exhausted'))
    assert closed_network_turn(dict(thread=dict(id='another')), 'thread','t0') is None


@pytest.mark.parametrize('error', [COMPACT_NET,
    dict(message='Connection refused (os error 111)'),
    dict(message='Connection failed: error sending request'),
    dict(message='Request failed', codexErrorInfo={'httpConnectionFailed': {'httpStatusCode': None}})])
def test_connection_failure_recognition(error):
    assert is_network_error(error)


@pytest.mark.parametrize('status', [400, 401, 403, 404, 422, 429])
def test_http_client_failures_not_reclassified_as_network(status):
    error = copy.deepcopy(COMPACT_NET)
    error['codexErrorInfo']['httpConnectionFailed']['httpStatusCode'] = status
    assert not is_network_error(error)


@pytest.mark.parametrize('message', ['quota exhausted', 'Authentication failed',
    'permission denied', 'invalid_api_key', 'Recovery EEF target exceeds 5 cm'])
def test_connection_marker_does_not_override_exclusions(message):
    assert not is_network_error(dict(COMPACT_NET, message=message+'; error sending request'))


def test_ephemeral_probe_never_reads_turn_history_or_starts_duplicate_turn():
    transport=Transport([])
    with pytest.raises(CodexTransportRecoveryError):
        probe_failed_turn(transport,'thread','t0',1,is_network_error,
                          ephemeral=True,network_failure=True)
    assert transport.requests==[]


def test_ephemeral_non_network_timeout_is_not_mislabeled_as_connection():
    transport=Transport([])
    with pytest.raises(TimeoutError):
        probe_failed_turn(transport,'thread','t0',1,is_network_error,ephemeral=True)
    assert transport.requests==[]


def test_persistent_probe_still_requires_matching_idle_failed_turn():
    snapshot=dict(thread=dict(id='thread',status=dict(type='idle'),
                  turns=[dict(id='t0',status='failed',error=NET)]))
    transport=Transport([],snapshot)
    assert probe_failed_turn(transport,'thread','t0',1,is_network_error)['id']=='t0'
    assert transport.requests==[('thread/read',dict(threadId='thread',includeTurns=True))]
