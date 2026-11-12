import json
from types import SimpleNamespace

import pytest

from . import abandon as a
from .output_contract import decision_id
from ..io import InputError, write_json


def row(mode, steps):
    return dict(response=dict(mode=mode), executed_steps=steps)


def test_gate_exact_threshold_latches_and_counts_actual_steps_only():
    assert not a.gate([row('eef',119)])['unlocked']
    history = [row('eef',60), row('edit',60)]
    assert a.gate(history)['unlocked']
    history += [row('student',15)]
    assert a.gate(history)['unlocked']
    assert a.gate(history)['consecutive_correction_steps'] == 0
    assert not a.gate([row('eef',60),row('student',1),row('edit',60)])['unlocked']
    assert a.gate([row('eef',60),row('eef',0),row('student',0),row('edit',60)])['unlocked']
    assert not a.gate([dict(response=dict(mode='eef',steps=120),executed_steps=1)])['unlocked']


def args():
    return dict(decision_id=decision_id('fresh'),reason='Only the failed flip remains',
                no_progress_evidence='Several adjustments did not change the face',unresolved=['Flip digit'])


def rollout(tmp_path, steps=120):
    r = SimpleNamespace(history=[row('eef',steps)],phase='execute',request=dict(request_id='fresh'),
        episode='episode',tick=steps,output=tmp_path,counters={},calls=[])
    def finish(reason):
        r.calls.append(reason); r.phase='done'
        return dict(reason=reason,success=False,complete=False,step_id=steps)
    r.finish=finish
    return r


def test_abandon_no_physics_and_no_replay(tmp_path):
    r=rollout(tmp_path)
    packet=a.execute(r,args())
    assert r.calls == ['teacher_abandon']
    assert packet['rollout_finished'] and packet['next_call'] is None
    assert packet['result']['collection_complete'] and not packet['result']['complete']
    assert json.loads((tmp_path/'abandon.json').read_text())['status']=='acknowledged'
    with pytest.raises(InputError): a.execute(r,args())
    assert len(r.calls)==1


@pytest.mark.parametrize('kind',['locked','stale','empty','extras'])
def test_reject_before_finish(tmp_path,kind):
    r=rollout(tmp_path,119 if kind=='locked' else 120); arguments=args()
    if kind=='stale':arguments['decision_id']='old'
    if kind=='empty':arguments['unresolved']=[]
    if kind=='extras':arguments['force']=True
    with pytest.raises(InputError):a.execute(r,arguments)
    assert not r.calls and not (tmp_path/'abandon.json').exists()


def test_sim_outcome_is_failed_sample_not_native_completion(tmp_path):
    from ..test_native_evaluation import NativeLikeEnv
    from ..robodojo_server.session import RoboDojoSession
    env=NativeLikeEnv();env.reward_manager=SimpleNamespace(get_score=lambda:[25])
    session=RoboDojoSession(env,tmp_path,'arrange_largest_number')
    session.episode_id='e';session.poisoned=False;session.step_id=120
    session.finish_reason='teacher_abandon'
    session._write_summary('teacher_abandon')
    outcome=json.loads((tmp_path/'evaluation_outcome.json').read_text())
    assert outcome['status']=='teacher_abandoned'
    assert not outcome['complete'] and not outcome['task_success']
    assert outcome['native_score'] is None and outcome['partial_score_at_abandon']==.25
    assert not outcome['valid_for_success_rate'] and not env.events
    result=a.execute(rollout(tmp_path),args())['result']
    assert a.acknowledged(result,outcome)
    assert not a.acknowledged(dict(result,complete=True),outcome)
    assert not a.acknowledged(result,dict(outcome,reason='controller_error'))
    session._write_summary('server_close')
    assert json.loads((tmp_path/'evaluation_outcome.json').read_text())['reason']=='teacher_abandon'


def test_collection_resume_accepts_only_acknowledged_abandonment(tmp_path):
    from ..account_pool import collection_completed
    r=rollout(tmp_path)
    result=a.execute(r,args())['result']
    outcome=dict(reason='teacher_abandon',status='teacher_abandoned',task_success=False,complete=False)
    runtime=dict(teacher_abandon=dict(version=a.VERSION))
    receipt=dict(all_owned_processes_exited=True,ports_released=True)
    (tmp_path/'dual_machine').mkdir()
    (tmp_path/'dual_machine/remote_cleanup_status.txt').write_text('0')
    root=tmp_path/'replica';(root/'attempt_0').mkdir(parents=True)
    item=dict(case_id='case',repeat_id=1,archive=str(tmp_path),reset=receipt,state='completed',
              artifact_status='verified',result=result,evaluation_outcome=outcome)
    write_json(root/'attempt_0/batch.json',dict(resume_identity=runtime,episodes=[item],final_reset=receipt))
    assert collection_completed(root,runtime,'case',1)
    assert not collection_completed(root,runtime,'case',2)


def test_default_closed_and_identity_changes():
    from ..dual_campaign import controller_route_identity
    assert not a.enabled({})
    env=dict(ROLLOUT_ENABLE_TEACHER_ABANDON='1',ROLLOUT_CONTROLLER_ROUTE='auto_infer_v1',
             ROLLOUT_EVALUATION_METHOD='pi05_plus_gpt')
    assert controller_route_identity(env)['teacher_abandon']['correction_streak_steps']==120
    with pytest.raises(ValueError):a.enabled(dict(env,ROLLOUT_CONTROLLER_ROUTE='baseline'))


def test_controller_dispatches_abandon_without_infer_or_motion(tmp_path):
    from . import run
    from ..test_network_continue import Transport, event, completed
    call = event('item/tool/call', callId='1', tool='robodojo_abandon', arguments=args())
    call['id'] = 1
    worker = object.__new__(run.CodexPolicy)
    worker.workspace = worker.agent_workspace = tmp_path/'codex_workspace'
    worker.workspace.mkdir()
    worker.thread_id = 'thread'
    worker.method, worker.timeout, worker.image_max_edge = 'pi05_plus_gpt', 1, 480
    worker.abandon_enabled = True
    worker.transport = Transport([call, completed('t0', 'completed', None)])
    r = rollout(tmp_path)
    r.next_call = lambda: None
    r.history[0]['decision'] = 0
    worker.run(r)
    assert r.calls == ['teacher_abandon'] and r.phase == 'done'
    assert len(worker.transport.replies) == 1
