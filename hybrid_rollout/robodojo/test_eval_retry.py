import json
from pathlib import Path

import pytest

from . import eval_retry as retry
from .io import write_json
from .test_connection_recovery import ROUTING_FAILURE, WRAPPED_SIM_FAILURE, COMPACTION_FAILURE


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path,value)


def fixture(tmp_path, kind=None, *, clean=True, completed=False):
    archive=tmp_path/'archive'
    (archive/'dual_machine').mkdir(parents=True)
    (archive/'dual_machine/remote_cleanup_status.txt').write_text('0' if clean else '1')
    receipt=dict(all_owned_processes_exited=True,ports_released=True)
    row=dict(case_id='case',state='failed',archive=str(archive),reset=receipt,failure_kind=kind)
    batch=dict(episodes=[row],final_reset=receipt)
    put(tmp_path/'results/exp/_replicas/replica_3/attempt_1/batch.json',batch)
    put(archive/'controller/failure.json',dict(error='TimeoutError: Timed out waiting for a Codex turn event'))
    cfg=dict(name='queue',models=[dict(experiment='exp')])
    reports=[dict(experiment='exp',episodes=[dict(case_id='case',state='completed' if completed else 'incomplete')])]
    put(tmp_path/'evaluation/queue/status.json',dict(state='completed_with_timeouts'))
    return cfg,reports


@pytest.mark.parametrize('kind',[None,'wall_time_limit','simulator_connection_failed','codex_transport_failed',
                                'codex_compaction_failed'])
def test_known_failures_require_cleanup(tmp_path,kind):
    cfg,reports=fixture(tmp_path,kind)
    assert retry.assess_queue(tmp_path,cfg,reports)['retryable']
    (tmp_path/'archive/dual_machine/remote_cleanup_status.txt').write_text('1')
    assert not retry.assess_queue(tmp_path,cfg,reports)['retryable']


@pytest.mark.parametrize('message',['ValueError: invalid action','TimeoutError: other timeout',
                                   'RuntimeError: CUDA out of memory'])
def test_unknown_errors_not_retried(tmp_path,message):
    cfg,reports=fixture(tmp_path)
    put(tmp_path/'archive/controller/failure.json',dict(error=message))
    assert not retry.assess_queue(tmp_path,cfg,reports)['retryable']


def test_completed_native_failure_not_retried_or_blocked_by_old_error(tmp_path):
    cfg,reports=fixture(tmp_path,completed=True)
    plan=retry.assess_queue(tmp_path,cfg,reports)
    assert plan==dict(retryable=True,reasons={},blocked=[])


@pytest.mark.parametrize('state',['interrupted','running','paused','draining_after_error'])
def test_operator_stop_or_nonterminal_queue_not_resumed(tmp_path,state):
    cfg,reports=fixture(tmp_path)
    put(tmp_path/'evaluation/queue/status.json',dict(state=state))
    assert not retry.assess_queue(tmp_path,cfg,reports)['retryable']


def test_worker_startup_error_not_hidden_by_other_recoverable_episode(tmp_path):
    cfg,reports=fixture(tmp_path)
    log=tmp_path/'worker.log'
    log.write_text(json.dumps(dict(event='parallel_worker_error',error='Source changed'))+'\n')
    put(tmp_path/'evaluation/queue/status.json',dict(state='failed',workers=[
        dict(worker='worker_0',exit_code=1,log=str(log),progress={})]))
    assert not retry.assess_queue(tmp_path,cfg,reports)['retryable']


def test_frozen_legacy_worker_timeout_can_resume(tmp_path):
    cfg,reports=fixture(tmp_path)
    log=tmp_path/'worker.log'
    log.write_text(json.dumps(dict(event='parallel_worker_error',
        error='Episode failed (exit 1); not an account failure, queue stopped'))+'\n')
    pool=tmp_path/'pool.json'
    put(pool,dict(jobs=[dict(attempts=[dict(archive=str(tmp_path/'archive'),failure_kind=None)])]))
    put(tmp_path/'evaluation/queue/status.json',dict(state='failed',workers=[
        dict(worker='worker_0',exit_code=1,log=str(log),progress=dict(pool_run=str(pool)))]))
    assert retry.assess_queue(tmp_path,cfg,reports)['retryable']


@pytest.mark.parametrize('message',[ROUTING_FAILURE,WRAPPED_SIM_FAILURE,COMPACTION_FAILURE])
def test_new_failures_recognized_in_legacy_worker_and_batch(tmp_path,message):
    cfg,reports=fixture(tmp_path)
    put(tmp_path/'archive/controller/failure.json',dict(error=message))
    log=tmp_path/'worker.log'
    log.write_text(json.dumps(dict(event='parallel_worker_error',
        error='Episode failed (exit 1); not an account failure, queue stopped'))+'\n')
    pool=tmp_path/'pool.json'
    put(pool,dict(jobs=[dict(attempts=[dict(archive=str(tmp_path/'archive'))])]))
    put(tmp_path/'evaluation/queue/status.json',dict(state='failed',workers=[
        dict(worker='worker_0',exit_code=1,log=str(log),progress=dict(pool_run=str(pool)))]))
    assert retry.assess_queue(tmp_path,cfg,reports)['retryable']


@pytest.mark.parametrize('error,step,expected',[
    ({'message':'Connection failed: error sending request'},12,True),
    ({'message':'Connection failed: error sending request'},11,False),
    ({'message':'invalid action'},12,False),
    ({'message':'quota exhausted'},12,False),
])
def test_network_exhaustion_requires_matching_terminal_evidence(tmp_path,error,step,expected):
    cfg,reports=fixture(tmp_path,'model_recovery_exhausted')
    put(tmp_path/'archive/controller/failure.json',dict(step_id=12,error=
        'hybrid_rollout.robodojo.skill_auto_infer.model_recovery.ModelRecoveryExhausted: Ten same-thread model recovery attempts exhausted'))
    log=tmp_path/'archive/logs/controller.log';log.parent.mkdir()
    log.write_text(json.dumps(dict(event='codex_model_wait_terminal',step_id=step,error=error))+'\n')
    assert retry.assess_queue(tmp_path,cfg,reports)['retryable']==expected


def test_invalid_native_layout_is_not_a_network_retry(tmp_path):
    cfg,reports=fixture(tmp_path)
    path=tmp_path/'results/exp/_replicas/replica_3/attempt_1/batch.json'
    batch=json.loads(path.read_text());row=batch['episodes'][0]
    row.update(state='completed',evaluation_outcome=dict(status='invalid_native_layout',complete=True,valid_for_success_rate=False))
    put(path,batch)
    (tmp_path/'archive/controller/failure.json').unlink()
    assert not retry.assess_queue(tmp_path,cfg,reports)['retryable']


@pytest.mark.parametrize('outcome',['success','exhausted','startup_error','interrupt'])
def test_supervisor_attaches_without_interrupt_and_has_bounded_restarts(tmp_path,monkeypatch,outcome):
    cfg=dict(schema='robodojo.eval_followup.v1',name='combined',
             predecessor_config='first.json',successor_config='second.json')
    path=tmp_path/'followup.json';put(path,cfg)
    queues=[dict(name='first',models=[]),dict(name='second',models=[])]
    monkeypatch.setattr(retry,'read_config',lambda p:queues[0 if p.name=='first.json' else 1])
    monkeypatch.setattr(retry,'validate_partition',lambda *a:None)
    state=dict(busy=True,runs=0)
    monkeypatch.setattr(retry,'busy',lambda p:state['busy'])
    monkeypatch.setattr(retry.time,'sleep',lambda s:state.update(busy=False))
    monkeypatch.setattr(retry,'DELAYS',(0,))
    def reports(shared,c):
        done=c['name']=='first' or (outcome=='success' and state['runs']>0)
        return [dict(task='task',experiment=c['name'],planned=1,completed=int(done),
                     successes=0,failures=int(done),episodes=[])]
    monkeypatch.setattr(retry,'summaries',reports)
    monkeypatch.setattr(retry,'assess_queue',lambda *a:dict(retryable=True,reasons={'wall_time_limit':1},blocked=[]))
    def command(argv,env,log):
        assert not state['busy']
        assert argv[-3:]==['--eval-followup','--config',str(path)]
        state['runs']+=1
        if outcome=='interrupt':raise KeyboardInterrupt()
        if outcome=='startup_error':return 1
        put(tmp_path/'evaluation/second/status.json',dict(queue_status=f'run_{state["runs"]}'))
        return 0 if outcome=='success' else 124
    monkeypatch.setattr(retry,'run_command',command)
    env=dict(ROLLOUT_SHARED_ROOT=str(tmp_path),CODE_ROOT=str(tmp_path))
    code=retry.run(path,env,max_restarts=3)
    result=json.loads((tmp_path/'evaluation/combined/retry_status.json').read_text())
    expected={'success':('completed',1,0),'exhausted':('retry_budget_exhausted',3,1),
              'startup_error':('manual_review_required',1,1),'interrupt':('stopped_by_operator',1,130)}[outcome]
    assert (result['state'],state['runs'],code)==expected


def test_attach_lock_is_exclusive(tmp_path):
    import fcntl
    lock=tmp_path/'lock'
    assert not retry.busy(lock)
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        assert retry.busy(lock)
    assert not retry.busy(lock)


def test_nfs_exclusive_lock_opens_existing_file_for_writing(tmp_path,monkeypatch):
    import fcntl,os
    lock=tmp_path/'lock';lock.write_text('preserve')
    real=fcntl.flock
    def nfs_flock(stream,flags):
        assert fcntl.fcntl(stream.fileno(),fcntl.F_GETFL)&os.O_ACCMODE != os.O_RDONLY
        return real(stream,flags)
    monkeypatch.setattr(fcntl,'flock',nfs_flock)
    assert not retry.busy(lock)
    assert lock.read_text()=='preserve'


@pytest.mark.parametrize('finished',[True,False])
def test_new_unpinned_queue_launches_once_with_zero_retry_budget(tmp_path,monkeypatch,finished):
    from . import eval_queue
    cfg=dict(schema='robodojo.eval_sequence.v1',name='fresh',scheduling='cross_task_queue_v1',models=[])
    path=tmp_path/'config.json';put(path,cfg)
    monkeypatch.setattr(retry,'read_config',lambda _:cfg)
    monkeypatch.setattr(eval_queue,'validate_queue',lambda _:None)
    calls=[]
    def reports(*args):
        done=bool(calls) and finished
        return [dict(task='t',experiment='e',planned=1,completed=int(done),successes=int(done),failures=0,episodes=[])]
    monkeypatch.setattr(retry,'summaries',reports)
    monkeypatch.setattr(retry,'assess_queue',lambda *args:dict(retryable=True,reasons={},blocked=[]))
    def command(argv,env,log):
        calls.append(argv)
        put(tmp_path/'evaluation/fresh/queue_config.json',cfg)
        put(tmp_path/'evaluation/fresh/status.json',dict(queue_status='run1'))
        return 0 if finished else 124
    monkeypatch.setattr(retry,'run_command',command)
    assert retry.run(path,dict(ROLLOUT_SHARED_ROOT=str(tmp_path),CODE_ROOT=str(tmp_path)),max_restarts=0)==(0 if finished else 1)
    state=json.loads((tmp_path/'evaluation/fresh/retry_status.json').read_text())
    assert len(calls)==1 and state['initial_launches']==1 and state['restarts']==0
    assert state['state']==('completed' if finished else 'retry_budget_exhausted')
