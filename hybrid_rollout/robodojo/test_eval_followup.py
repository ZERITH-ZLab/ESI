import copy
import fcntl
import json
from pathlib import Path

import pytest

from . import eval_followup as followup


def configs():
    root=Path(__file__).parent
    return [json.loads((root/name).read_text()) for name in
            ('eval_queue.round3_pilot20.json','eval_queue.round3_remaining180.json')]


def test_full200_partition_and_same_execution():
    a,b=configs()
    followup.validate_partition(a,b)
    assert a['workers']==b['workers']==8
    assert sum(len(m['cases']) for m in a['models'])==20
    assert sum(len(m['cases']) for m in b['models'])==180
    for m in b['models']:
        assert ['standard',0] not in m['cases']
        assert ['standard',1] not in m['cases']


@pytest.mark.parametrize('change,message',[
    ('overlap','Overlapping'),('missing','Incomplete'),
    ('policy','Policy settings'),('source','same episode source')])
def test_reject_invalid_complements(change,message):
    a,b=configs()
    if change=='overlap':b['models'][0]['cases'].append(['standard',0])
    if change=='missing':b['models'][0]['cases'].pop()
    if change=='policy':
        for m in b['models']:m['policy_input']='task_only'
    if change=='source':b['execution_source']='different'
    with pytest.raises(ValueError,match=message):followup.validate_partition(a,b)


def row(planned,completed,successes=0):
    return dict(task='task',experiment='exp',planned=planned,completed=completed,
                successes=successes,failures=completed-successes,episodes=[])


def test_completion_gate_counts_failed_tasks_but_not_unfinished_or_live_cleanup(tmp_path):
    # Native task failures are completed evaluations, not infrastructure errors.
    rows=[row(20,20,3)]
    with (tmp_path/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert followup.queue_busy(tmp_path)
        assert not followup.predecessor_ready(rows,True)
    assert not followup.queue_busy(tmp_path)
    assert followup.predecessor_ready(rows,False)
    assert not followup.predecessor_ready([row(20,19)],False)
    merged=followup.aggregate(rows,[row(180,180,100)])
    assert (merged['planned'],merged['completed'],merged['successes'],merged['failures'])==(200,200,103,97)


def test_wait_then_launch_once_and_write_combined_report(tmp_path,monkeypatch):
    a,b=configs()
    config=dict(schema='robodojo.eval_followup.v1',name='full200',
                predecessor_config='first.json',successor_config='second.json')
    path=tmp_path/'chain.json';path.write_text(json.dumps(config))
    monkeypatch.setattr(followup,'read_config',lambda p:copy.deepcopy(a if p.name=='first.json' else b))
    monkeypatch.setattr(followup,'execution_root',lambda *args:None)
    env=dict(ROLLOUT_SHARED_ROOT=str(tmp_path),ROLLOUT_MODEL='teacher',
             ROLLOUT_REASONING_EFFORT='xhigh',ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT='96000')
    teacher={k:v for k,v in env.items() if k!='ROLLOUT_SHARED_ROOT'}
    prior=tmp_path/'evaluation'/a['name'];prior.mkdir(parents=True)
    (prior/'queue_config.json').write_text(json.dumps(dict(config=a,teacher=teacher)))
    state=dict(poll=0,launched=False);calls=[]
    def reports(shared,cfg):
        if cfg['name']==a['name']:return [row(20,19 if state['poll']==0 else 20,3)]
        return [row(180,180 if state['launched'] else 0,100 if state['launched'] else 0)]
    monkeypatch.setattr(followup,'summaries',reports)
    monkeypatch.setattr(followup,'queue_busy',lambda path:state['poll']<2)
    monkeypatch.setattr(followup.time,'sleep',lambda seconds:state.update(poll=state['poll']+1))
    def launch(cfg,env,**kw):
        assert state['poll']==2
        calls.append((cfg['name'],kw))
        if cfg['name']==b['name']:state['launched']=True
        return 0
    monkeypatch.setattr(followup,'run_queue',launch)
    assert followup.run(path,env)==0
    assert calls==[(a['name'],dict(dry_run=True,check_runtime=True)),(b['name'],{})]
    result=json.loads((tmp_path/'evaluation/full200/summary.json').read_text())
    assert result['completed']==200 and result['failures']==97
