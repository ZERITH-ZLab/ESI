from collections import defaultdict
from copy import deepcopy

import pytest

from data_processing.sampling.sampler import HierarchicalSampler, MULTITASK_CONFIG, MULTITASK_RATIO_CONFIG, checked_config
from data_processing.training.config import load_config


def make_pool():
    episodes, rows = {}, []
    # Unequal trajectory lengths and sharply different success rates must not
    # defeat task balance or amplify rare successes beyond the requested weight.
    for task, successes in [('hard',2),('easy',18)]:
        for i in range(20):
            key = f'{task}:{i}'
            episodes[key] = dict(task=task,case_key=key,case_id=key,variant='standard',
                episode_success=int(i<successes),policy_version='p',collection_round='r')
            for control in ('pi05','gpt'):
                for t in range(i+1):
                    rows.append(dict(sample_id=len(rows),episode_key=key,start_step=t,start_control=control,
                                     valid_length=50,gpt_valid_steps=10,is_decision_start=t==0))
    return rows,episodes


def sampler(rows, episodes, cfg=None):
    s=HierarchicalSampler.__new__(HierarchicalSampler)
    s._initialize(rows,episodes,cfg or MULTITASK_CONFIG,'index')
    return s


def test_weights_sparse_success_and_task_balance():
    rows,eps=make_pool();s=sampler(rows,eps)
    mass=defaultdict(float);outcomes=defaultdict(float)
    for leaf in s.leaves:
        mass[leaf['episode_key']]+=leaf['probability']
        if leaf['outcome']=='success':outcomes[leaf['task']]+=leaf['probability']
    assert s.plan()['effective_probabilities']['task']==pytest.approx(dict(hard=.5,easy=.5))
    assert s.plan()['effective_probabilities']['control']==pytest.approx(dict(gpt=.8,pi05=.2))
    assert outcomes['hard']/.5==pytest.approx(4/22)
    assert outcomes['easy']/.5==pytest.approx(36/38)
    for task in ('hard','easy'):
        assert mass[f'{task}:0']/mass[f'{task}:19']==pytest.approx(2)
    a=s.draw_indices(100);state=s.state_dict();expected=s.draw_indices(100)
    other=sampler(rows,eps);other.load_state_dict(state)
    assert (other.draw_indices(100)==expected).all()
    assert sum(s.report(a)['window_counts']['task'].values())==100


def test_missing_control_falls_back_inside_task_empty_task_errors():
    rows,eps=make_pool()
    rows=[r for r in rows if not (eps[r['episode_key']]['task']=='hard' and r['start_control']=='gpt')]
    s=sampler(rows,eps)
    assert s.plan()['effective_probabilities']['task']==pytest.approx(dict(hard=.5,easy=.5))
    assert s.plan()['effective_probabilities']['control']['gpt']==pytest.approx(.4)
    assert any(f.get('context',{}).get('task')=='hard' for f in s.fallbacks)
    with pytest.raises(ValueError,match='Tasks without trainable windows'):
        sampler([r for r in rows if eps[r['episode_key']]['task']=='easy'],eps)


def test_variant_balance_within_each_task():
    rows,eps=make_pool()
    for i in range(10):eps[f'easy:{i}']['variant']='random'
    s=sampler(rows,eps)
    mass=defaultdict(float)
    for leaf in s.leaves:mass[leaf['task'],leaf['variant']]+=leaf['probability']
    assert mass==pytest.approx({('hard','standard'):.5,('easy','standard'):.25,('easy','random'):.25})


@pytest.mark.parametrize('weight',[0,-1,True,float('inf')])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):checked_config(dict(MULTITASK_CONFIG,successful_case_weight=weight))


def test_training_config_and_launch_overrides():
    path='data_processing/training/config.multitask_round1.json'
    cfg=load_config(path,environ={'TRAIN_SUCCESSFUL_CASE_WEIGHT':'3','TRAIN_GPT_START_WEIGHT':'5',
                                'TRAIN_PI05_START_WEIGHT':'2'},mode='subtask')
    assert cfg['sampling']['successful_case_weight']==3
    assert cfg['conditioning']['mode']=='subtask'
    assert cfg['sampling']['gpt_start_weight']==5 and cfg['sampling']['pi05_start_weight']==2
    with pytest.raises(ValueError,match='does not apply'):
        load_config(path,environ={'TRAIN_SUCCESSFUL_EPISODE_RATIO':'.7'})
    with pytest.raises(ValueError,match='does not apply'):
        load_config(path,environ={'TRAIN_GPT_START_RATIO':'.5'})


def test_sparse_gpt_is_weighted_by_count_after_case_selection():
    eps={k:dict(task='t',case_key=k,case_id=k,variant='standard',episode_success=1,
               policy_version='p',collection_round='r') for k in ('sparse','dense')}
    rows=[]
    for key,g,p in [('sparse',6,327),('dense',60,60)]:
        for control,n in [('gpt',g),('pi05',p)]:
            for i in range(n):rows.append(dict(sample_id=len(rows),episode_key=key,start_step=i,
                start_control=control,valid_length=50,gpt_valid_steps=10,is_decision_start=i==0))
    s=sampler(rows,eps);groups={(l['episode_key'],l['control']):l for l in s.leaves}
    for key in eps:
        g,p=groups[key,'gpt'],groups[key,'pi05']
        assert g['probability']+p['probability']==pytest.approx(.5)
        assert g['per_window_probability']/p['per_window_probability']==pytest.approx(4)
    assert groups['sparse','gpt']['probability']==pytest.approx(.5*24/(24+327))
    # A repeated episode in the same case/outcome shares, rather than doubles, case mass.
    eps['repeat']=dict(eps['sparse'])
    original=[r for r in rows if r['episode_key']=='sparse']
    for r in original:rows.append(dict(r,sample_id=len(rows),episode_key='repeat'))
    repeated=sampler(rows,eps)
    assert sum(l['probability'] for l in repeated.leaves if l['case_key']=='sparse')==pytest.approx(.5)
    assert sum(l['probability'] for l in repeated.leaves if l['episode_key']=='repeat')==pytest.approx(.25)


def test_old_multitask_ratio_strategy_remains_explicitly_available():
    rows,eps=make_pool();s=sampler(rows,eps,MULTITASK_RATIO_CONFIG)
    assert s.plan()['effective_probabilities']['control']==pytest.approx(dict(gpt=.5,pi05=.5))
    with pytest.raises(ValueError,match='differs'):
        sampler(rows,eps).load_state_dict(s.state_dict())


@pytest.mark.parametrize('name',['gpt_start_weight','pi05_start_weight'])
@pytest.mark.parametrize('weight',[0,-1,True,float('nan')])
def test_invalid_control_weight(name,weight):
    with pytest.raises(ValueError):checked_config(dict(MULTITASK_CONFIG,**{name:weight}))
