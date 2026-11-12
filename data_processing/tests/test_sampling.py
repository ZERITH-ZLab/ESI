from copy import deepcopy
from collections import defaultdict
import json

import numpy as np
import pytest

from data_processing.sampling.sampler import HierarchicalSampler, DEFAULT_CONFIG, load_config
from data_processing.subtasks.io import write
from data_processing.tests.test_subtasks import episode, candidate
from data_processing.tests.test_training_index import indexed_sources
from data_processing.training_index.build import build


def pool():
    episodes,rows={},[]
    for variant in ('standard','random'):
        for case in range(2):
            case_key=f'{variant}:case-{case}'
            for outcome in (0,1):
                for repeat in range(2 if outcome else 1):
                    key=f'{case_key}:{outcome}:{repeat}'
                    episodes[key]=dict(case_key=case_key,case_id=f'case-{case}',variant=variant,
                        episode_success=outcome,policy_version='policy0',collection_round='round0')
                    for control in ('gpt','pi05'):
                        # Vary window counts strongly: fairness must be by case and episode.
                        for t in range(2 if repeat==0 else 10):
                            rows.append(dict(sample_id=10+3*len(rows),episode_key=key,start_step=t,
                                start_control=control,valid_length=30 if control=='gpt' else 50,
                                gpt_valid_steps=20 if control=='gpt' else 5,is_decision_start=int(t==0)))
    return rows,episodes


def sampler(rows=None,episodes=None,config=None):
    if rows is None:rows,episodes=pool()
    instance=HierarchicalSampler.__new__(HierarchicalSampler)
    instance._initialize(rows,episodes,config or DEFAULT_CONFIG,'test-index-sha')
    return instance


def test_hierarchical_probability_and_fairness():
    s=sampler();p=s.plan()
    assert p['effective_probabilities']['control']==pytest.approx(dict(gpt=.5,pi05=.5))
    assert p['effective_probabilities']['variant']==pytest.approx(dict(standard=.75,random=.25))
    assert p['effective_probabilities']['episode_outcome']==pytest.approx(dict(success=.7,failure=.3))
    assert not p['fallbacks']
    cases,eps=defaultdict(float),defaultdict(list)
    for leaf in p['episode_strata']:
        cases[(leaf['control'],leaf['variant'],leaf['case_key'])]+=leaf['probability']
        eps[(leaf['control'],leaf['variant'],leaf['case_key'],leaf['outcome'])].append(leaf)
    for (control,variant,case),mass in cases.items():
        assert mass==pytest.approx(.5*(.75 if variant=='standard' else .25)/2)
    for leaves in eps.values():
        assert len({round(x['probability'],12) for x in leaves})==1
        if len(leaves)==2:
            assert leaves[0]['windows']!=leaves[1]['windows']
            assert leaves[0]['per_window_probability']!=leaves[1]['per_window_probability']
    for i,r in enumerate(s.rows):
        ep=s.episodes[r['episode_key']]
        same=[j for j,other in enumerate(s.rows) if other['episode_key']==r['episode_key']
              and other['start_control']==r['start_control']]
        expected=.5*(.75 if ep['variant']=='standard' else .25)/2*(.7/2 if ep['episode_success'] else .3)/len(same)
        assert s.probabilities[i]==pytest.approx(expected)


def test_missing_outcomes_keep_case_fairness_and_report_global_shift():
    rows,eps=pool()
    # case-0 has only failed episodes, but valid successful subtasks still exist there.
    rows=[r for r in rows if not (eps[r['episode_key']]['case_id']=='case-0' and eps[r['episode_key']]['episode_success'])]
    eps={k:v for k,v in eps.items() if any(r['episode_key']==k for r in rows)}
    s=sampler(rows,eps)
    assert s.plan()['effective_probabilities']['episode_outcome']['success']==pytest.approx(.35)
    assert len(s.fallbacks)==4
    assert all(f['level']=='episode_outcome' and f['effective']=={'failure':1} for f in s.fallbacks)


def test_empty_control_variant_and_zero_weight_fallback():
    rows,eps=pool()
    rows=[r for r in rows if r['start_control']=='pi05' and eps[r['episode_key']]['variant']=='standard']
    s=sampler(rows,eps,dict(DEFAULT_CONFIG,gpt_start_ratio=1,random_ratio=1))
    p=s.plan()
    assert p['effective_probabilities']['control']=={'pi05':1}
    assert p['effective_probabilities']['variant']=={'standard':1}
    assert len([f for f in s.fallbacks if f['reason']=='all_available_weights_zero_uniform_fallback'])==2
    assert all(s.rows[i]['start_control']=='pi05' for i in s.draw_indices(100))


def test_reproducibility_batching_and_checkpoint():
    a,b=sampler(),sampler()
    first=a.draw_indices(17)
    expected=a.draw_indices(100)
    assert np.array_equal(np.concatenate([first,expected]),b.draw_indices(117))
    state=a.state_dict();following=a.draw_indices(1000)
    c=sampler();c.load_state_dict(json.loads(json.dumps(state)))
    assert np.array_equal(following,c.draw_indices(1000))
    assert c.draw_count==1117
    ids_sampler=sampler();index_sampler=sampler()
    assert np.array_equal(ids_sampler.draw_sample_ids(100),index_sampler.sample_ids[index_sampler.draw_indices(100)])
    different=sampler(config=dict(DEFAULT_CONFIG,gpt_start_ratio=.4))
    with pytest.raises(ValueError,match='differs'):different.load_state_dict(state)


def test_observed_draws_supervised_steps_and_repeats():
    s=sampler();draws=s.draw_indices(50000);report=s.report(draws)
    assert report['fractions']['control']['gpt']==pytest.approx(.5,abs=.01)
    assert report['fractions']['variant']['random']==pytest.approx(.25,abs=.01)
    assert report['fractions']['episode_outcome']['success']==pytest.approx(.7,abs=.01)
    assert report['repeated_draws']==50000-report['unique_windows']
    assert sum(report['episode_draws'].values())==50000
    assert report['supervised_steps']['gpt']==sum(s.rows[i]['gpt_valid_steps'] for i in draws)
    # Window ratio and supervised action ratio are intentionally different.
    assert report['supervised_steps']['gpt']/report['supervised_steps']['total']!=pytest.approx(.5,abs=.03)
    assert s.report(s.draw_indices(0))['draws']==0


@pytest.mark.parametrize('bad',[{'gpt_start_ratio':1.1},{'random_ratio':float('nan')},
                                {'successful_episode_ratio':True},{'seed':-1},{'unknown':1}])
def test_invalid_config(bad,tmp_path):
    p=tmp_path/'config.json';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):load_config(p)


def test_real_sqlite_mapping_and_integrity(indexed_sources):
    run,export,out=indexed_sources
    build(run,[export],out,'policy0','round0',min_valid=5)
    s=HierarchicalSampler(out)
    assert s.sample_ids.tolist()==[1,2]
    assert {r['start_step'] for r in s.rows}=={4,5}
    assert s.plan()['effective_probabilities']['control']=={'gpt':1}
    assert s.plan()['effective_probabilities']['episode_outcome']=={'failure':1}
    with (out/'index.sqlite').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='modified'):HierarchicalSampler(out)
