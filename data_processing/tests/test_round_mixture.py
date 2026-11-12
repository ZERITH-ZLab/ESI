from copy import deepcopy

import numpy as np
import pytest

from data_processing.sampling.sampler import MULTITASK_CONFIG, checked_config
from data_processing.tests.test_multitask_sampling import make_pool, sampler


def test_round_mass_task_balance_and_exact_resume():
    rows, eps = make_pool()
    for e in eps.values():
        e['collection_round'] = 'r1'
    old_rows, old_eps = deepcopy(rows), deepcopy(eps)
    for key, e in old_eps.items():
        eps['new:'+key] = dict(e, collection_round='r2', policy_version='p2')
    for r in old_rows:
        rows.append(dict(r, sample_id=len(rows), episode_key='new:'+r['episode_key']))
    cfg = dict(MULTITASK_CONFIG, strategy='task_round_case_control_weighted', round_weights={'r1':3,'r2':7})
    s = sampler(rows, eps, cfg)
    plan = s.plan()['effective_probabilities']
    assert plan['collection_round'] == pytest.approx({'r1':.3,'r2':.7})
    assert plan['task'] == pytest.approx({'easy':.5,'hard':.5})
    for task in ('easy','hard'):
        for name, mass in [('r1',.15),('r2',.35)]:
            assert sum(x['probability'] for x in s.leaves if x['task']==task and x['collection_round']==name)==pytest.approx(mass)
    # Identical per-round pools differ only by explicit round mass.
    n=len(old_rows)
    np.testing.assert_allclose(s.probabilities[n:]/s.probabilities[:n], 7/3)
    s.draw_sample_ids(17)
    saved=s.state_dict();expected=s.draw_sample_ids(64)
    restored=sampler(rows, eps, cfg);restored.load_state_dict(saved)
    np.testing.assert_array_equal(expected,restored.draw_sample_ids(64))
    with pytest.raises(ValueError,match='round_weights'):
        sampler(rows,eps,dict(cfg,round_weights={'r1':3,'typo':7}))
    # Retain unsuccessful cases; no success-only filtering in the mixture.
    assert all(s.probabilities[i]>0 for i,r in enumerate(rows) if not eps[r['episode_key']]['episode_success'])


def test_round_config_validation_and_single_round_guard():
    for weights in ({}, {'r1':1}, {'r1':0,'r2':1}, {'r1':True,'r2':1}):
        with pytest.raises(ValueError,match='round_weights'):
            checked_config(dict(MULTITASK_CONFIG,strategy='task_round_case_control_weighted',round_weights=weights))
    rows,eps=make_pool();eps[next(iter(eps))]['collection_round']='other'
    with pytest.raises(ValueError,match='explicit mixing'):
        sampler(rows,eps)
