import pytest

from .eval_queue import select_retry_campaigns, select_retry_slots, queue_slots
from .dual_campaign import resource_keys
from .test_parallel_collection import environment


def test_retry_selects_experiment_and_case_without_rewriting_panel():
    a = dict(ROLLOUT_EXPERIMENT_ID='a', ROLLOUT_EVAL_MANIFEST='original.json')
    b = dict(a, ROLLOUT_EXPERIMENT_ID='b')
    cases = [dict(case_id='same'), dict(case_id='other')]
    original = [(a, cases), (b, cases)]
    result = select_retry_campaigns(original, ['b=same'])
    assert result == [(b, [cases[0]])]
    assert result[0][0] is b
    assert len(cases) == 2
    assert select_retry_campaigns(original, None) is original
    for bad in [['a=missing'], ['a=same', 'a=same']]:
        with pytest.raises(ValueError):
            select_retry_campaigns(original, bad)


def test_three_retry_workers_have_disjoint_gpu_and_port_locks():
    env = dict(environment(), ROLLOUT_STRICT_EVAL='1', ROLLOUT_PARALLEL_WORKERS='4',
               ROLLOUT_EXPERIMENT_ID='pretrain', ROLLOUT_REPLICA_ID='3')
    first = select_retry_slots(queue_slots({'workers': 8}, env), [0, 1])
    second = select_retry_slots(queue_slots({'workers': 3}, dict(env, ROLLOUT_EXPERIMENT_ID='order')), [2])
    assert second[0]['ROLLOUT_WORKER_ID'] == 'worker_0'
    assert [(e['ROLLOUT_SIM_SSH_PORT'], e['SIM_GPU']) for e in first+second] == [
        ('4205', '0'), ('5727', '0'), ('4205', '1')]
    keys = lambda slots: {tuple(k) for e in slots for k in resource_keys(e)}
    assert not keys(first) & keys(second)
    assert len({e[k] for e in first+second for k in ('POLICY_PORT','SIM_PORT')}) == 6
    for bad in [[], [0, 0], [8], [-1]]:
        with pytest.raises(ValueError):
            select_retry_slots(queue_slots({'workers': 8}, env), bad)
