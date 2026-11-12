from concurrent.futures import ThreadPoolExecutor
from .evaluation import archive_path, reserve_sample_attempt


def test_local_attempt_allocation(tmp_path):
    case = dict(task='task', variant='standard', eval_seed=0, layout_id=0)
    def reserve(_):
        return reserve_sample_attempt(tmp_path, 'experiment', case, 3, 1)
    with ThreadPoolExecutor(max_workers=8) as workers:
        assert sorted(workers.map(reserve, range(16))) == list(range(16))
    assert reserve_sample_attempt(tmp_path, 'experiment', case, 3, 2) == 0
    assert reserve_sample_attempt(tmp_path, 'experiment', dict(case, layout_id=1), 3, 1) == 0
    path = archive_path(tmp_path, 'experiment', case, 3, 0, 1)
    assert str(path).endswith('layout_0/replica_3/repeat_001/attempt_000')
    assert not path.exists()


def test_existing_archive_is_not_overwritten(tmp_path):
    case = dict(task='task', variant='random', eval_seed=0, layout_id=0)
    archive_path(tmp_path, 'experiment', case, 3, 0, 1).mkdir(parents=True)
    assert reserve_sample_attempt(tmp_path, 'experiment', case, 3, 1) == 1


def test_unmigrated_failure_counts_as_first_attempt(tmp_path):
    import json
    case = dict(task='task', variant='random', eval_seed=0, layout_id=0)
    sample = archive_path(tmp_path, 'experiment', case, 3, 0, 2).parent
    old = sample.parent/'attempt_39'
    old.mkdir(parents=True)
    (old/'collection_job.json').write_text(json.dumps(dict(repeat_id=2)))
    assert reserve_sample_attempt(tmp_path, 'experiment', case, 3, 2) == 1
    assert reserve_sample_attempt(tmp_path, 'experiment', case, 3, 1) == 0
