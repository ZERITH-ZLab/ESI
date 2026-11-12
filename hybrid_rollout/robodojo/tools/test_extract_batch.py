import json
from pathlib import Path

import pytest

from .extract_batch import discover, run
from .test_extract_episode import fixture


def source(tmp_path):
    raw = fixture(tmp_path)
    root = tmp_path/'experiment'
    sample = root/'task/standard/eval_seed_0/layout_0/replica_3/repeat_001'
    sample.mkdir(parents=True)
    archive = sample/'attempt_000'
    raw.rename(archive)
    (archive/'component_reset.json').write_text(json.dumps(dict(all_owned_processes_exited=True, ports_released=True)))
    (archive/'collection_job.json').write_text('{}')
    (sample/'sample.json').write_text(json.dumps(dict(state='completed', archive=str(archive),
        attempts=[dict(archive=str(archive), finished_at=1)])))
    return root, archive


def test_batch_reuse_and_corruption(tmp_path):
    root, archive = source(tmp_path)
    output = tmp_path/'out'
    first = run(root, output, 1, 1)
    assert first['failed'] == 0 and first['total_steps'] == 2
    second = run(root, output, 1, 1)
    assert second['results'][0]['state'] == 'reused'
    (Path(first['results'][0]['output'])/'steps.npz').write_bytes(b'bad')
    assert run(root, output, 1, 1)['failed'] == 1


def test_unfinished_excluded(tmp_path):
    root, archive = source(tmp_path)
    (archive/'component_reset.json').write_text('{}')
    candidates, skipped = discover(root)
    assert not candidates and len(skipped) == 1
    with pytest.raises(ValueError, match='Only 0'):
        run(root, tmp_path/'out', 1)


def test_teacher_abandoned_is_finalized(tmp_path):
    root, archive = source(tmp_path)
    p = archive.parent/'sample.json'
    data = json.loads(p.read_text()); data['state'] = 'teacher_abandoned'
    p.write_text(json.dumps(data))
    p = archive/'sim/summary.json'
    data = json.loads(p.read_text()); data['reason'] = 'teacher_abandon'
    p.write_text(json.dumps(data))
    rows, skipped = discover(root)
    assert len(rows) == 1 and not skipped


def test_source_change_not_reused(tmp_path):
    root, archive = source(tmp_path)
    out = tmp_path/'out'
    assert run(root, out, 1)['failed'] == 0
    (archive/'controller/history.json').write_text('[]')
    assert run(root, out, 1)['failed'] == 1


def test_null_archive_is_skipped(tmp_path):
    root, archive = source(tmp_path)
    (archive.parent/'sample.json').write_text(json.dumps(dict(state='failed', archive=None)))
    rows, skipped = discover(root)
    assert rows == [] and len(skipped) == 1


def test_active_sample_is_not_read_even_with_old_summary(tmp_path):
    root, archive = source(tmp_path)
    p = archive.parent/'sample.json'
    data = json.loads(p.read_text()); data['state'] = 'running'
    p.write_text(json.dumps(data))
    rows, skipped = discover(root)
    assert not rows and len(skipped) == 1
