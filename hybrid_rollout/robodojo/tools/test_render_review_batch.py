import json
from pathlib import Path
from types import SimpleNamespace

from . import render_review_batch as batch
from .render_review_video import VERSION, input_signature


def fixture(tmp_path):
    root = tmp_path/'results/experiment'
    episode = root/'task/standard/eval_seed_0/layout_0/replica_0/repeat_1/attempt_0'
    (episode/'controller').mkdir(parents=True)
    (episode/'sim').mkdir()
    (episode/'controller/history.json').write_text(json.dumps([dict(decision=0,
        start_tick=0, end_tick=1, executed_steps=1, response=dict(mode='student'))]))
    (episode/'sim/summary.json').write_text(json.dumps(dict(step_id=1, video_frames=2,
        evaluation_case=dict(case_id='case', task='task'))))
    (episode/'sim/sensors.mp4').write_bytes(b'video fixture')
    font = tmp_path/'font'; font.write_bytes(b'font fixture')
    args = SimpleNamespace(experiment=root, output_root=tmp_path/'reviews', font=font,
                           pause=.8, workers=2, dry_run=False)
    return args, episode


def test_discovery_and_read_only_dry_run(tmp_path):
    args, episode = fixture(tmp_path)
    # A source snapshot with history must not count as an episode.
    (episode/'source_snapshot').mkdir()
    (episode/'source_snapshot/history.json').write_text('[]')
    jobs, excluded = batch.discover(args.experiment, args.output_root)
    assert len(jobs) == 1 and not excluded
    args.dry_run = True
    assert batch.run_batch(args) == 0
    assert not args.output_root.exists()


def test_resume_checks_inputs_and_options(tmp_path):
    args, episode = fixture(tmp_path)
    job = batch.discover(args.experiment, args.output_root)[0][0]
    output = Path(job['output']); output.parent.mkdir(parents=True)
    output.write_bytes(b'encoded video')
    index = dict(version=VERSION, episode=str(episode), pause_seconds=.8,
                 input_signature=input_signature(episode, args.font))
    output.with_suffix('.index.json').write_text(json.dumps(index))
    assert batch.process(job, args)['status'] == 'skipped'
    args.pause = .4
    assert batch.process(job, args)['status'] == 'failed'
    args.pause = .8
    (episode/'sim/sensors.mp4').write_bytes(b'changed')
    assert batch.process(job, args)['status'] == 'failed'
    assert output.read_bytes() == b'encoded video'


def test_batch_records_failures_and_continues(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    jobs, _ = batch.discover(args.experiment, args.output_root)
    monkeypatch.setattr(batch, 'discover', lambda *a: (jobs*2, []))
    calls = []
    def process(job, args):
        calls.append(job)
        return dict(job, status='failed', error='encoder failed') if len(calls) == 1 else dict(job, status='rendered')
    monkeypatch.setattr(batch, 'process', process)
    assert batch.run_batch(args) == 1
    assert len(calls) == 2
    report = json.loads(next(args.output_root.glob('batch_*.json')).read_text())
    assert report['counts'] == dict(rendered=1, skipped=0, failed=1)


def test_incomplete_episode_and_output_inside_source(tmp_path):
    import pytest
    args, episode = fixture(tmp_path)
    (episode/'sim/summary.json').unlink()
    jobs, excluded = batch.discover(args.experiment, args.output_root)
    assert not jobs and len(excluded) == 1
    with pytest.raises(ValueError, match='outside'):
        batch.discover(args.experiment, args.experiment/'reviews')


def test_wrong_input_directory(tmp_path):
    import pytest
    args, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='Keep the full experiment path'):
        batch.discover(args.experiment.parent, args.output_root)
    with pytest.raises(ValueError, match='does not exist'):
        batch.discover(tmp_path/'missing', args.output_root)
