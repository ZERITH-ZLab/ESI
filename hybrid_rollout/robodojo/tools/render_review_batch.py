"""Batch review videos from a collection experiment; original archives are read-only.

Fixed-depth discovery excludes source snapshots. Existing compatible outputs
are skipped; incompatible outputs and partial encodes are preserved and reported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

from ..io import write_json
from .render_review_video import VERSION, destination, input_signature, render, validate_history

PATTERN = '*/*/eval_seed_*/layout_*/replica_*/repeat_*/attempt_*/controller/history.json'


def discover(root, output_root):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f'Experiment directory does not exist: {root}')
    if root.name == 'results':
        raise ValueError('Expected results/<experiment>, but received the results directory. '
                         'Keep the full experiment path in one shell argument; '
                         'do not insert a newline between results/ and the experiment name.')
    if output_root.resolve().is_relative_to(root):
        raise ValueError('Batch output must be outside the original experiment')
    jobs, excluded, outputs = [], [], set()
    for history_path in sorted(root.glob(PATTERN)):
        episode = history_path.parent.parent
        try:
            summary = json.loads((episode/'sim/summary.json').read_text())
            history = json.loads(history_path.read_text())
            validate_history(history, summary['step_id'])
            if summary['video_frames'] != summary['step_id'] + 1:
                raise ValueError('Summary video frame count disagrees with step count')
            if not (episode/'sim/sensors.mp4').is_file():
                raise ValueError('Missing sensors.mp4')
            output = destination(episode, output_root, summary['evaluation_case'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            excluded.append(dict(episode=str(episode), error=str(error)))
            continue
        if output in outputs:
            raise ValueError(f'Duplicate output destination: {output}')
        outputs.add(output)
        jobs.append(dict(episode=str(episode), output=str(output)))
    return jobs, excluded


def process(job, args):
    episode, output = Path(job['episode']), Path(job['output'])
    try:
        index_path = output.with_suffix('.index.json')
        if output.exists() or index_path.exists():
            index = json.loads(index_path.read_text())
            if (not output.is_file() or not output.stat().st_size
                    or index.get('version') != VERSION
                    or index.get('episode') != str(episode)
                    or index.get('pause_seconds') != args.pause
                    or index.get('input_signature') != input_signature(episode, args.font)):
                raise ValueError('Existing output is incompatible; choose a fresh --output-root')
            return dict(job, status='skipped')
        render(SimpleNamespace(episode=episode, output_root=args.output_root, output=output,
                               pause=args.pause, font=args.font, index=True))
        return dict(job, status='rendered')
    except Exception as error:
        return dict(job, status='failed', error=f'{type(error).__name__}: {error}')


def run_batch(args):
    jobs, excluded = discover(args.experiment, args.output_root)
    print(json.dumps(dict(event='review_batch_plan', episodes=len(jobs), excluded=excluded,
                          jobs=jobs if args.dry_run else None), ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    if not jobs:
        raise ValueError('No renderable collection episodes found')
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root/f'batch_{args.experiment.name}_{uuid.uuid4().hex}.json'
    report = dict(version=VERSION, experiment=str(args.experiment.resolve()),
                  started_at=datetime.now(timezone.utc).isoformat(), pause_seconds=args.pause,
                  workers=args.workers, excluded=excluded, results=[])
    write_json(report_path, report)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, job, args) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            report['results'].append(result)
            write_json(report_path, report)
            print(json.dumps(dict(event='review_batch_progress', completed=len(report['results']),
                                  total=len(jobs), **result), ensure_ascii=False), flush=True)
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report['counts'] = {status: sum(r['status'] == status for r in report['results'])
                        for status in ('rendered', 'skipped', 'failed')}
    write_json(report_path, report)
    print(json.dumps(dict(report=str(report_path), **report['counts']), ensure_ascii=False), flush=True)
    return int(bool(report['counts']['failed'] or excluded))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path, help='Collection results/<experiment> directory')
    parser.add_argument('--output-root', type=Path, default=Path(
        '/mnt/esi/review_videos_v2'))
    parser.add_argument('--workers', type=int, default=2, help='Parallel encoders (each uses 2 FFmpeg threads)')
    parser.add_argument('--pause', type=float, default=.8)
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    parser.add_argument('--dry-run', action='store_true', help='List episodes and outputs without writing or encoding')
    args = parser.parse_args()
    if not 0 <= args.pause <= 10 or not 1 <= args.workers <= 32:
        parser.error('--pause must be 0..10; --workers must be 1..32')
    try:
        code = run_batch(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
