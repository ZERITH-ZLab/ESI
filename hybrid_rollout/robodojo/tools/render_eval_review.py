"""Render paired eval reviews in the same layout/seed structure for all models."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from ..io import write_json
from ..regression_eval import report
from .render_review_batch import process
from .render_review_video import VERSION


def plan(shared, output_root, experiments):
    jobs = []
    seen = set()
    for label, experiment, unified in experiments:
        summary = report(shared, experiment, unified=unified)
        if summary['completed'] != summary['planned']:
            raise ValueError(f'{experiment}: {summary["completed"]}/{summary["planned"]} valid episodes; finish eval first')
        for row in summary['episodes']:
            seed = row['policy_rng_seed']
            output = (output_root/label/row['variant']/f'layout_{row["layout_id"]}'/
                      f'repeat_{seed+1:03d}.mp4')
            if output in seen:
                raise ValueError(f'Duplicate output: {output}')
            seen.add(output)
            jobs.append(dict(episode=row['archive'], output=str(output), model=label,
                experiment=experiment, variant=row['variant'], layout_id=row['layout_id'],
                policy_rng_seed=seed, native_success=row['native_success'],
                native_score=row['native_score'], steps=row['steps'], gpt_steps=row['gpt_steps']))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-root', type=Path, default=Path('/mnt/esi'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('/mnt/esi/review_videos_v2/eval_compare6000_v1'))
    parser.add_argument('--base-experiment', default='arrange_regression5_base_parallel4_v1')
    parser.add_argument('--plain-experiment', default='arrange_regression15_train6000_v1')
    parser.add_argument('--subtask-experiment', default='arrange_regression15_subtask6000_no_instruction_v1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pause', type=float, default=0,
                        help='Inserted review hold in seconds; default 0 preserves source timing')
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 0 <= args.pause <= 10:
        parser.error('Require workers in 1..32 and pause in 0..10')
    args.output_root = args.output_root.resolve()
    if args.output_root.is_relative_to((args.shared_root/'results').resolve()):
        parser.error('Review output must be outside original results')
    jobs = plan(args.shared_root, args.output_root, [
        ('base', args.base_experiment, False), ('plain6000', args.plain_experiment, True),
        ('subtask6000_no_instruction', args.subtask_experiment, True)])
    print(json.dumps(dict(event='eval_review_plan', episodes=len(jobs), output=str(args.output_root),
                          jobs=jobs if args.dry_run else None)), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root/'review_manifest.json'
    record = dict(version=VERSION, started_at=datetime.now(timezone.utc).isoformat(),
        pause_seconds=args.pause, workers=args.workers, planned=len(jobs), jobs=jobs, results=[],
        layout='model/variant/layout_N/repeat_001..003.mp4; repeat = policy_rng_seed + 1',
        legend={'green':'pi05', 'yellow':'GPT edit', 'orange':'GPT short', 'purple':'GPT long',
                'white':'inserted review pause', 'red':'execution failure feedback (not episode success label)'},
        subtask_input='task_only, without online subtask guidance')
    write_json(manifest, record)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, job, args) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            record['results'].append(result)
            write_json(manifest, record)
            print(json.dumps(dict(event='eval_review_progress', completed=len(record['results']),
                total=len(jobs), **result)), flush=True)
    record['finished_at'] = datetime.now(timezone.utc).isoformat()
    record['counts'] = {status: sum(r['status'] == status for r in record['results'])
                        for status in ('rendered', 'skipped', 'failed')}
    write_json(manifest, record)
    print(json.dumps(dict(event='eval_review_finished', manifest=str(manifest), **record['counts'])), flush=True)
    raise SystemExit(int(bool(record['counts']['failed'])))


if __name__ == '__main__':
    main()
