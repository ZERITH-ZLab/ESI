import argparse
import json
from data_processing.subtasks.catalog import build_catalog
from .prepare import prepare_batch
from .pipeline import load_config, run_batch, validate_run


def main():
    parser = argparse.ArgumentParser(description='Preserve online subtask inputs; label outcomes, not new skill boundaries')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Freeze finalized online inputs, no model calls')
    p.add_argument('inputs', nargs='+')
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--append', action='store_true')
    p.add_argument('--latest-per-case', action='store_true')
    r = commands.add_parser('run', help='Skip verified successes; retry failed/unstarted episodes')
    r.add_argument('batch')
    r.add_argument('--config', required=True)
    r.add_argument('--limit', type=int)
    v = commands.add_parser('validate', help='Verify all outputs and raw fingerprints; no model calls')
    v.add_argument('batch')
    v.add_argument('--run-id', help='Default: latest_run.json')
    c = commands.add_parser('catalog')
    c.add_argument('batch')
    c.add_argument('--run-id')
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare_batch(args.inputs, args.output, args.limit, append=args.append,
                                 latest_per_case=args.latest_per_case)
        report = dict(output=args.output, episodes=len(manifest['episodes']),
                      excluded=manifest['excluded'], failures=manifest['failures'])
        failed = bool(report['failures']) or not report['episodes']
    elif args.command == 'run':
        report = run_batch(args.batch, load_config(args.config), args.limit)
        failed = bool(report['failures'])
    elif args.command == 'validate':
        report = validate_run(args.batch, args.run_id)
        failed = bool(report['failures'])
    else:
        report = build_catalog(args.batch, args.run_id)
        failed = False
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
