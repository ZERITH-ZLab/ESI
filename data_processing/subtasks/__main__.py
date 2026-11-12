import argparse
import json

from .pipeline import load_config, prepare_batch, run_batch, validate_run
from .catalog import build_catalog


def main():
    parser = argparse.ArgumentParser(description="Evidence-grounded subtask annotation; raw rollouts remain unchanged")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Discover finished attempts and freeze text inputs; no model calls")
    prepare.add_argument("inputs", nargs="+", help="Experiment roots or explicit attempt directories")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--append", action="store_true",
                         help="Rescan the same sources and append new finished episodes; preserve existing inputs/results")
    prepare.add_argument("--latest-per-case", action="store_true",
                         help="Select only the latest attempt per case/repeat; skip it if still running")
    run = sub.add_parser("run", help="Annotate frozen inputs, resume validated results")
    run.add_argument("batch")
    run.add_argument("--config", required=True)
    run.add_argument("--limit", type=int, help="Pilot first N prepared episodes; omit for entire batch")
    validate = sub.add_parser("validate", help="Revalidate artifacts and source fingerprints without model calls")
    validate.add_argument("batch")
    validate.add_argument("--run-id", required=True)
    catalog = sub.add_parser("catalog", help="Build readable names and episode lookup tables; no model calls")
    catalog.add_argument("batch")
    catalog.add_argument("--run-id", help="Omit to catalogue all runs, including older annotation versions")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_batch(args.inputs, args.output, args.limit,
                               append=args.append, latest_per_case=args.latest_per_case)
        report = dict(episodes=len(result["episodes"]), steps=sum(r["steps"] for r in result["episodes"]),
                      excluded=result["excluded"], failures=result["failures"], output=args.output)
        failed = bool(result["failures"]) or not result["episodes"]
    elif args.command == "run":
        report = run_batch(args.batch, load_config(args.config), args.limit)
        failed = report["failures"] > 0
    elif args.command == "catalog":
        report = build_catalog(args.batch, args.run_id)
        failed = False
    else:
        report = validate_run(args.batch, args.run_id)
        failed = report["failures"] > 0
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
