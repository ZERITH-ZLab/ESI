"""Analyze a completed collection suite without starting any rollout."""
import argparse
import json
import os
from pathlib import Path
from .pipeline import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-root',type=Path,required=True)
    p.add_argument('--settings',type=Path,help='Offline settings object, not a collection config')
    p.add_argument('--prepare-only',action='store_true',help='Check coverage and save inventory; no model/account calls')
    p.add_argument('--output',type=Path,help='Optional output directory; use mounted storage')
    args=p.parse_args()
    overrides=json.loads(args.settings.read_text()) if args.settings else {}
    result=run(args.suite_root,os.environ,overrides=overrides,prepare_only=args.prepare_only,output=args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['state'] in ('prepared','published','no_change','candidate_only') else 2


if __name__=='__main__':raise SystemExit(main())
