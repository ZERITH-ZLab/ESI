import argparse
import json
from .build import build, validate


def main():
    parser=argparse.ArgumentParser(description='Build training indexes from audited subtasks and existing command exports')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build')
    p.add_argument('--annotation-run',required=True)
    p.add_argument('--exports',nargs='+',required=True,help='Completed export_campaign directories')
    p.add_argument('--output',required=True)
    p.add_argument('--policy-version',required=True)
    p.add_argument('--collection-round',required=True)
    p.add_argument('--horizon',type=int,default=50)
    p.add_argument('--min-valid',type=int,default=15)
    v=sub.add_parser('validate')
    v.add_argument('output')
    v.add_argument('--artifacts-only',action='store_true',help='Skip rehashing source files')
    m=sub.add_parser('merge')
    m.add_argument('--indexes',nargs='+',required=True)
    m.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.command=='merge':
        from .merge import merge_indexes
        print(json.dumps(merge_indexes(a.indexes,a.output),ensure_ascii=False,indent=2))
        return
    result=(build(a.annotation_run,a.exports,a.output,a.policy_version,a.collection_round,a.horizon,a.min_valid)
            if a.command=='build' else validate(a.output,not a.artifacts_only))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
