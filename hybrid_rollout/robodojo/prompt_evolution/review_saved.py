"""Re-propose and independently audit frozen summaries; isolated experiment only."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

from ..io import write_json
from .bundle import ROOT
from .config import settings
from .conversation import run_conversation
from .data import read
from .replay import export_text, fingerprints


def replay_saved(source, output, *, overrides=None, prepare_only=False, backend=None):
    source=Path(source).resolve();output=Path(output).resolve()
    names=('manifest.json','base_bundle.json','dossiers.json','patch_reviews.json','status.json')
    hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
    old=read(source/'manifest.json');base=read(source/'base_bundle.json')
    if read(source/'status.json')['state'] not in ('no_change','candidate_only','published'):
        raise ValueError('Source analysis is not complete')
    manifest=old['inventory'];suite=Path(old['suite']).resolve()
    forbidden=[ROOT.parents[1],suite,source.parent,*[Path(r['archive']).resolve() for r in manifest['episodes']]]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in forbidden):
        raise ValueError('Output overlaps production code, original analysis or collection records')
    issues=read(source/'dossiers.json');rows={r['key']:r for r in manifest['episodes']}
    if len({i['ref'] for i in issues})!=len(issues):raise ValueError('Duplicate saved evidence references')
    for i in issues:
        if i['episode_key'] not in rows or i['history_sha256']!=rows[i['episode_key']]['history_sha256']:
            raise ValueError('Saved evidence source identity mismatch')
    opts=settings(dict(old['config'],**(overrides or {})))
    opts.update(workflow='conversation_v2',publish='candidate_only',enabled=False)
    identity=dict(config=opts,inventory=manifest,base_sha256=base['sha256'],experiment_only=True,
        replay_kind='saved_summary_proposal_audit_v1',source_analysis=str(source),source_hashes=hashes)
    output.mkdir(parents=True,exist_ok=True)
    before=fingerprints(ROOT)
    reuse=dict(issues={i['ref']:i for i in issues},reviews=read(source/'patch_reviews.json'))
    print(json.dumps(dict(event='saved_summary_review',output=str(output),episodes=len(rows),
        saved_issues=len(issues),prior_candidates=len(reuse['reviews']),model=opts['model'],
        effort=opts['reasoning_effort'],publish='candidate_only'),ensure_ascii=False),flush=True)
    export_text(output,'before',base,base)
    try:
        result=run_conversation(suite,dict(name='isolated_saved_review'),opts,manifest,base,identity,
            output/'analysis',prepare_only=prepare_only,backend=backend,reuse_analysis=reuse)
        path=output/'analysis/candidate_bundle.json'
        if path.exists():
            export_text(output,'after',read(path),base)
            (output/'prompt_changes.diff').write_text(''.join(''.join(difflib.unified_diff(
                p.read_text().splitlines(True),(output/'after'/p.name).read_text().splitlines(True),
                fromfile='before/'+p.name,tofile='after/'+p.name)) for p in sorted((output/'before').glob('*.md'))))
        write_json(output/'status.json',result)
        return result
    finally:
        if fingerprints(ROOT)!=before:raise RuntimeError('Live guidance changed concurrently during isolated review')
        if any(hashlib.sha256((source/n).read_bytes()).hexdigest()!=hashes[n] for n in names):
            raise RuntimeError('Source analysis changed during replay')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--settings',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    result=replay_saved(args.source_analysis,args.output,overrides=read(args.settings) if args.settings else None,
        prepare_only=args.prepare_only)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
