"""Isolated historical-prompt replay; never publish or launch a collection."""
import argparse
from collections import defaultdict
import difflib
import hashlib
import json
from pathlib import Path

from ..io import write_json
from .bundle import ROOT, digest, render, seed
from .config import settings
from .data import inventory, read
from .pipeline import run

FILES = ('manipulation_guidance.md', 'task_completion_briefs.md')
HERE = Path(__file__).parent


def fingerprints(source):
    return {name: hashlib.sha256((Path(source)/name).read_bytes()).hexdigest() for name in FILES}


def export_text(root, name, value, original):
    folder = root/name
    folder.mkdir(parents=True, exist_ok=True)
    for task in sorted(value['tasks']):
        shared, brief = render(value, task, original=original)
        (folder/(task+'.md')).write_text(brief)
    (folder/'manipulation_guidance.md').write_text(shared)


def replay(config, *, prepare_only=False, backend=None):
    cfg = read(config)
    suite = Path(cfg['suite_root']).resolve()
    source = Path(cfg['baseline_source']).resolve()
    output = Path(cfg['output_root']).resolve()
    # Keep all artifacts outside live code, source suites and original episode records.
    if any(output.is_relative_to(p) for p in (ROOT.parents[1], suite, source.parents[2])):
        raise ValueError('Replay output overlaps production code or source data')
    if fingerprints(source) != cfg['baseline_sha256']:
        raise ValueError('Historical source hash changed')
    opts = settings(dict(cfg.get('settings', {}), publish='candidate_only'))
    manifest = inventory(suite, read(suite/'config.json'))
    if manifest['missing_cases'] or manifest['anomalies']:
        raise ValueError('Collection coverage incomplete; inspect inventory before replay')
    versions = defaultdict(list)
    found = False
    for row in manifest['episodes']:
        archived = Path(row['archive'])/'source_snapshot/hybrid_rollout/robodojo'
        hashes = fingerprints(archived)
        found |= archived.resolve() == source
        versions[digest(hashes)].append(dict(case_id=row['case_id'], archive=row['archive'], hashes=hashes))
    if not found:
        raise ValueError('Historical baseline must belong to a selected episode in this suite')
    identity = dict(suite_root=str(suite), baseline_source=str(source),
                    baseline_sha256=cfg['baseline_sha256'], settings=opts, inventory=manifest,
                    code_sha256=digest({p.name:p.read_text() for p in HERE.iterdir() if p.suffix in ('.py','.md')}))
    root = output/'runs'/('run_'+digest(identity)[:16])
    root.mkdir(parents=True, exist_ok=True)
    before = fingerprints(ROOT)
    write_json(root/'replay_manifest.json', identity)
    write_json(root/'prompt_versions.json', [dict(version=k, episodes=len(v), records=v) for k,v in versions.items()])
    frozen = root/'historical_source'
    frozen.mkdir(exist_ok=True)
    for name in FILES:
        target = frozen/name
        data = (source/name).read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Frozen historical source was modified')
        target.write_bytes(data)
    original = seed(frozen)
    export_text(root, 'before', original, original)
    (root/'historical_vs_current.diff').write_text(''.join(
        ''.join(difflib.unified_diff((source/name).read_text().splitlines(True),
            (ROOT/name).read_text().splitlines(True), fromfile='historical/'+name, tofile='current/'+name))
        for name in FILES))
    write_json(output/'latest.json', dict(run_root=str(root), experiment_only=True))
    print(json.dumps(dict(event='prompt_replay_output', run_root=str(root), episodes=len(manifest['episodes']),
                         historical_versions=len(versions), publish='candidate_only'),ensure_ascii=False),flush=True)
    try:
        result = run(suite, overrides=opts, prepare_only=prepare_only, backend=backend,
                     output=root/'analysis', historical_source=frozen, experiment_only=True)
        candidate = root/'analysis/candidate_bundle.json'
        if candidate.exists():
            export_text(root, 'after', read(candidate), original)
            (root/'prompt_changes.diff').write_text(''.join(
                ''.join(difflib.unified_diff(p.read_text().splitlines(True),
                    (root/'after'/p.name).read_text().splitlines(True),
                    fromfile='before/'+p.name,tofile='after/'+p.name))
                for p in sorted((root/'before').glob('*.md'))))
        write_json(root/'status.json', result)
        return dict(result, replay_root=str(root))
    finally:
        if fingerprints(ROOT) != before:
            raise RuntimeError('Live prompt files changed during replay; investigate concurrent edits')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'replay.round1.json')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    result = replay(args.config,prepare_only=args.prepare_only)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['state'] in ('prepared','candidate_only','no_change') else 2


if __name__ == '__main__':
    raise SystemExit(main())
