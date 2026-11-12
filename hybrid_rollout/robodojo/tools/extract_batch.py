"""Freeze a completed-episode selection, extract incrementally, never modify rollouts.

Selection order is completion time (not layout order). Source images stay referenced.
Failed/interrupted export staging directories are preserved for diagnostics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import uuid

from .extract_episode import VERSION, extract, read, require, write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(archive):
    """Fast change detection; RGB contents are hashed by the single-episode extractor."""
    files = [archive/'component_reset.json', archive/'collection_job.json',
             archive/'sim/summary.json', archive/'sim/evaluation_outcome.json',
             archive/'controller/history.json']
    s = read(archive/'sim/summary.json')
    sim = archive/'sim'/s['episode_id']
    files += sorted(sim.glob('action_*.json'))
    files += sorted((sim/'observations').glob('*.npz'))
    files += sorted((archive/'controller').glob('execution_*.npz'))
    files += [p for p in [archive/'policy_identity.json', archive/'resume_identity.json',
              archive/'evaluation_case.json', archive/'controller/result.json',
              archive/'controller/scene_query_usage.json', archive/'controller/controller_protocol.json'] if p.exists()]
    rows = []
    for p in files:
        stat = p.stat()
        rows.append((str(p.relative_to(archive)), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def discover(root):
    candidates, skipped = [], []
    # Fixed-depth glob deliberately avoids traversing source snapshots and image trees.
    for sample in root.glob('*/*/eval_seed_*/layout_*/replica_*/repeat_*/sample.json'):
        try:
            data = read(sample)
            require(data.get('state') in ('completed', 'failed', 'teacher_abandoned'), 'Sample still running or pending')
            require(isinstance(data.get('archive'), str) and bool(data['archive']), 'No finalized archive')
            archive = Path(data['archive']).resolve()
            require(archive.is_relative_to(root) and archive.parent == sample.parent,
                    'Invalid sample archive')
            attempt = next(a for a in reversed(data['attempts']) if a.get('archive') == str(archive))
            require(attempt.get('finished_at') is not None, 'Attempt still running')
            reset = read(archive/'component_reset.json')
            require(reset.get('all_owned_processes_exited') is True and reset.get('ports_released') is True,
                    'Process cleanup not complete')
            summary = read(archive/'sim/summary.json')
            require(summary.get('reason') in ('terminal', 'teacher_abandon'),
                    'Nonterminal interruption: excluded from basic export')
            candidates.append(dict(archive=str(archive), relative=str(archive.relative_to(root)),
                                   finished_at=attempt['finished_at']))
        except (ValueError, TypeError, KeyError, OSError, StopIteration) as exc:
            skipped.append(dict(sample=str(sample), reason=str(exc)))
    candidates.sort(key=lambda r: (r['finished_at'], r['relative']))
    return candidates, skipped


def one(row, output):
    archive, destination = Path(row['archive']), output/'episodes'/row['relative']
    result = dict(archive=str(archive), output=str(destination))
    try:
        require(signature(archive) == row['source_signature'], 'Source changed since selection; use a new batch')
        if destination.exists():
            manifest = read(destination/'manifest.json')
            require(manifest['schema'] == VERSION and manifest['source_archive'] == str(archive),
                    'Existing export identity/version differs')
            for name, checksum in manifest['file_sha256'].items():
                require(digest(destination/name) == checksum, f'Export corrupted: {name}')
            return dict(result, state='reused', steps=manifest['steps'])
        staging = output/'staging'/uuid.uuid4().hex
        report = extract(archive, staging)
        require(signature(archive) == row['source_signature'], 'Source changed during extraction')
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
        return dict(result, state='extracted', steps=report['steps'])
    except Exception as exc:
        return dict(result, state='failed', error=f'{type(exc).__name__}: {exc}')


def run(root, output, limit=8, workers=2):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root.is_dir() and not output.is_relative_to(root), 'Output must be outside source experiment')
    require(limit > 0 and 1 <= workers <= 4, 'Invalid limit/workers')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = output/'selection.json'
        if plan_path.exists():
            plan = read(plan_path)
            require(plan['source'] == str(root) and plan['limit'] == limit and plan['version'] == VERSION,
                    'Existing selection differs; use a new output directory')
        else:
            rows, skipped = discover(root)
            require(len(rows) >= limit, f'Only {len(rows)} finalized eligible episodes; requested {limit}')
            for row in rows[:limit]:
                row['source_signature'] = signature(Path(row['archive']))
            plan = dict(version=VERSION, source=str(root), limit=limit,
                        order='finished_at ascending', selected=rows[:limit], skipped=skipped)
            write(plan_path, plan)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda row: one(row, output), plan['selected']))
        index = dict(version=VERSION, results=results, total_steps=sum(r.get('steps', 0) for r in results),
                     failed=sum(r['state'] == 'failed' for r in results))
        temporary = output/f'.index.{uuid.uuid4().hex}.json'
        write(temporary, index)
        temporary.replace(output/'index.json')
        print(json.dumps(index, ensure_ascii=False, indent=2), flush=True)
        return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    raise SystemExit(1 if run(args.experiment, args.output, args.limit, args.workers)['failed'] else 0)


if __name__ == '__main__':
    main()
