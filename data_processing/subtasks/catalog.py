"""Readable navigation over immutable run/episode identities; never relabel data."""
import csv
import io
import os
from pathlib import Path
import re
import tempfile

from .io import read, require, write


def slug(value, limit=32):
    # Directory labels are navigation only; the complete hash disambiguates
    # repeated/truncated labels and archives from multiple collection roots.
    text = re.sub(r'[^a-zA-Z0-9_.-]+', '-', str(value)).strip('.-')
    return (text[:limit] or 'unspecified')


def episode_name(row):
    parts = Path(row['archive']).parts[-7:]
    return '__'.join([*(slug(p, 24) for p in parts), slug(row['key'], 24)])


def run_name(run_root):
    cfg = read(run_root / 'config.json')
    spec = read(run_root / 'protocol.json')
    return '__'.join(slug(v, 48) for v in
                     (spec['version'], cfg['model'], cfg['reasoning_effort'], run_root.name))


def link_directory(link, target):
    """Do not replace existing user directories or redirect an existing link."""
    if link.is_symlink():
        require(link.resolve() == target.resolve(), f'Navigation link points elsewhere: {link}')
        return
    require(not link.exists(), f'Navigation path already exists and is not a symlink: {link}')
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)


def write_csv(path, rows):
    fields = ['name', 'state', 'key', 'episode_id', 'source_archive', 'prepared_input',
              'annotation', 'gpt_review', 'episode_directory']
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: row.get(k) for k in fields} for row in rows)
    fd, temp = tempfile.mkstemp(prefix='.tmp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            stream.write(buf.getvalue())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def build_run_catalog(root, run_root, manifest):
    """File discovery only: present labels are not independently revalidated here."""
    label = run_name(run_root)
    link_directory(root / 'runs_by_name' / label, run_root)
    rows = []
    for row in manifest['episodes']:
        destination = run_root / 'episodes' / row['key']
        name = episode_name(row)
        if destination.is_dir():
            link_directory(run_root / 'episodes_by_name' / name, destination)
        annotation = destination / 'annotation.json'
        status = destination / 'status.json'
        state = (read(status).get('state', 'unknown') if status.exists() else
                 'annotation_present' if annotation.is_file() else
                 'incomplete' if destination.is_dir() else 'not_started')
        rows.append(dict(name=name, state=state, key=row['key'], episode_id=row['episode_id'],
                         source_archive=row['archive'], prepared_input=str(root / row['path']),
                         annotation=str(annotation) if annotation.is_file() else None,
                         gpt_review=str(destination / 'gpt_review.json')
                         if (destination / 'gpt_review.json').is_file() else None,
                         episode_directory=str(destination)))
    result = dict(run_id=run_root.name, name=label, run_root=str(run_root),
                  named_run_root=str(root / 'runs_by_name' / label),
                  segments=str(run_root / 'segments.json'),
                  note='Navigation only; drafts and failed episodes are not training labels. '
                       'Use segments.json and training_eligibility for sampling.', episodes=rows)
    write(run_root / 'episodes.index.json', result)
    write_csv(run_root / 'episodes.index.csv', rows)
    return result


def build_catalog(root, run_id=None):
    # Late import avoids a cycle with run_batch's automatic catalogue update.
    from .pipeline import batch_lock
    root = Path(root).resolve()
    require((root / 'manifest.json').is_file(), 'Batch manifest does not exist')
    if run_id is not None:
        require(bool(re.fullmatch(r'[a-zA-Z0-9_-]+', run_id)), 'Invalid run ID')
    with batch_lock(root):
        manifest = read(root / 'manifest.json')
        runs = [root / 'runs' / run_id] if run_id else sorted(
            p for p in (root / 'runs').iterdir() if p.is_dir() and not p.is_symlink())
        require(bool(runs), 'No runs found')
        results = []
        for run in runs:
            require((run / 'config.json').is_file() and (run / 'protocol.json').is_file(),
                    f'Run configuration missing: {run}')
            catalog = build_run_catalog(root, run, manifest)
            results.append(dict(run_id=run.name, name=catalog['name'],
                                index=str(run / 'episodes.index.csv'),
                                named_run_root=catalog['named_run_root'],
                                episodes=len(catalog['episodes'])))
        return dict(runs=results, model_calls=0)
