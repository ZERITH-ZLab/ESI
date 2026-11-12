"""Combine immutable indexes without copying trajectories or changing eligibility."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile

from data_processing.subtasks.io import digest, read, require, write
from data_processing.subtasks.pipeline import batch_lock
from . import VERSION
from .build import SCHEMA, checksum, validate


CONTRACT_KEYS = ('horizon', 'min_valid', 'stride', 'selection_rule', 'supervision',
                 'interior_tail', 'episode_padding', 'attention_mask', 'action_label', 'instruction')


def merge_indexes(inputs, output):
    inputs = [Path(p).resolve() for p in inputs]
    output = Path(output).resolve()
    require(len(inputs) >= 2 and len(set(inputs)) == len(inputs), 'Need distinct source indexes')
    require(not output.is_relative_to(Path(__file__).resolve().parents[2]), 'Write index outside repository')
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in inputs),
            'Output must not overlap source indexes')
    manifests = []
    for p in inputs:
        validate(p, check_sources=True)
        manifests.append(read(p/'manifest.json'))
    cfg = {k: manifests[0]['config'][k] for k in CONTRACT_KEYS}
    require(all({k: m['config'][k] for k in CONTRACT_KEYS} == cfg for m in manifests),
            'Source indexes have different supervision/window contracts')
    cfg['merged_indexes'] = [dict(path=str(p), index_sha256=m['files_sha256']['index.sqlite'])
                             for p, m in zip(inputs, manifests)]
    implementation = checksum(__file__)
    with batch_lock(output):
        if (output/'manifest.json').exists():
            old = read(output/'manifest.json')
            require(old['config'] == cfg and old['implementation_sha256'] == implementation,
                    'Merged index changed; choose a new output')
            return dict(state='reused', output=str(output), **validate(output))
        require(not (output/'index.sqlite').exists(), 'Unpublished index exists; choose new output')
        with tempfile.TemporaryDirectory(prefix='.merging-', dir=output) as tmp:
            tmp = Path(tmp)
            with closing(sqlite3.connect(tmp/'index.sqlite', uri=True)) as db:
                db.executescript(SCHEMA)
                source_counts = []
                for p, manifest in zip(inputs, manifests):
                    db.execute('ATTACH DATABASE ? AS src', (f'{(p/"index.sqlite").as_uri()}?mode=ro',))
                    require(not db.execute('SELECT 1 FROM episodes e JOIN src.episodes s '
                        'ON e.episode_key=s.episode_key OR e.episode_id=s.episode_id LIMIT 1').fetchone(),
                        'Duplicate episode across indexes')
                    require(not db.execute('SELECT 1 FROM sources a JOIN src.sources b ON a.path=b.path '
                        'WHERE a.sha256!=b.sha256 LIMIT 1').fetchone(), 'Conflicting source hashes')
                    offset = db.execute('SELECT COALESCE(MAX(sample_id)+1,0) FROM samples').fetchone()[0]
                    db.execute('INSERT INTO episodes SELECT * FROM src.episodes')
                    db.execute('INSERT INTO steps SELECT * FROM src.steps')
                    columns = [r[1] for r in db.execute('PRAGMA table_info(samples)')]
                    db.execute('INSERT INTO samples SELECT sample_id+?, '+','.join(columns[1:])+' FROM src.samples', (offset,))
                    db.execute('INSERT OR IGNORE INTO sources SELECT * FROM src.sources')
                    for name, sha in dict(manifest['files_sha256'], **{'manifest.json': checksum(p/'manifest.json')}).items():
                        db.execute('INSERT INTO sources VALUES (?,?)', (str(p/name), sha))
                    source_counts.append(dict(path=str(p), sample_id_offset=offset,
                        **{t: db.execute(f'SELECT COUNT(*) FROM src.{t}').fetchone()[0]
                           for t in ('episodes', 'steps', 'samples')}))
                    db.commit()
                    db.execute('DETACH DATABASE src')
                require(not db.execute('PRAGMA foreign_key_check').fetchall(), 'Invalid merged relationships')
                require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Invalid merged SQLite')
                counts = {t: db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                          for t in ('episodes', 'steps', 'samples')}
                rounds = [dict(collection_round=r, policy_version=p, episodes=n)
                          for r, p, n in db.execute('SELECT collection_round,policy_version,COUNT(*) '
                                                  'FROM episodes GROUP BY 1,2')]
                hashes = dict(db.execute('SELECT path,sha256 FROM sources'))
            write(tmp/'report.json', dict(version=VERSION, config=cfg, totals=counts,
                rounds=rounds, sources=source_counts,
                note='Only sample IDs reassigned. Episode identity, round, policy, labels, windows and masks preserved.'))
            files = {n: checksum(tmp/n) for n in ('index.sqlite', 'report.json')}
            for n in files:
                (tmp/n).replace(output/n)
            write(output/'manifest.json', dict(version=VERSION, status='completed', config=cfg,
                implementation_sha256=implementation, files_sha256=files, source_fingerprint=digest(hashes)))
        return dict(state='merged', output=str(output), **validate(output, check_sources=False))
