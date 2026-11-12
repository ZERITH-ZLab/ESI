"""Join audited labels with existing command exports; never copy RGB or actions."""
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from data_processing.subtasks.audit import validate_final
from data_processing.subtasks.io import digest, read, require, write
from data_processing.subtasks.pipeline import batch_lock, load_input
from . import VERSION
from .windows import select_windows


SCHEMA = '''
PRAGMA foreign_keys=ON;
CREATE TABLE episodes (
 episode_key TEXT PRIMARY KEY, episode_id TEXT UNIQUE NOT NULL, source_archive TEXT NOT NULL,
 annotation_path TEXT NOT NULL, annotation_version TEXT NOT NULL, input_hash TEXT NOT NULL,
 source_batch TEXT NOT NULL, policy_version TEXT NOT NULL, collection_round TEXT NOT NULL,
 checkpoint_sha256 TEXT NOT NULL, task TEXT NOT NULL, variant TEXT NOT NULL,
 episode_success INTEGER NOT NULL, collected_at_json TEXT NOT NULL,
 total_steps INTEGER NOT NULL, control_dt REAL NOT NULL, task_instruction TEXT NOT NULL,
 steps_npz TEXT NOT NULL, action_array TEXT NOT NULL, state_array TEXT NOT NULL,
 observations_json TEXT NOT NULL, mcap_path TEXT, lerobot_root TEXT,
 shard_episode_index INTEGER NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE steps (
 episode_key TEXT NOT NULL REFERENCES episodes, step INTEGER NOT NULL,
 decision_id INTEGER NOT NULL, is_decision_start INTEGER NOT NULL,
 control_type TEXT NOT NULL, segment_id TEXT NOT NULL, skill_type TEXT,
 outcome TEXT NOT NULL, needs_review INTEGER NOT NULL, training_valid INTEGER NOT NULL,
 high_level_eligible INTEGER NOT NULL, review_reasons_json TEXT NOT NULL,
 observation_relpath TEXT NOT NULL, observation_sha256 TEXT NOT NULL,
 action_receipt_relpath TEXT NOT NULL,
 PRIMARY KEY (episode_key,step)
);
CREATE TABLE samples (
 sample_id INTEGER PRIMARY KEY, episode_key TEXT NOT NULL, start_step INTEGER NOT NULL,
 valid_end_step INTEGER NOT NULL, valid_length INTEGER NOT NULL, real_length INTEGER NOT NULL,
 pad_length INTEGER NOT NULL, gpt_valid_steps INTEGER NOT NULL, selection TEXT NOT NULL,
 UNIQUE (episode_key,start_step), FOREIGN KEY(episode_key,start_step) REFERENCES steps(episode_key,step)
);
CREATE TABLE sources (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
CREATE INDEX steps_control ON steps(control_type,training_valid);
CREATE VIEW training_samples AS
 SELECT w.*, s.control_type AS start_control, s.decision_id, s.is_decision_start,
 s.segment_id, s.skill_type, s.outcome AS subtask_outcome, s.high_level_eligible,
 s.observation_relpath, s.observation_sha256,
 e.episode_id, e.source_archive, e.episode_success, e.variant, e.task,
 e.policy_version, e.collection_round, e.source_batch, e.collected_at_json,
 e.steps_npz, e.action_array, e.state_array
 FROM samples w JOIN steps s ON s.episode_key=w.episode_key AND s.step=w.start_step
 JOIN episodes e ON e.episode_key=w.episode_key;
'''


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Sources:
    def __init__(self):
        self.hashes = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        value = checksum(path)
        require(expected is None or value == expected, f'Source fingerprint mismatch: {path}')
        require(str(path) not in self.hashes or self.hashes[str(path)] == value,
                f'Source changed while indexing: {path}')
        self.hashes[str(path)] = value
        return path

    def json(self, path, expected=None):
        return read(self.add(path, expected))

    def verify(self):
        for path, expected in self.hashes.items():
            require(checksum(path) == expected, f'Source changed while indexing: {path}')


def export_catalog(roots, sources):
    result = {}
    for root in roots:
        collection = sources.json(root/'dataset_collection.json')
        require(collection['status'] == 'completed' and collection['action_label'] == 'command',
                f'Require completed command export: {root}')
        if collection.get('schema') == 'robodojo_command_collection.v1':
            require(len(collection['episodes']) == collection['total_episodes'], 'Export count differs')
            for row in collection['episodes']:
                sidecar = (root/row['sidecar']).resolve()
                require(sidecar.is_relative_to(root) and row['state'] == 'completed', 'Invalid command sidecar')
                eid = row['episode_id']
                require(eid not in result, f'Duplicate exported episode: {eid}')
                result[eid] = dict(folder=root, row=row, mcap=None, sidecar=sidecar)
            continue
        count = 0
        for shard in collection['shards']:
            folder = (root/shard['shard']).resolve()
            require(folder.is_relative_to(root), 'Shard outside export root')
            manifest = sources.json(folder/'export_manifest.json')
            require(manifest['status'] in ('passed', 'written') and manifest['action_label'] == 'command',
                    f'Invalid shard export: {folder}')
            require(manifest['episodes'] == shard['episodes'], 'Collection/shard episode mappings differ')
            for row in manifest['episodes']:
                eid = row['episode_id']
                require(eid not in result, f'Duplicate exported episode: {eid}')
                mcap = (folder/row['mcap']).resolve() if row.get('mcap') else None
                require(mcap is None or (mcap.is_relative_to(folder) and mcap.is_file()), 'Missing/invalid MCAP')
                result[eid] = dict(folder=folder, row=row, mcap=mcap,
                                   sidecar=folder/'research'/f"episode_{row['episode_index']:06d}")
                count += 1
        require(count == collection['total_episodes'], 'Export collection episode count differs')
    return result


def load_export(item, archive, annotation, prepared, sources):
    sidecar, n = item['sidecar'], annotation['total_steps']
    manifest = sources.json(sidecar/'manifest.json')
    require(manifest['schema'] in ('robodojo_training_extract.v2', 'robodojo_training_extract.v3'), 'Unsupported extraction schema')
    require(Path(manifest['source_archive']).resolve() == archive and
            manifest['episode_id'] == annotation['metadata']['episode_id'] and
            manifest['steps'] == item['row']['steps'] == n, 'Export/annotation identity or step count mismatch')
    for name, expected in manifest['file_sha256'].items():
        path = (sidecar/name).resolve()
        require(path.is_relative_to(sidecar), 'Unsafe export manifest path')
        sources.add(path, expected)
    metadata = read(sidecar/'episode.json')
    require(metadata['summary'] == read(archive/'sim/summary.json'), 'Export summary differs from raw summary')
    refs = read(sidecar/'observations.json')
    require(len(refs) == n+1 and [r['step'] for r in refs] == list(range(n+1)), 'Expected T+1 ordered observations')
    # One directory listing verifies every path without reading all RGB files.
    obsdir = archive/'sim'/annotation['metadata']['episode_id']/'observations'
    names = {p.name for p in obsdir.iterdir() if p.is_file()}
    for t, ref in enumerate(refs):
        expected = f"sim/{annotation['metadata']['episode_id']}/observations/{t:06d}.npz"
        require(ref['path'] == expected and f'{t:06d}.npz' in names, 'Missing/misaligned observation reference')
    arrays_path = sidecar/'steps.npz'
    with np.load(arrays_path, allow_pickle=False) as z:
        states, commands, owners = z['state'], z['command'], z['decision_id']
        require(states.shape == commands.shape == (n,14) and np.isfinite(states).all()
                and np.isfinite(commands).all(), 'Invalid state/command arrays')
        require(np.array_equal(z['step'], np.arange(n)) and owners.shape == (n,), 'Invalid array row ordering')
        require(np.all((commands[:,[6,13]] >= 0) & (commands[:,[6,13]] <= 1)), 'Invalid gripper actions')
        for d in prepared['payload']['decisions']:
            require(np.all(owners[d['start_step']:d['end_step']] == d['decision_id']), 'Decision/action row mismatch')
        # Re-read raw evidence at deterministic endpoints and first GPT start.
        probes = {0, n//2, n-1}
        first_gpt = next((d['start_step'] for d in prepared['payload']['decisions']
                          if d['control_type']=='gpt' and d['end_step']>d['start_step']), None)
        if first_gpt is not None:
            probes.add(first_gpt)
        for t in sorted(probes):
            path = sources.add(archive/refs[t]['path'], refs[t]['sha256'])
            with np.load(path, allow_pickle=False) as obs:
                np.testing.assert_allclose(states[t], obs['states'], atol=1e-6, rtol=0)
                require(str(obs['instruction'].item()) == metadata['task_instruction'], 'Task text differs')
                for camera in manifest['cameras']:
                    image = obs[camera]
                    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1]==3, 'Invalid sampled RGB')
            receipt = sources.json(archive/'sim'/annotation['metadata']['episode_id']/f'action_{t:06d}.json')
            require(receipt['valid'] and receipt['step_id']==t+1, 'Invalid action receipt alignment')
            np.testing.assert_allclose(commands[t], receipt['executed_action'], atol=1e-6, rtol=0)
    return manifest, metadata, refs, sorted(probes)


def insert(db, table, row):
    db.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))


def index_episode(db, row, batch, run, exported, sources, cfg):
    archive = Path(row['archive']).resolve()
    prepared_path = batch/row['path']
    sources.add(prepared_path)
    prepared = load_input(batch, row)
    for name, expected in prepared['metadata']['source_sha256'].items():
        sources.add(archive/name, expected)
    dest = run/'episodes'/row['key']
    annotation = sources.json(dest/'annotation.json')
    candidate = sources.json(dest/'candidate.json')
    audit = sources.json(dest/'audit_final.json')
    validator = validate_final
    if annotation.get('route') == 'online_subtask_outcome':
        from data_processing.online_subtask_outcomes.contract import validate_final as validator
    require(validator(prepared, candidate, audit) == annotation, f'Invalid audited annotation: {row["key"]}')
    require(annotation['input_hash']==row['input_hash'] and annotation['metadata']==prepared['metadata'], 'Annotation input mismatch')
    n, eid = annotation['total_steps'], annotation['metadata']['episode_id']
    require(n>0 and eid in exported, f'Nonempty episode has no export: {eid}')
    item = exported[eid]
    manifest, metadata, refs, probes = load_export(item, archive, annotation, prepared, sources)
    policy = sources.json(archive/'policy_identity.json')
    checkpoint = policy['checkpoint_sha256']
    require(policy['action_horizon']==cfg['horizon'] and policy['action_dim']==14, 'Policy action shape differs')
    require(metadata['provenance']['policy_identity']['checkpoint_sha256']==checkpoint, 'Export policy identity differs')
    case = metadata['summary']['evaluation_case']
    require(type(metadata['summary']['success']) is bool, 'Missing episode outcome')
    record = dict(episode_key=row['key'], episode_id=eid, source_archive=str(archive),
        annotation_path=str(dest/'annotation.json'), annotation_version=annotation['version'], input_hash=row['input_hash'],
        source_batch=archive.parts[-8], policy_version=cfg['policy_version'], collection_round=cfg['collection_round'],
        checkpoint_sha256=checkpoint, task=case['task'], variant=case['variant'],
        episode_success=int(metadata['summary']['success']), collected_at_json=json.dumps(annotation['metadata']['collected_at']),
        total_steps=n, control_dt=manifest['control_dt'], task_instruction=metadata['task_instruction'],
        steps_npz=str(item['sidecar']/'steps.npz'), action_array='command', state_array='state',
        observations_json=str(item['sidecar']/'observations.json'), mcap_path=str(item['mcap']) if item['mcap'] else None,
        lerobot_root=str(item['folder']/'lerobot') if (item['folder']/'lerobot').is_dir() else None,
        shard_episode_index=item['row']['episode_index'],
        metadata_json=json.dumps(dict(evaluation_case=case, policy_config=policy['config'], sampled_raw_steps=probes)))
    insert(db, 'episodes', record)
    valid, controls, decision_ids, decision_starts = [False]*n, ['unknown']*n, [-1]*n, [False]*n
    for d in prepared['payload']['decisions']:
        a,b = d['start_step'], d['end_step']
        if a<b:
            controls[a:b]=[d['control_type']]*(b-a)
            decision_ids[a:b]=[d['decision_id']]*(b-a)
            decision_starts[a]=True
    step_rows=[]
    for s in annotation['segments']:
        eligible = s['training_eligibility']['low_level']
        require(not eligible or (s['outcome']=='success' and not s['needs_review']), 'Invalid training eligibility')
        a,b=s['start_step'],s['end_step']
        valid[a:b]=[eligible]*(b-a)
        for t in range(a,b):
            step_rows.append((row['key'],t,decision_ids[t],int(decision_starts[t]),controls[t],s['segment_id'],
                (s.get('instruction') or {}).get('skill_type'),s['outcome'],int(s['needs_review']),int(eligible),
                int(s['training_eligibility']['high_level']),json.dumps(s['review_reasons']),refs[t]['path'],
                refs[t]['sha256'],f'sim/{eid}/action_{t:06d}.json'))
    require(len(step_rows)==n and all(d>=0 for d in decision_ids), 'Incomplete step ownership')
    db.executemany('INSERT INTO steps VALUES ('+','.join('?' for _ in range(15))+')',step_rows)
    runs, windows = select_windows(valid, controls, cfg['horizon'], cfg['min_valid'])
    for w in windows:
        insert(db,'samples',dict(episode_key=row['key'],**w))
    selected={w['start_step'] for w in windows}
    covered=[False]*n
    for w in windows:
        t=w['start_step']; covered[t:t+w['valid_length']]=[True]*w['valid_length']
    counts=Counter(episodes=1,steps=n,valid_steps=sum(valid),windows=len(windows),
                   full_windows=sum(w['selection']=='full' for w in windows),
                   masked_windows=sum(w['selection']=='gpt_tail' for w in windows))
    for control in ('gpt','pi05'):
        positions=[t for t in range(n) if valid[t] and controls[t]==control]
        calls=[t for t in positions if decision_starts[t]]
        counts.update({f'{control}_valid_steps':len(positions),
                       f'{control}_kept_starts':sum(t in selected for t in positions),
                       f'{control}_lost_starts':sum(t not in selected for t in positions),
                       f'{control}_uncovered_steps':sum(not covered[t] for t in positions),
                       f'{control}_valid_decision_starts':len(calls),
                       f'{control}_lost_decision_starts':sum(t not in selected for t in calls)})
    return dict(key=row['key'],episode_id=eid,task=case['task'],variant=case['variant'],episode_success=record['episode_success'],
                checkpoint_sha256=checkpoint,counts=dict(counts),effective_runs=runs)


def validate(output, check_sources=True):
    output=Path(output).resolve()
    manifest=read(output/'manifest.json')
    require(manifest['version']==VERSION and manifest['status']=='completed', 'Incomplete/unsupported index')
    for name, expected in manifest['files_sha256'].items():
        require(checksum(output/name)==expected, f'Index artifact modified: {name}')
    with closing(sqlite3.connect(f'{(output/"index.sqlite").as_uri()}?mode=ro',uri=True)) as db:
        require(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok', 'SQLite integrity failure')
        require(not db.execute('PRAGMA foreign_key_check').fetchall(), 'Invalid index relationships')
        if check_sources:
            for path, expected in db.execute('SELECT path,sha256 FROM sources'):
                require(checksum(path)==expected, f'Index source changed: {path}')
        counts={table:db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in ('episodes','steps','samples')}
    return dict(status='validated',source_checks=check_sources,**counts)


def build(run, exports, output, policy_version, collection_round, horizon=50, min_valid=15):
    run,output=Path(run).resolve(),Path(output).resolve()
    exports=[Path(p).resolve() for p in exports]
    require(policy_version.strip() and collection_round.strip(), 'Name policy version and collection round')
    require(1<=min_valid<=horizon, 'Invalid window lengths')
    batch=run.parent.parent
    cfg=dict(annotation_run=str(run),exports=[str(p) for p in exports],policy_version=policy_version,
             collection_round=collection_round,horizon=horizon,min_valid=min_valid,stride=1,
             selection_rule='extend_to_min_last_gpt_and_boundary_minus_min_valid',
             supervision='continuous_valid_prefix',interior_tail='real_actions',episode_padding='repeat_last',
             attention_mask='unchanged',action_label='executed_command',instruction='episode_task')
    implementation=digest({p.name:p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})
    require(not output.is_relative_to(Path(__file__).resolve().parents[2]), 'Write indexes outside repository')
    for source in [batch,*exports]:
        require(not output.is_relative_to(source), 'Output must be outside annotation/export sources')
    with batch_lock(output):
        if (output/'manifest.json').exists():
            old=read(output/'manifest.json')
            require(old['config']==cfg and old['implementation_sha256']==implementation,
                    'Index configuration/code changed; use a new output directory')
            return dict(state='reused',output=str(output),**validate(output))
        require(not (output/'index.sqlite').exists(), 'Unpublished index exists; use a new output directory')
        sources=Sources()
        manifest=sources.json(batch/'manifest.json')
        require(not manifest['failures'] and not manifest['excluded'], 'Annotation selection has unresolved preparation failures/exclusions')
        sources.json(run/'protocol.json'); sources.json(run/'config.json')
        rows=sorted(manifest['episodes'],key=lambda r:r['archive'])
        require(rows, 'No episodes selected')
        for row in rows:
            require(not output.is_relative_to(Path(row['archive']).resolve()), 'Output inside original archive')
        catalog=export_catalog(exports,sources)
        totals=Counter();groups=defaultdict(Counter);tasks=defaultdict(Counter);reports=[]
        with tempfile.TemporaryDirectory(prefix='.building-',dir=output) as staging:
            staging=Path(staging)
            with closing(sqlite3.connect(staging/'index.sqlite')) as db:
                db.executescript(SCHEMA)
                for row in rows:
                    report=index_episode(db,row,batch,run,catalog,sources,cfg)
                    reports.append(report); totals.update(report['counts'])
                    groups[report['variant']].update(report['counts'])
                    tasks[report['task']].update(report['counts'])
                    print(json.dumps(dict(event='index_episode',completed=len(reports),total=len(rows),
                                          key=row['key'],**report['counts'])),flush=True)
                require(len({r['checkpoint_sha256'] for r in reports})==1, 'Mixed checkpoints in one policy version')
                sources.verify()
                db.executemany('INSERT INTO sources VALUES (?,?)',sorted(sources.hashes.items()))
                db.commit()
                require(not db.execute('PRAGMA foreign_key_check').fetchall(), 'Invalid index relationships')
                require(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok', 'Invalid SQLite output')
            report=dict(version=VERSION,config=cfg,totals=dict(totals),by_variant=dict(groups),by_task=dict(tasks),episodes=reports,
                validation='All audited labels revalidated; exported arrays/metadata hashed; T+1 image paths checked; '
                'raw RGB/state/action alignment checked at recorded sampled steps. No full MCAP decode.',
                note='Index only. Masked samples require a mask-aware loader/loss before training; no model changes made.')
            write(staging/'report.json',report)
            files={name:checksum(staging/name) for name in ('index.sqlite','report.json')}
            for name in files:
                (staging/name).replace(output/name)
            write(output/'manifest.json',dict(version=VERSION,status='completed',config=cfg,
                  implementation_sha256=implementation,files_sha256=files,source_fingerprint=digest(sources.hashes)))
        return dict(state='built',output=str(output),**dict(totals))
