"""Explicit, source-only collection upgrades; never rewrite predecessor identities."""
import hashlib
import json
from pathlib import Path
import time

from .io import write_json


def compatible(old, new):
    return (isinstance(old, dict) and bool(old.get('source_sha256'))
            and {k: v for k, v in old.items() if k != 'source_sha256'} ==
                {k: v for k, v in new.items() if k != 'source_sha256'})


def accepted_identities(base, runtime):
    key = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    path = Path(base)/'_resume_upgrades'/f'{key}.json'
    if not path.exists():
        return [runtime]
    record = json.loads(path.read_text())
    if record.get('target_identity') != runtime or not all(
            compatible(old, runtime) for old in record.get('predecessor_identities', [])):
        raise ValueError('Invalid collection upgrade receipt')
    return [runtime, *record['predecessor_identities']]


def prepare(env):
    if env.get('ROLLOUT_ALLOW_SOURCE_UPGRADE') != '1':
        return
    if env.get('ROLLOUT_RUN_MODE') != 'resume' or not env.get('ROLLOUT_ACCOUNT_POOL_FILE'):
        raise ValueError('Source upgrade requires account-pool collection resume')
    from .collection import SCHEMA
    from .evaluation import read_panel
    panel = read_panel(env['ROLLOUT_EVAL_MANIFEST'])
    if panel.get('schema') != SCHEMA:
        raise ValueError('Source upgrade is only for training collection, not evaluation')
    base = Path(env['ROLLOUT_SHARED_ROOT'])/'results'/env['ROLLOUT_EXPERIMENT_ID']
    runtime = json.loads(env['ROLLOUT_RESUME_IDENTITY'])
    paths = list(base.glob('_replicas/replica_*/attempt_*/batch.json'))
    paths += list(base.glob('_account_pool/run_*/pool_run.json'))
    old_identities = []
    for path in paths:
        old = json.loads(path.read_text()).get('resume_identity')
        if not compatible(old, runtime):
            raise ValueError(f'Upgrade changes more than source/prompt: {path}')
        if old != runtime and old not in old_identities:
            old_identities.append(old)
    if not paths:
        raise ValueError('No prior collection receipts to upgrade')
    key = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    directory = base/'_resume_upgrades'
    directory.mkdir(exist_ok=True)
    path = directory/f'{key}.json'
    if not path.exists():
        write_json(path, dict(schema='collection_source_upgrade.v1', created_at=time.time(),
            target_identity=runtime, predecessor_identities=old_identities,
            scope='completed_sampling_items_only; task memory retains exact identity matching'))
    accepted_identities(base, runtime)
    print(json.dumps(dict(event='collection_source_upgrade', receipt=str(path),
                          predecessor_versions=len(old_identities))), flush=True)
