"""Small deployment helpers; no policy or task semantics."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .io import sha256, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


LEGACY_SOURCE_SCOPE = 'legacy_v1'
RUNTIME_SOURCE_SCOPE = 'runtime_v2'


def source_scope(root, scope=None):
    if scope is None:
        manifest = Path(root)/'source_manifest.json'
        scope = json.loads(manifest.read_text()).get('source_scope', LEGACY_SOURCE_SCOPE) if manifest.is_file() else LEGACY_SOURCE_SCOPE
    if scope not in (LEGACY_SOURCE_SCOPE, RUNTIME_SOURCE_SCOPE):
        raise ValueError(f'Unknown source snapshot scope: {scope}')
    return scope


def runtime_resource(rel):
    """Code plus execution resources, excluding research recipes and prose docs.

    Intentionally conservative for Python/shell code. Changing an actual prompt
    must change identity; changing a README or an unused experiment JSON must not.
    """
    if rel.suffix in {'.py', '.sh', '.otf', '.ttf'}:
        return True
    if rel.name == 'requirements-motion.txt':
        return True
    if rel.suffix == '.toml':
        return 'codex_backend' in rel.parts
    if rel.suffix == '.png':
        return 'task_references' in rel.parts
    if rel.suffix == '.json':
        return ('collision_assets' in rel.parts or 'eval_panels' in rel.parts or
                rel.name in {'motion_distribution_profiles.json', 'BASELINE.json'})
    if rel.suffix == '.md':
        return (rel.name in {'SKILL.md', 'gate_prompt.md', 'feedforward_prompt.md',
                            'feedforward_gate_prompt.md', 'feedforward_freeform_prompt.md',
                            'manipulation_guidance.md', 'task_completion_briefs.md',
                            'task_success_conditions.md'} or
                'context' in rel.parts or
                ('prompt_evolution' in rel.parts and rel.name != 'README.md'))
    return False


def source_files(root, *, scope=None):
    root = Path(root)
    scope = source_scope(root, scope)
    candidates = list((root/'hybrid_rollout/robodojo').rglob('*'))
    candidates += list((root/'hybrid_rollout/assets/fonts').rglob('*'))
    candidates += [root/'hybrid_rollout/__init__.py']
    for path in sorted(set(candidates)):
        rel = path.relative_to(root)
        if (not path.is_file() or path.is_symlink() or
                any(p.startswith('.') or p == '__pycache__' for p in rel.parts) or
                path.name.startswith('test_') or 'tools' in rel.parts or
                path.name in {'launch_dual.sh', 'launch_collection.sh', 'DUAL_MACHINE.md'}):
            continue
        if scope == RUNTIME_SOURCE_SCOPE and not runtime_resource(rel):
            continue
        if (path.suffix in {'.py', '.sh', '.md', '.json', '.toml', '.otf', '.ttf'}
                or path.name == 'requirements-motion.txt'
                or (path.suffix == '.png' and 'task_references' in rel.parts)):
            yield path, rel


def source_identity(root, *, scope=None):
    return digest({str(rel): sha256(path) for path, rel in source_files(root, scope=scope)})


def snapshot(root, destination, *, scope=None):
    scope = source_scope(root, scope)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for path, rel in source_files(root, scope=scope):
        target = destination/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[str(rel)] = sha256(target)
    write_json(destination/'source_manifest.json', dict(files=files, sha256=digest(files), source_scope=scope))
    return digest(files)


def verify_snapshot(root):
    """Verify legacy or release snapshots without trusting only the stored hash."""
    root = Path(root)
    record = json.loads((root/'source_manifest.json').read_text())
    if record['sha256'] != digest(record['files']) or record['sha256'] != source_identity(root):
        raise ValueError(f'Pinned episode source snapshot changed: {root}')
    return record['sha256']


def native_identity(root):
    """Tracked changes/submodules plus untracked source; never export file contents."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    extra = {name: sha256(Path(root)/name) for name in untracked
             if name and Path(name).suffix in {'.py', '.yml', '.yaml', '.toml'}
             and (Path(root)/name).is_file() and not (Path(root)/name).is_symlink()}
    return digest(dict(head=git('rev-parse', 'HEAD').decode().strip(),
        diff=hashlib.sha256(git('diff', 'HEAD', '--submodule=diff')).hexdigest(),
        submodules=git('submodule', 'status', '--recursive').decode(), untracked=extra))


def prepare_episode(env):
    archive = Path(env['ROLLOUT_ARCHIVE'])
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'dual_machine', 'sim'):
        (archive/name).mkdir()
    identity = json.loads(env.get('ROLLOUT_RESUME_IDENTITY', 'null'))
    if identity:
        if native_identity(env['ROBODOJO_SOURCE']) != identity['native_a100']:
            raise ValueError('A100 RoboDojo source changed during campaign')
        if native_identity(env['OPENPI_SOURCE']) != identity['openpi']:
            raise ValueError('OpenPI source changed during campaign')
    frozen_hash = snapshot(env['CODE_ROOT'], archive/'source_snapshot')
    if identity and frozen_hash != identity['source_sha256']:
        raise ValueError('Frozen source identity changed')
    write_json(archive/'resume_identity.json', identity)
    write_json(archive/'archive_location.json', dict(schema='repeat_attempt_v1',
        repeat_id=int(env.get('ROLLOUT_COLLECTION_REPEAT_ID', '1')),
        sample_attempt=env.get('ROLLOUT_SAMPLE_ATTEMPT', env['ROLLOUT_ATTEMPT']),
        batch_attempt=env['ROLLOUT_ATTEMPT'], batch_path=env.get('ROLLOUT_BATCH_RECEIPT')))
    if env.get('ROLLOUT_COLLECTION_REPEAT_ID'):
        write_json(archive/'collection_job.json', dict(
            repeat_id=int(env['ROLLOUT_COLLECTION_REPEAT_ID']),
            job_index=int(env['ROLLOUT_COLLECTION_JOB_INDEX']),
            **({'suite_job_index': int(env['ROLLOUT_SUITE_JOB_INDEX'])}
               if env.get('ROLLOUT_SUITE_JOB_INDEX') else {}),
            case_id=env['ROLLOUT_CASE_ID'], attempt=env.get('ROLLOUT_SAMPLE_ATTEMPT', env['ROLLOUT_ATTEMPT']),
            batch_attempt=env['ROLLOUT_ATTEMPT'], archive_layout='repeat_attempt_v1',
            auth_profile=env.get('ROLLOUT_AUTH_PROFILE'),
            worker_id=env.get('ROLLOUT_WORKER_ID'),
            step_limit=int(env['ROLLOUT_COLLECTION_STEP_LIMIT']),
            scope='training_collection_not_official_evaluation'))
    shutil.copyfile(env['ROLLOUT_CASE_FILE'], archive/'evaluation_case.json')
    shutil.copyfile(Path(env['ROLLOUT_CASE_FILE']).parent/'evaluation_manifest.json',
                    archive/'evaluation_manifest.json')


def verify_policy(archive):
    archive = Path(archive)
    expected = json.loads((archive/'resume_identity.json').read_text())
    actual = json.loads((archive/'policy_identity.json').read_text())
    if expected and expected['checkpoint_sha256'] != actual['checkpoint_sha256']:
        raise ValueError('Loaded policy checkpoint differs from campaign identity')
    if expected and expected.get('eval_protocol'):
        case = json.loads((archive/'evaluation_case.json').read_text())['case']
        if actual.get('policy_rng_seed') != case['policy_rng_seed'] or actual.get('inferences') != 0:
            raise ValueError('Eval requires the paired policy seed and a fresh policy process')


if __name__ == '__main__':
    import os
    import sys
    if sys.argv[1:] == ['verify-policy']:
        verify_policy(os.environ['ROLLOUT_ARCHIVE'])
    else:
        prepare_episode(os.environ)
