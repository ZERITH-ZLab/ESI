"""Frozen-registry compatibility without GPUs, credentials or network requests."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from .auth_profile_compat import prepare
from .deployment import source_identity


def frozen_env(tmp_path, name='codex_a_12'):
    root = tmp_path/'pool/source_snapshot'
    package = root/'hybrid_rollout/robodojo'
    (package/'codex_backend').mkdir(parents=True)
    for d in [root/'hybrid_rollout', package, package/'codex_backend']:
        (d/'__init__.py').write_text('')
    (package/'codex_backend/profiles.py').write_text('''
import os
PROFILES = {'codex_a': ('chatgpt','openai','https://chatgpt.com/backend-api/codex')}
CHATGPT_PROFILES = tuple(PROFILES)
def profile():
    name = os.environ['ROLLOUT_AUTH_PROFILE']
    return name, PROFILES[name]
''')
    (package/'probe.py').write_text('''
import json, os
from .codex_backend.profiles import profile, CHATGPT_PROFILES
print(json.dumps(dict(profile=profile(), registered=CHATGPT_PROFILES,
    home=os.environ['ROLLOUT_CODEX_HOME_DIR'], lock=os.fstat(int(os.environ['ROLLOUT_ACCOUNT_LOCK_FD'])).st_ino)))
''')
    return dict(os.environ, CODE_ROOT=str(root), PYTHONPATH=str(root),
        ROBODOJO_PYTHON=sys.executable, ROLLOUT_AUTH_PROFILE=name,
        ROLLOUT_CODEX_HOME_DIR=f'/private/{name}/codex_home',
        ROLLOUT_RESUME_IDENTITY=json.dumps(dict(source_sha256=source_identity(root))))


@pytest.mark.parametrize('name', [f'codex_a_{i}' for i in range(12, 16)] +
                         [f'codex_b_{i}' for i in range(8, 16)])
def test_new_profile_nested_snapshot_preserves_identity_credentials_and_lock(tmp_path, name):
    env = frozen_env(tmp_path, name)
    before = source_identity(env['CODE_ROOT'])
    wrapped = prepare(env)
    assert wrapped['ROLLOUT_AUTH_PROFILE'] == name
    # Simulate the entrypoint's second snapshot and PYTHONPATH replacement.
    episode = tmp_path/'episode/source_snapshot'
    shutil.copytree(env['CODE_ROOT'], episode)
    wrapped.update(CODE_ROOT=str(episode), PYTHONPATH=str(episode))
    with (tmp_path/'account.lock').open('w') as lock:
        wrapped['ROLLOUT_ACCOUNT_LOCK_FD'] = str(lock.fileno())
        result = subprocess.run([wrapped['ROBODOJO_PYTHON'], '-u', '-m',
            'hybrid_rollout.robodojo.probe'], env=wrapped, cwd=tmp_path,
            pass_fds=(lock.fileno(),), capture_output=True, text=True, check=True)
        value = json.loads(result.stdout.splitlines()[-1])
        assert value['profile'][0] == name and name in value['registered']
        assert value['home'] == env['ROLLOUT_CODEX_HOME_DIR']
        assert value['lock'] == os.fstat(lock.fileno()).st_ino
    assert source_identity(env['CODE_ROOT']) == source_identity(episode) == before
    manifest = json.loads(next((Path(env['CODE_ROOT']).parent/'auth_runtime').glob('*/manifest.json')).read_text())
    assert manifest['source_sha256'] == before
    assert 'auth.json' not in json.dumps(manifest)


def test_existing_profile_needs_no_shim(tmp_path):
    env = frozen_env(tmp_path, 'codex_a')
    assert prepare(env) is env
    assert not (Path(env['CODE_ROOT']).parent/'auth_runtime').exists()


def test_unknown_profile_is_not_implicitly_accepted(tmp_path):
    with pytest.raises(ValueError, match='registered ChatGPT'):
        prepare(frozen_env(tmp_path, 'codex_a_999'))


def test_changed_registry_fails_closed(tmp_path):
    env = prepare(frozen_env(tmp_path))
    path = Path(env['CODE_ROOT'])/'hybrid_rollout/robodojo/codex_backend/profiles.py'
    path.write_text(path.read_text()+'\n# changed\n')
    result = subprocess.run([env['ROBODOJO_PYTHON'], '-m', 'hybrid_rollout.robodojo.probe'],
        env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and 'Frozen account registry changed' in result.stderr


def test_regular_python_commands_still_work(tmp_path):
    env = prepare(frozen_env(tmp_path))
    result = subprocess.run([env['ROBODOJO_PYTHON'], '-c', 'print("READY")'],
        env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'READY'
