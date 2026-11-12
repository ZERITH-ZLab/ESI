"""Dispatcher-only account registration bridge for immutable episode snapshots.

No credentials, account aliases, policy imports from the working tree, or source
identity rewrites. The copied bootstrap only extends a frozen profile registry.
"""
import importlib
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import uuid


def prepare(env):
    from .codex_backend.profiles import PROFILES
    from .io import sha256, write_json

    name = env['ROLLOUT_AUTH_PROFILE']
    expected = PROFILES.get(name)
    if expected is None or expected[0:2] != ('chatgpt', 'openai'):
        raise ValueError('Account compatibility requires a registered ChatGPT profile')
    source = Path(env['CODE_ROOT']).resolve()
    module = source/'hybrid_rollout/robodojo/codex_backend/profiles.py'
    registered = runpy.run_path(str(module))['PROFILES']
    if name in registered:
        if registered[name] != expected:
            raise ValueError('Frozen account provider differs from dispatcher')
        return env
    # This lives next to, never inside, the immutable pool source snapshot.
    directory = source.parent/'auth_runtime'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    bootstrap = directory/'bootstrap.py'
    bootstrap.write_bytes(Path(__file__).read_bytes())
    interpreter = str(Path(env['ROBODOJO_PYTHON']).resolve())
    manifest = directory/'manifest.json'
    write_json(manifest, dict(version='frozen_auth_registry_v1', auth_profile=name,
        registration=list(expected), interpreter=interpreter,
        frozen_profiles_sha256=sha256(module), bootstrap_sha256=sha256(bootstrap),
        source_sha256=json.loads(env.get('ROLLOUT_RESUME_IDENTITY', '{}')).get('source_sha256')))
    launcher = directory/'python'
    launcher.write_text('#!/bin/sh\nexec ' + ' '.join(map(shlex.quote,
        [interpreter, '-u', str(bootstrap), str(manifest)])) + ' "$@"\n')
    launcher.chmod(0o700)
    print(json.dumps(dict(event='auth_profile_compat', auth_profile=name,
        manifest=str(manifest), frozen_source_unchanged=True)), flush=True)
    return dict(env, ROBODOJO_PYTHON=str(launcher))


def bootstrap():
    manifest = Path(sys.argv[1])
    record = json.loads(manifest.read_text())
    original = sys.argv[2:]
    args = list(original)
    while args and args[0] == '-u':
        args.pop(0)
    # Ordinary Python commands (including readiness probes) keep their semantics.
    if len(args) < 2 or args[0] != '-m' or not args[1].startswith('hybrid_rollout.robodojo.'):
        os.execv(record['interpreter'], [record['interpreter'], *original])
    name = os.environ.get('ROLLOUT_AUTH_PROFILE')
    if name != record['auth_profile']:
        raise ValueError('Compatibility profile changed within episode')
    root = Path(os.environ['CODE_ROOT']).resolve()
    # The shell advances CODE_ROOT through pool -> batch -> episode snapshots.
    # Never import policy code from the dispatcher working tree or this shim.
    sys.path[0] = str(root)
    profiles = importlib.import_module('hybrid_rollout.robodojo.codex_backend.profiles')
    module = root/'hybrid_rollout/robodojo/codex_backend/profiles.py'
    import hashlib
    if (Path(profiles.__file__).resolve() != module or
            hashlib.sha256(module.read_bytes()).hexdigest() != record['frozen_profiles_sha256']):
        raise ValueError('Frozen account registry changed')
    registration = tuple(record['registration'])
    if name in profiles.PROFILES and profiles.PROFILES[name] != registration:
        raise ValueError('Conflicting frozen account registration')
    profiles.PROFILES[name] = registration
    profiles.CHATGPT_PROFILES = tuple(k for k, v in profiles.PROFILES.items() if v[0] == 'chatgpt')
    print(json.dumps(dict(event='auth_profile_compat_loaded', auth_profile=name,
        module=args[1], manifest=str(manifest))), flush=True)
    sys.argv = args[1:]
    runpy.run_module(args[1], run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    bootstrap()
