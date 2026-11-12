"""Annotation account selection and leases for persistent conversations."""
from contextlib import contextmanager
import json
import time
from pathlib import Path


def annotation_accounts(path):
    from ..codex_backend.profiles import CHATGPT_PROFILES
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    config = json.loads(path.read_text())
    if config.get('backend') != 'codex':
        raise ValueError('Prompt evolution requires Codex annotation accounts')
    names = config.get('profiles')
    workers = config.get('workers')
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not all(n in CHATGPT_PROFILES for n in names)
            or type(workers) is not int or not 1 <= workers <= len(names)):
        raise ValueError('Invalid annotation profiles/workers')
    return dict(profiles=names[:workers], shared_root=config['shared_root'],
                codex_bin=config['codex_bin'], proxy_file=config.get('proxy_file'),
                source=str(path.resolve()))


@contextmanager
def annotation_lease(config, timeout, offset=0):
    from ..codex_backend.profiles import account_lock, validate_credential
    started = time.monotonic()
    names = config['profiles'][offset:] + config['profiles'][:offset]
    while True:
        unavailable = {}
        for name in names:
            lock = account_lock(name, config['shared_root'])
            try:
                validate_credential(name, config['shared_root'])
                fd = lock.__enter__()
            except (ValueError, OSError):
                unavailable[name] = 'busy_or_login_unavailable'
                continue
            try:
                yield name, fd
            finally:
                lock.__exit__(None, None, None)
            return
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('No annotation profile available within account wait budget')
        print(json.dumps(dict(event='prompt_evolution_waiting_for_annotation_accounts',
                              reasons=unavailable)), flush=True)
        time.sleep(min(10, remaining))
