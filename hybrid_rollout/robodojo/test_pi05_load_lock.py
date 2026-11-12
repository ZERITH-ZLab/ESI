"""Real process/lock tests, without model, CUDA, SSH or mounted writes."""
import os
from pathlib import Path
import subprocess
import time

import pytest


HELPER = Path(__file__).with_name('pi05_load_lock.sh')
SCRIPT = r'''
set -euo pipefail
source "$HELPER"
child=''
cleanup() {
    rc=$?; trap - EXIT INT TERM; set +e
    if [[ -n "$child" ]]; then kill "$child"; wait "$child"; fi
    pi05_load_release
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
pi05_load_acquire
touch "$MARKERS/acquired"
# Inherit the lock fd just like the long-running policy process.
sleep 60 & child=$!
while [[ ! -e "$MARKERS/ready" ]]; do sleep .05; done
pi05_load_release
touch "$MARKERS/released"
wait "$child"
'''


def wait_for(path, process):
    deadline = time.monotonic() + 8
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, str(path)
        time.sleep(.05)


def launch(tmp_path, name, timeout='15'):
    markers = tmp_path/name
    markers.mkdir()
    p = subprocess.Popen(['bash', '-c', SCRIPT], env=dict(os.environ,
        HELPER=str(HELPER), MARKERS=str(markers),
        ROLLOUT_PI05_LOAD_LOCK=str(tmp_path/'load.lock'),
        ROLLOUT_PI05_LOAD_WAIT_TIMEOUT=timeout),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p, markers


def stop(p):
    if p.poll() is None:
        p.terminate()
    p.communicate(timeout=5)


@pytest.mark.parametrize('first_finishes_loading', [True, False])
def test_loads_serialize_but_inference_does_not_and_failure_releases(tmp_path, first_finishes_loading):
    first, a = launch(tmp_path, 'first')
    second = None
    try:
        wait_for(a/'acquired', first)
        second, b = launch(tmp_path, 'second')
        time.sleep(.3)
        assert not (b/'acquired').exists()
        if first_finishes_loading:
            (a/'ready').touch()
            wait_for(a/'released', first)
        else:
            stop(first)  # Interrupted load kills its child before releasing.
        wait_for(b/'acquired', second)
        if first_finishes_loading:
            assert first.poll() is None  # First policy still serving.
        (b/'ready').touch()
        wait_for(b/'released', second)
        assert (tmp_path/'load.lock').exists()
    finally:
        stop(first)
        if second is not None:
            stop(second)


def test_wait_timeout_never_starts_model_or_unlocks_another_owner(tmp_path):
    first, a = launch(tmp_path, 'first')
    second = None
    try:
        wait_for(a/'acquired', first)
        second, b = launch(tmp_path, 'second', timeout='1')
        _, err = second.communicate(timeout=5)
        assert second.returncode == 3
        assert 'load slot wait timeout' in err
        assert not (b/'acquired').exists()
        probe = subprocess.run(['flock', '-n', str(tmp_path/'load.lock'), 'true'])
        assert probe.returncode == 1  # Waiting owner's cleanup cannot unlock first.
    finally:
        stop(first)
        if second is not None:
            stop(second)


def test_cancel_waiter_is_prompt_and_does_not_start_model(tmp_path):
    first, a = launch(tmp_path, 'first')
    second = None
    try:
        wait_for(a/'acquired', first)
        second, b = launch(tmp_path, 'second')
        time.sleep(.3)
        stop(second)
        assert second.returncode == 143
        assert not (b/'acquired').exists()
        assert first.poll() is None
    finally:
        stop(first)
        if second is not None:
            stop(second)
