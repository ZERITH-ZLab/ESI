import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from . import eval_sequence as sequence
from . import regression_eval as evaluation
from .deployment import digest
from .dual_campaign import acquire_resources, configure
from .evaluation import archive_path, verify_assets
from .io import write_json
from .test_regression_eval import setup


def test_unified_pairing_archives_and_protocol(tmp_path):
    source, _, env = setup(tmp_path)
    panel = evaluation.freeze_queue(source)
    write_json(env['ROLLOUT_EVAL_MANIFEST'], panel)
    env.pop('ROLLOUT_POLICY_RNG_SEED')
    env['ROLLOUT_COUNT'] = '15'
    _, cases = configure(env)
    verify_assets(panel, source)
    assert len(cases) == len({c['case_id'] for c in cases}) == 15
    paths = [archive_path(tmp_path, 'test', c, 3, 0) for c in cases]
    assert len(set(paths)) == 15
    for seed in range(3):
        selected = cases[seed*5:seed*5+5]
        assert [(c['variant'], c['layout_id']) for c in selected] == list(evaluation.CASES)
        assert all(c['policy_rng_seed'] == seed and c['evaluation_repeat'] == seed+1 for c in selected)
    with pytest.raises(ValueError, match='per episode'):
        configure(dict(env, ROLLOUT_POLICY_RNG_SEED='0'))


def test_batch_passes_case_seed_to_episode_process(tmp_path):
    from .test_batch import FAKE_EPISODE
    source, _, env = setup(tmp_path)
    panel = evaluation.freeze_queue(source)
    write_json(env['ROLLOUT_EVAL_MANIFEST'], panel)
    env.pop('ROLLOUT_POLICY_RNG_SEED')
    selected = [panel['cases'][5], panel['cases'][10]]
    env.update(ROLLOUT_COUNT='2', ROLLOUT_CASE_IDS=','.join(c['case_id'] for c in selected),
        ROLLOUT_ALLOW_SEQUENTIAL_BATCH='1', ROBODOJO_SOURCE=str(source))
    env, _ = configure(env)
    env = dict(os.environ, **env)
    # Fake episodes create no servers. Bind ephemeral ports in the cleanup
    # probe so this test cannot collide with a real evaluation on this host.
    env.update(POLICY_PORT='0', SIM_PORT='0')
    env.pop('ROLLOUT_POLICY_RNG_SEED', None)
    env['PYTHONPATH'] = str(Path(__file__).parents[2])
    fake = FAKE_EPISODE.replace("'ROLLOUT_EVAL_SEED')", "'ROLLOUT_EVAL_SEED', 'ROLLOUT_POLICY_RNG_SEED')")
    code = ('import os,sys; from hybrid_rollout.robodojo.batch import run_batch; '
            f'raise SystemExit(run_batch(os.environ, episode_command=[sys.executable,"-c",{fake!r}]))')
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    batch = json.loads((tmp_path/'results/test/_replicas/replica_3/attempt_0/batch.json').read_text())
    assert len(batch['episodes']) == 2
    for row, case in zip(batch['episodes'], selected):
        archive = Path(row['archive'])
        received = json.loads((archive/'received_env.json').read_text())
        assert received['ROLLOUT_POLICY_RNG_SEED'] == str(case['policy_rng_seed'])
        assert f'repeat_{case["evaluation_repeat"]:03d}' in archive.parts


def config():
    return dict(schema='robodojo.eval_sequence.v1', name='sequence', workers=4,
        account_pool_file='/unused', models=[dict(experiment=name, checkpoint='/ckpt/'+name,
        policy_input='task_only') for name in ('plain', 'subtask')])


def complete(shared, name, count=15):
    rows = [dict(case_id=c['case_id'], archive='/'+c['case_id'], state='completed',
        artifact_status='verified', reset=dict(all_owned_processes_exited=True, ports_released=True),
        result=dict(complete=True, step_id=1050),
        evaluation_outcome=dict(complete=True, valid_for_success_rate=True, native_success=False))
        for c in list(evaluation.queue_specs())[:count]]
    path = Path(shared)/f'results/{name}/_replicas/replica_3/attempt_0/batch.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(episodes=rows))


@pytest.mark.parametrize('returncode,count,expected', [(0,15,2), (0,14,1), (1,15,1)])
def test_sequence_waits_for_complete_model(tmp_path, monkeypatch, returncode, count, expected):
    source, _, env = setup(tmp_path)
    env['ROBODOJO_SOURCE'] = str(source)
    # Freeze before mocking subprocess.run (git fixture also uses subprocess).
    panel = evaluation.freeze_queue(source)
    monkeypatch.setattr(evaluation, 'freeze_queue', lambda source: panel)
    launched = []
    def launch(argv, env):
        name = env['ROLLOUT_EXPERIMENT_ID']
        assert '--wait-for-resources' in argv
        assert env['ROLLOUT_COUNT'] == '15' and 'ROLLOUT_POLICY_RNG_SEED' not in env
        assert env['CHECKPOINT'] == '/ckpt/'+name
        if name == 'subtask':
            assert evaluation.report(tmp_path, 'plain', unified=True)['completed'] == 15
        launched.append(name)
        complete(tmp_path, name, count)
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(sequence.subprocess, 'run', launch)
    result = sequence.run_sequence(config(), env)
    assert len(launched) == expected
    assert result == (0 if expected == 2 else 1)
    if expected == 2:
        launched.clear()
        assert sequence.run_sequence(config(), env) == 0
        assert not launched  # Both completed campaigns are retained, including native failures.


def test_resume_and_dry_run(tmp_path, monkeypatch):
    source, _, env = setup(tmp_path)
    env['ROBODOJO_SOURCE'] = str(source)
    panel = evaluation.freeze_queue(source)
    monkeypatch.setattr(evaluation, 'freeze_queue', lambda source: panel)
    complete(tmp_path, 'plain', 3)
    calls = []
    monkeypatch.setattr(sequence.subprocess, 'run', lambda argv, env:
        calls.append((argv, env)) or SimpleNamespace(returncode=0))
    assert sequence.run_sequence(config(), env, dry_run=True) == 0
    assert [env['ROLLOUT_RUN_MODE'] for _, env in calls] == ['resume', 'fresh']
    assert all('--dry-run' in argv for argv, _ in calls)
    assert not (tmp_path/'evaluation').exists() and not (tmp_path/'eval_panels').exists()


def test_wait_releases_partial_locks(tmp_path, monkeypatch):
    from . import dual_campaign as campaign
    keys = sorted([['resource', i] for i in range(3)], key=digest)
    held = acquire_resources(tmp_path, [keys[1]])
    slept = []
    def sleep(seconds):
        # Earlier lock must be released before sleeping, avoiding hold-and-wait deadlocks.
        probe = acquire_resources(tmp_path, [keys[0]])
        for fd in probe + held:
            os.close(fd)
        slept.append(seconds)
    monkeypatch.setattr(campaign.time, 'sleep', sleep)
    with pytest.raises(SystemExit, match='Resource already in use'):
        acquire_resources(tmp_path, keys)
    acquired = acquire_resources(tmp_path, keys, wait=True)
    assert slept == [5] and len(acquired) == 3
    for fd in acquired:
        os.close(fd)


def test_duplicate_sequence_cannot_start(tmp_path):
    path = tmp_path/'evaluation/sequence'
    path.mkdir(parents=True)
    with (path/'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='already running'):
            sequence.run_sequence(config(), {'ROLLOUT_SHARED_ROOT': str(tmp_path)})
