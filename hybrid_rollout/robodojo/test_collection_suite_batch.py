"""Exercise the real batch entry with every pilot task/variant; no GPT or GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import collection_suite as suite
from .collection import sampling_job_index
from .dual_campaign import configure
from .io import write_json
from .test_batch import FAKE_EPISODE
from .test_collection_suite import fixture

VARIANTS = [(task, variant) for task in suite.TASKS
            for variant in (('standard', 'random') if task in suite.RANDOM_TASKS else ('standard',))]


def setup(tmp_path, task):
    source = fixture(tmp_path)
    panel = suite.freeze_task(source, task, [10, 7])
    manifest = tmp_path/'panel.json'
    write_json(manifest, panel)
    cfg = dict(name='test', workers=8, model='gpt-6-astra', reasoning_effort='xhigh',
               account_pool_file='/unused', checkpoint='/unused')
    env, cases = configure(suite.campaign_env(
        dict(ROLLOUT_SHARED_ROOT=str(tmp_path), ROBODOJO_SOURCE=str(source)),
        cfg, panel, manifest, tmp_path/'workers.json'))
    return panel, env, cases


def batch_process(env):
    # Test command replaces simulator/model services, but batch membership,
    # selection, environment mapping, receipts and cleanup all run for real.
    fake = FAKE_EPISODE.replace("'ROLLOUT_EVAL_SEED')",
        "'ROLLOUT_EVAL_SEED', 'ROLLOUT_BASE_TASK', 'ROLLOUT_VARIANT')")
    code = ('import os,sys; from hybrid_rollout.robodojo.batch import run_batch; '
            f'raise SystemExit(run_batch(os.environ, episode_command=[sys.executable,"-c",{fake!r}]))')
    return subprocess.run([sys.executable, '-c', code],
        env=dict(os.environ, **env, PYTHONPATH=str(Path(__file__).parents[2])),
        capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize('task,variant', VARIANTS)
def test_real_batch_accepts_all_pilot_variants(tmp_path, task, variant):
    panel, env, cases = setup(tmp_path, task)
    case = next(c for c in cases if c['variant'] == variant)
    env.update(ROLLOUT_CASE_ID=case['case_id'], ROLLOUT_CASE_IDS='', ROLLOUT_COUNT='1',
               ROLLOUT_ALLOW_SEQUENTIAL_BATCH='1', ROLLOUT_COLLECTION_REPEAT_ID='1',
               ROLLOUT_COLLECTION_JOB_INDEX=str(sampling_job_index(panel, case['case_id'], 1)),
               POLICY_PORT='0', SIM_PORT='0')
    result = batch_process(env)
    assert result.returncode == 0, result.stdout + result.stderr
    root = tmp_path/'results'/env['ROLLOUT_EXPERIMENT_ID']/'_replicas'/f'replica_{env["ROLLOUT_REPLICA_ID"]}'
    receipt = json.loads((root/'attempt_0/batch.json').read_text())
    row, = receipt['episodes']
    assert row['state'] == 'completed' and row['artifact_status'] == 'verified'
    received = json.loads((Path(row['archive'])/'received_env.json').read_text())
    assert received['ROBODOJO_TASK'] == case['runtime_task']
    assert received['ROLLOUT_BASE_TASK'] == task and received['ROLLOUT_VARIANT'] == variant
    assert received['ROBODOJO_TASK'] == task + ('_random' if variant == 'random' else '')
    assert row['case_id'] == case['case_id']
    assert row['collection_job_index'] == (10 if variant == 'random' else 0)


@pytest.mark.parametrize('wrong_task', ['arrange_largest_number_random', 'build_tower', 'unknown_task'])
def test_batch_rejects_task_not_in_selected_manifest(tmp_path, wrong_task):
    panel, env, _ = setup(tmp_path, 'arrange_largest_number')
    env['ROBODOJO_TASK'] = wrong_task
    result = batch_process(env)
    assert result.returncode != 0 and 'Base task is not in the frozen panel' in result.stderr
    assert not (tmp_path/'results').exists()


def test_manifest_rejects_wrong_runtime_variant_mapping(tmp_path):
    from .evaluation import canonical_sha256, read_panel
    panel, env, _ = setup(tmp_path, 'sort_nesting_dolls_by_size')
    panel['cases'][10]['runtime_task'] = 'sort_nesting_dolls_by_size'
    panel['panel_sha256'] = canonical_sha256({k:v for k,v in panel.items() if k != 'panel_sha256'})
    write_json(env['ROLLOUT_EVAL_MANIFEST'], panel)
    with pytest.raises(ValueError, match='case identity/order'):
        read_panel(env['ROLLOUT_EVAL_MANIFEST'])
