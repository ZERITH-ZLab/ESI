import json
import os
import subprocess
import sys

import pytest

from .model_config import model_selection
from .dual_campaign import configure
from .test_dual_deployment import configuration


def test_default_and_explicit_selection():
    assert model_selection({}) == ('gpt-6-astra', 'xhigh')
    assert model_selection({'ROLLOUT_MODEL': 'gpt-6-astra', 'ROLLOUT_REASONING_EFFORT': 'medium'}) == ('gpt-6-astra', 'medium')
    assert model_selection({'ROLLOUT_MODEL': 'gpt-6-sol', 'ROLLOUT_REASONING_EFFORT': 'high'}) == ('gpt-6-sol', 'high')


@pytest.mark.parametrize('env', [{'ROLLOUT_MODEL': ''}, {'ROLLOUT_MODEL': 'gpt 6'},
                               {'ROLLOUT_REASONING_EFFORT': 'xhight'},
                               {'ROLLOUT_REASONING_EFFORT': ''}])
def test_invalid_selection_fails(env):
    with pytest.raises(ValueError): model_selection(env)


def test_frozen_identity_enforced():
    env = {'ROLLOUT_REASONING_EFFORT': 'medium', 'ROLLOUT_RESUME_IDENTITY': json.dumps(dict(model='gpt-6-astra', effort='xhigh'))}
    with pytest.raises(ValueError, match='frozen'): model_selection(env)


@pytest.mark.parametrize('mode', ['resume', 'rerun'])
def test_experiment_rejects_changed_effort(tmp_path, mode):
    env, _ = configuration(tmp_path)
    previous = tmp_path/'results/test/_replicas/replica_8/attempt_0'
    previous.mkdir(parents=True)
    (previous/'batch.json').write_text(json.dumps(dict(resume_identity=dict(model='gpt-6-astra', effort='xhigh'))))
    env.update(ROLLOUT_RUN_MODE=mode, ROLLOUT_REASONING_EFFORT='medium')
    with pytest.raises(ValueError, match='new EXPERIMENT_ID'): configure(env)


def test_controller_and_archive_import_same_selection():
    env = dict(os.environ, ROLLOUT_MODEL='gpt-6-astra', ROLLOUT_REASONING_EFFORT='medium')
    env.pop('ROLLOUT_RESUME_IDENTITY', None)
    code = '''from hybrid_rollout.robodojo import settings, dual_campaign
from hybrid_rollout.robodojo.skill_auto_infer import run
from hybrid_rollout.robodojo.robodojo_server import client
assert all(m.MODEL == 'gpt-6-astra' and m.EFFORT == 'medium' for m in (settings, dual_campaign, run, client))
from pathlib import Path
assert run.agent_config(Path('/audit'), Path('/agent'))['model_reasoning_effort'] == 'medium'
'''
    subprocess.run([sys.executable, '-c', code], env=env, check=True)


def test_rerouted_model_aborts_before_actions(tmp_path):
    from .skill_auto_infer import run
    from .skill_auto_infer.test_cycle import Rollout
    from .test_network_continue import Transport, event
    worker = object.__new__(run.CodexPolicy)
    worker.workspace = worker.agent_workspace = tmp_path
    worker.thread_id = 'thread'
    worker.method, worker.timeout, worker.image_max_edge = 'pi05_plus_gpt', 1, 480
    worker.transport = Transport([event('model/rerouted',
        fromModel=run.MODEL, toModel='unexpected-model', reason='test')])
    rollout = Rollout()
    with pytest.raises(RuntimeError, match='rerouted'): worker.run(rollout)
    assert rollout.calls == []
    assert (tmp_path/'model_identity_error.json').exists()
