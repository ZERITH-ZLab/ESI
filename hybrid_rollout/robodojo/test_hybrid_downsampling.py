import json
from types import SimpleNamespace

import numpy as np
import pytest

from .downsampling import HybridDownsampling, configuration
from .flywheel.action_downsampling import SelectionConfig
from .test_contract import Sim, Student, response, invoke
from .robodojo_server.client import RoboDojoTools


class DownsampleSim(Sim):
    def request(self, op, **args):
        r = super().request(op, **args)
        if op == 'fk_preview':
            r['physical_steps'] = 0
        if op == 'teacher_observation':
            r['states'][[6, 13]] = .3
        return r


def test_real_client_compresses_only_approved_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv('ROLLOUT_ACTION_DOWNSAMPLING', 'error')
    monkeypatch.setenv('ROLLOUT_DOWNSAMPLING_CONFIG', '{"gripper_epsilon":0.005}')
    r = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=DownsampleSim)
    invoke(r); invoke(r)
    decision = response(r.request); decision['steps'] = 15
    invoke(r, response=decision)
    sent = next(args['actions'] for op,args in r.sim.requests if op == 'chunk_step')
    selection = json.loads((tmp_path/'selection_000.json').read_text())
    assert len(sent) < 15
    assert selection['retained_indices'][0] == 0 and selection['retained_indices'][-1] == 14
    assert r.tick == len(sent) == r.history[0]['executed_steps']
    data = np.load(tmp_path/'execution_000.npz')
    np.testing.assert_array_equal(data['proposal_action_index'], selection['actual_ack_original_indices'])
    np.testing.assert_array_equal(sent, np.load(tmp_path/'proposal_000.npz')['actions'][data['proposal_action_index']])
    assert r.history[0]['action_downsampling']['approved_original_steps'] == 15


def test_fk_failure_is_archived_no_execution(tmp_path, monkeypatch):
    monkeypatch.setenv('ROLLOUT_ACTION_DOWNSAMPLING', 'error')
    r = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=Sim)
    invoke(r); invoke(r)
    with pytest.raises(RuntimeError, match='FK preview'):
        invoke(r, response=response(r.request))
    assert 'chunk_step' not in r.sim.calls and r.tick == 0
    assert json.loads((tmp_path/'selection_000.json').read_text())['status'] == 'failed'


def test_gpt_correction_is_not_thinned_and_updates_guard(tmp_path, monkeypatch):
    monkeypatch.setenv('ROLLOUT_ACTION_DOWNSAMPLING', 'error')
    r = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=DownsampleSim)
    invoke(r); invoke(r)
    invoke(r, response=response(r.request, 'eef'))
    assert 'eef_short_motion' in r.sim.calls and 'chunk_step' not in r.sim.calls
    assert not list(tmp_path.glob('selection_*'))
    assert r.downsampling.selector.hold_remaining.tolist() == [14, 14]
    assert r.downsampling.last_tick == r.tick


def test_config_rejects_force_hold():
    with pytest.raises(ValueError, match='disabled'):
        configuration(dict(ROLLOUT_ACTION_DOWNSAMPLING='error', ROBODOJO_PI_FORCE_HOLD_ENABLED='1'))


@pytest.mark.parametrize('version',['v4','v5'])
def test_versioned_downsampling_presets(version):
    from pathlib import Path
    from .downsampling import identity
    preset = json.loads((Path(__file__).parent/f'downsampling.gripper_{version}.json').read_text())
    env = dict(ROLLOUT_ACTION_DOWNSAMPLING='error', ROBODOJO_PI_FORCE_HOLD_ENABLED='0',
               ROLLOUT_DOWNSAMPLING_CONFIG=json.dumps(preset))
    expected={'v4':'joint_fk_rdp_closing_phase_v4','v5':'joint_fk_rdp_significant_closing_v5'}
    assert identity(env)['action_downsampling']['algorithm_version'] == expected[version]


def test_release_final_eval_uses_v5_without_private_source_pin(tmp_path):
    from pathlib import Path
    from .eval_sequence import read_config
    from .eval_queue import validate_queue
    root = Path(__file__).parent
    preset = json.loads((root/'downsampling.gripper_v5.json').read_text())
    recipe = json.loads((root/'eval_queue.round3_final200_updated.json').read_text())
    # Recipe validation must not require a developer's checkpoint installation.
    checkpoint = tmp_path/'checkpoint'
    (checkpoint/'params').mkdir(parents=True)
    assets = checkpoint/'assets/arx_x5_sim'
    assets.mkdir(parents=True)
    (assets/'norm_stats.json').write_text('{}')
    for model in recipe['models']:
        model['checkpoint'] = str(checkpoint)
    config_path = tmp_path/'eval.json'
    config_path.write_text(json.dumps(recipe))
    cfg = read_config(config_path)
    validate_queue(cfg)
    assert cfg['workers'] == 8 and 'execution_source' not in cfg
    assert sum(len(m['cases']) for m in cfg['models']) == 200
    for model in cfg['models']:
        assert model['downsampling']['config'] == preset
        assert model['policy_input'] == 'task_subtask'
        assert len(model['cases']) == len(set(map(tuple,model['cases'])))


@pytest.mark.parametrize('version',['v4','v5'])
def test_versioned_gpt_actions_are_not_thinned_but_commit_closure_phase(tmp_path, monkeypatch, version):
    from pathlib import Path
    preset = Path(__file__).parent/f'downsampling.gripper_{version}.json'
    monkeypatch.setenv('ROLLOUT_ACTION_DOWNSAMPLING','error')
    monkeypatch.setenv('ROLLOUT_DOWNSAMPLING_CONFIG',preset.read_text())
    r = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=DownsampleSim)
    invoke(r); invoke(r)
    invoke(r, response=response(r.request, 'eef'))
    assert 'eef_short_motion' in r.sim.calls and 'chunk_step' not in r.sim.calls
    assert not list(tmp_path.glob('selection_*'))
    assert r.downsampling.selector.closing_phases is not None
    assert r.downsampling.selector.hold_remaining.tolist() == [9,9]


def test_disabled_hold_identity_survives_json_worker_handoff():
    from .dual_campaign import gripper_runtime_identity
    identity = gripper_runtime_identity(dict(ROBODOJO_PI_FORCE_HOLD_ENABLED='0',
                                            ROBODOJO_PI_FORCE_HOLD_CONFIG='{}'))
    assert identity == json.loads(json.dumps(identity))


def test_partial_ack_does_not_commit_unexecuted_gripper(tmp_path):
    from .flywheel.test_action_downsampling import toy_fk, sample
    adapter = HybridDownsampling(SelectionConfig(), None)
    adapter.selector.fk = toy_fk
    actions = sample(); actions[-1, 6] = .1
    indices, _ = adapter.select(actions, 15, actions[0], 0, tmp_path/'selection.json')
    adapter.acknowledge([dict(valid=True, executed_action=actions[0])], actions[0], 1)
    assert adapter.selector.hold_remaining.tolist() == [0, 0]
    with pytest.raises(RuntimeError, match='stale'):
        adapter.select(actions, 15, actions[0], 2, tmp_path/'stale.json')


def test_task_panel_native_budget_selection_and_resume(tmp_path):
    from .test_collection_suite import fixture
    from .task_eval import freeze
    from . import regression_eval as evaluation
    from .robodojo_server.step_budget import apply
    from .evaluation import verify_assets
    from .eval_sequence import campaign_env
    from .io import write_json
    source = fixture(tmp_path)
    for task, selection in [('pack_objects_into_box', [['standard',0],['random',0]]),
                            ('imitate_sorting_sequence', [['standard',0],['standard',1]])]:
        p = freeze(source, task, selection)
        verify_assets(p, source)
        assert apply(SimpleNamespace(step_lim=1600), 0, p)['effective_step_limit'] == 1600
        with pytest.raises(ValueError, match='changed'):
            apply(SimpleNamespace(step_lim=1050), 0, p)
        path = tmp_path/'eval_panels'/task/'queue.json'; path.parent.mkdir(parents=True)
        write_json(path, p)
        model = dict(experiment=task, task=task, cases=selection, checkpoint='/test', policy_input='task_subtask',
                     downsampling=dict(mode='error', config={'gripper_epsilon':.005}))
        env = campaign_env(dict(ROLLOUT_SHARED_ROOT=str(tmp_path), ROBODOJO_PI_FORCE_HOLD_ENABLED='1'),
            dict(workers=2, account_pool_file='/pool'), model, path)
        assert env['ROBODOJO_PI_FORCE_HOLD_ENABLED'] == '0'
        assert env['ROLLOUT_COUNT'] == '2'
        assert evaluation.runtime_settings(env, p)['native_step_limit'] == 1600
        summary = evaluation.report(tmp_path, task, unified=True)
        assert summary['planned'] == 2 and summary['incomplete_or_unstarted'] == 2


def test_eval_prompt_preserves_task_removes_unavailable_tools():
    from .skill_auto_infer.eval_prompt import adapt
    from .skill_auto_infer.run import build_prompt, SKILL_ROOT
    from .task_success_conditions import append_task_success_conditions
    from pathlib import Path
    text = adapt(append_task_success_conditions(build_prompt(SKILL_ROOT, Path('/agent')), 'arrange_largest_number'))
    assert 'Flip recovery:' in text and 'robodojo_history' in text
    assert 'end the episode only if the existing early-abandon' not in text
    assert 'For targeted lookup use rg' not in text
