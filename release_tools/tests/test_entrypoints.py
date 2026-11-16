import json
from copy import deepcopy
from pathlib import Path

import pytest

from release_tools.config import ROOT, load, read, resolve, recursive_round_weights
from release_tools.__main__ import plan, freeze, main
from hybrid_rollout.robodojo.parallel_collection import explicit_slots


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    import os
    for key in list(os.environ):
        if key.startswith(('TRAIN_', 'JAX_', 'ROLLOUT_')):
            monkeypatch.delenv(key)


def recipe(name, *settings, deployment='deployment.example.json'):
    return load(ROOT/'configs'/name, ROOT/'configs'/deployment,
                ['run.name=unit_test', *settings])


def test_resolver_types_and_cycles(tmp_path):
    (tmp_path/'data.json').write_text('{"x": 2}')
    (tmp_path/'parent.json').write_text('{"v":{"$file":"data.json"},"a":1}')
    (tmp_path/'child.json').write_text('{"extends":"parent.json","a":3,"b":"${v}"}')
    assert resolve(read(tmp_path/'child.json')) == dict(v=dict(x=2),a=3,b=dict(x=2))
    with pytest.raises(ValueError,match='cycle'):resolve(dict(a='${b}',b='${a}'))
    with pytest.raises(ValueError,match='required'):resolve(dict(a=None,b='${a}'))
    with pytest.raises(ValueError,match='Unknown'):recipe('eval/final.json','run.typo=bad')
    (tmp_path/'data.json').write_text('{"extends":"child.json"}')
    with pytest.raises(ValueError,match='cycle'):read(tmp_path/'child.json')


def test_round_weights():
    assert recursive_round_weights(['r1']) == {'r1':1.0}
    assert recursive_round_weights(['r1','r2']) == pytest.approx(dict(r1=.4,r2=.6))
    assert recursive_round_weights(['r1','r2','r3']) == pytest.approx(dict(r1=.16,r2=.24,r3=.6))
    with pytest.raises(ValueError):recursive_round_weights(['r1','r1'])


@pytest.mark.parametrize('name,route',[('bootstrap','auto_infer_v1'),('feedforward','subtask_feedforward_v1')])
def test_collection_preserves_recipe(tmp_path,name,route):
    cfg=recipe(f'collection/{name}.json','run.checkpoint=/models/ckpt')
    cmd,env,files=plan(cfg,tmp_path,inspect=True)
    native=files['stage_config.json']
    assert native['controller_route']==route
    assert native['pi_force_hold']['config']['checkpoint_sha256']=='<computed on start>'
    assert not native['prompt_evolution']['enabled']
    assert native['prompt_evolution']['workflow']=='conversation_v2'
    assert native['budget_multiplier']==[10,7]
    assert native['max_seconds']==10800
    assert env['ROLLOUT_EVAL_EXCLUSIONS_FILE']==''
    assert len(json.loads(env['ROLLOUT_WORKER_SLOTS_JSON']))==8
    assert not tmp_path.joinpath('stage_config.json').exists()


@pytest.mark.parametrize('route', ['bootstrap', 'feedforward'])
@pytest.mark.parametrize('enabled,tighten', [(True, True), (True, False), (False, True), (False, False)])
def test_robodojo_intervention_switches(tmp_path, route, enabled, tighten):
    from hybrid_rollout.robodojo.robodojo_server.pi_force_hold import configured, PiForceHold
    cfg = recipe(f'collection/{route}.json', 'run.checkpoint=/models/ckpt',
        f'robodojo.pi_force_hold.enabled={json.dumps(enabled)}',
        f'robodojo.pi_force_hold.config.tighten_enabled={json.dumps(tighten)}')
    _, env, files = plan(cfg, tmp_path, inspect=True)
    hold = files['stage_config.json']['pi_force_hold']
    assert hold['enabled'] is enabled
    if enabled and tighten:
        assert hold['config']['checkpoint_sha256'] == '<computed on start>'
        hold['config']['checkpoint_sha256'] = 'a'*64
    elif not enabled:
        assert hold['config'] == {}
    config = configured(hold['config'])
    assert (PiForceHold(config).zero_latch is not None) == (enabled and tighten)
    assert json.loads(env['ROBODOJO_GRIPPER_HOLD_CONFIG']) == dict(
        min_steps=3, stable_transitions=2, max_steps=8, delta_m=.0001)


def test_robodojo_eval_defaults_and_overrides(tmp_path):
    final = recipe('eval/final.json', 'run.checkpoint=/models/ckpt')
    _, _, files = plan(final, tmp_path, inspect=True)
    model = files['stage_config.json']['models'][0]
    assert not final['robodojo']['pi_force_hold']['enabled']
    from hybrid_rollout.robodojo.downsampling import configuration
    from hybrid_rollout.robodojo.flywheel.action_downsampling import algorithm_version
    config = configuration(dict(ROLLOUT_ACTION_DOWNSAMPLING=model['downsampling']['mode'],
        ROLLOUT_DOWNSAMPLING_CONFIG=json.dumps(model['downsampling']['config'])))
    assert algorithm_version(config) == 'joint_fk_rdp_significant_closing_v5'
    off = recipe('eval/final.json', 'run.checkpoint=/models/ckpt', 'robodojo.eval_downsampling.mode=off')
    assert plan(off, tmp_path, inspect=True)[2]['stage_config.json']['models'][0]['downsampling']['mode'] == 'off'
    baseline = recipe('eval/baseline.json')
    assert plan(baseline, tmp_path, inspect=True)[2]['stage_config.json']['models'][0]['downsampling']['mode'] == 'off'
    final['robodojo']['pi_force_hold']['enabled'] = True
    with pytest.raises(ValueError, match='eval disables'):
        plan(final, tmp_path, inspect=True)


@pytest.mark.parametrize('name,policy',[('final','task_subtask'),('baseline','task_only')])
def test_eval_without_historical_locks(tmp_path,name,policy,monkeypatch):
    monkeypatch.setenv('ROLLOUT_ENABLE_SCENE_QUERY','1')
    monkeypatch.setenv('ROLLOUT_EVAL_EXCLUSIONS_FILE','/historical/list.json')
    cfg=recipe(f'eval/{name}.json','run.checkpoint=/models/ckpt')
    _,env,files=plan(cfg,tmp_path,inspect=True)
    native=files['stage_config.json']
    assert sum(len(m['cases']) for m in native['models'])==200
    assert all(m['policy_input']==policy for m in native['models'])
    assert not {'expected_runtime','execution_source'} & native.keys()
    assert 'ROLLOUT_ENABLE_SCENE_QUERY' not in env
    assert env['ROLLOUT_EVAL_EXCLUSIONS_FILE']==''
    from hybrid_rollout.robodojo.regression_eval import required_env
    rules=required_env('subtask_feedforward_v1' if name=='final' else 'auto_infer_v1')
    assert rules['ROLLOUT_STRICT_EVAL']=='1'
    assert rules['ROLLOUT_ENABLE_SCENE_QUERY']=='0'
    assert rules['ROLLOUT_ENABLE_TASK_MEMORY']=='0'


@pytest.mark.parametrize('deployment,devices,micro',[('deployment.example.json',8,64),('deployments/train16.example.json',16,128)])
def test_training_config_is_valid(tmp_path,deployment,devices,micro):
    cfg=recipe('training/round3.json','run.checkpoint=/models/ckpt','run.index=/data/index','run.norm=/data/norm',deployment=deployment)
    cmd,env,files=plan(cfg,tmp_path,inspect=True)
    native=files['stage_config.json']
    assert native['conditioning']['mode']=='subtask'
    assert native['parallelism']['num_devices']==devices
    assert native['training']['micro_batch_size']==micro
    assert native['training']['batch_size']==256
    assert native['sampling']['round_weights']==pytest.approx(dict(round1=.16,round2=.24,round3=.6))
    path=tmp_path/'training.json';path.write_text(json.dumps(native))
    from data_processing.training.config import load_config
    assert load_config(path,environ={})['training']['batch_size']==256


def test_training_overrides_are_frozen(tmp_path,monkeypatch):
    monkeypatch.setenv('TRAIN_MODE','task')
    monkeypatch.setenv('TRAIN_LEARNING_RATE','0.00001')
    cfg=recipe('training/round1.json','run.index=/data/index','run.norm=/data/norm')
    _,_,files=plan(cfg,tmp_path,inspect=True)
    assert files['stage_config.json']['conditioning']['mode']=='task'
    assert files['stage_config.json']['optimizer']['learning_rate']==1e-5


def test_freeze_rejects_config_mutation(tmp_path):
    freeze(tmp_path,dict(a=1),{'stage_config.json':dict(b=2)})
    freeze(tmp_path,dict(a=1),{'stage_config.json':dict(b=2)})
    with pytest.raises(ValueError,match='changed'):
        freeze(tmp_path,dict(a=1),{'stage_config.json':dict(b=3)})
    assert json.loads((tmp_path/'stage_config.json').read_text())==dict(b=2)


def test_custom_slot_count_and_capacity():
    cfg=recipe('collection/bootstrap.json')
    row=cfg['deployment']['collection']['slots'][0]
    rows=[dict(row,POLICY_PORT=18000+i,SIM_PORT=19000+i,ROLLOUT_SIM_REMOTE_PORT=20000+i) for i in range(12)]
    env=dict(ROLLOUT_WORKER_SLOTS_JSON=json.dumps(rows),ROLLOUT_ACCOUNT_POOL_FILE='/pool.json',ROLLOUT_PARALLEL_WORKERS='3')
    assert len(explicit_slots(env))==3
    assert len(explicit_slots(dict(env,ROLLOUT_WORKER_CONTROL_FILE='/control.json')))==12
    with pytest.raises(ValueError,match='more workers'):
        explicit_slots(dict(env,ROLLOUT_PARALLEL_WORKERS='13'))
    rows[1]['SIM_PORT']=rows[0]['POLICY_PORT']
    with pytest.raises(ValueError,match='duplicate'):
        explicit_slots(dict(env,ROLLOUT_WORKER_SLOTS_JSON=json.dumps(rows)))


@pytest.mark.parametrize('route',['offline','online'])
def test_annotation_accounts_and_command(tmp_path,route):
    cfg=recipe(f'annotation/{route}.json','run.inputs=["/collection/a","/collection/b"]')
    cmd,_,files=plan(cfg,tmp_path,inspect=True)
    assert cmd[-4:-2]==['/collection/a','/collection/b']
    assert files['stage_config.json']['model']=='gpt-6.1-sol'
    assert files['stage_config.json']['profiles']==['codex_a']


def test_required_values_and_preview(capsys,tmp_path):
    with pytest.raises(ValueError,match='checkpoint'):plan(recipe('eval/final.json','run.checkpoint=null'),tmp_path,inspect=True)
    with pytest.raises(ValueError,match='output'):plan(recipe('data.json'),tmp_path,inspect=True)
    main(['--config',str(ROOT/'configs/eval/baseline.json'),'--deployment',str(ROOT/'configs/deployment.example.json'),
          '--set','run.name=preview','--set',f'deployment.paths.output_root={tmp_path}', '--show-config'])
    assert json.loads(capsys.readouterr().out)['generated_configs']['stage_config.json']['workers']==8
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('route', ['collection/bootstrap.json', 'collection/feedforward.json',
                                  'eval/final.json', 'eval/baseline.json'])
def test_output_directory_portability(route, tmp_path, capsys, monkeypatch):
    output = tmp_path/'new outputs'
    args = ['--config', str(ROOT/'configs'/route), '--deployment',
            str(ROOT/'configs/deployment.example.json'), '--set', 'run.name=portable',
            '--set', 'run.checkpoint=/models/policy',
            '--set', f'deployment.paths.output_root={output}']
    main([*args, '--show-config'])
    assert json.loads(capsys.readouterr().out)['config']['deployment']['paths']['output_root'] == str(output)
    assert not output.exists()
    # Exercise configuration freezing and dispatch without launching services.
    import release_tools.__main__ as entry
    import hybrid_rollout.robodojo.pi05_server.checkpoint as checkpoint
    monkeypatch.setattr(checkpoint, 'checkpoint_identity', lambda path: {'checkpoint_sha256': 'test-identity'})
    calls = []
    monkeypatch.setattr(entry.subprocess, 'call', lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    assert main(args) == 0
    assert calls[0][1]['env']['ROLLOUT_SHARED_ROOT'] == str(output)
    assert (output/'release_runs/portable').is_dir()
    with pytest.raises(ValueError, match='absolute path'):
        main([*args, '--set', 'deployment.paths.output_root=relative/outputs', '--show-config'])
    with pytest.raises(ValueError, match='outside the source'):
        main([*args, '--set', f'deployment.paths.output_root={ROOT}/outputs', '--show-config'])


def test_single_node_does_not_inherit_scheduler_world_size(tmp_path,monkeypatch):
    monkeypatch.setenv('WORLD_SIZE','16')
    cfg=recipe('training/round1.json','run.index=/data/index','run.norm=/data/norm')
    _,env,_=plan(cfg,tmp_path,inspect=True)
    assert env['JAX_PROCESS_COUNT']=='1'
    monkeypatch.setenv('JAX_PROCESS_COUNT','2')
    with pytest.raises(ValueError,match='num_nodes'):plan(cfg,tmp_path,inspect=True)


def test_cli_mode_is_reflected_in_saved_config(tmp_path,monkeypatch):
    monkeypatch.setenv('TRAIN_MODE','subtask')
    cfg=recipe('training/round1.json','run.index=/data/index','run.norm=/data/norm')
    _,_,files=plan(cfg,tmp_path,inspect=True,training_mode='task')
    assert files['stage_config.json']['conditioning']['mode']=='task'


@pytest.mark.parametrize('name',['offline','online'])
def test_generated_annotation_config_native_parser(tmp_path,name):
    cfg=recipe(f'annotation/{name}.json','run.action=run')
    _,_,files=plan(cfg,tmp_path,inspect=True)
    path=tmp_path/'config.json';path.write_text(json.dumps(files['stage_config.json']))
    from data_processing.subtasks.pipeline import load_config
    assert load_config(path)['workers']==1


@pytest.mark.parametrize('stage,name',[('collection','bootstrap'),('collection','feedforward'),('eval','baseline'),('eval','final')])
def test_generated_rollout_config_native_parser(tmp_path,stage,name,monkeypatch):
    checkpoint=tmp_path/'checkpoint'
    (checkpoint/'params').mkdir(parents=True)
    (checkpoint/'assets/arx_x5_sim').mkdir(parents=True)
    (checkpoint/'assets/arx_x5_sim/norm_stats.json').write_text('{}')
    from hybrid_rollout.robodojo.pi05_server import checkpoint as module
    monkeypatch.setattr(module,'checkpoint_identity',lambda p:dict(checkpoint_sha256='a'*64))
    cfg=recipe(f'{stage}/{name}.json',f'run.checkpoint={checkpoint}')
    _,_,files=plan(cfg,tmp_path)
    path=tmp_path/'config.json';path.write_text(json.dumps(files['stage_config.json']))
    if stage=='collection':from hybrid_rollout.robodojo.collection_suite import read_config
    else:from hybrid_rollout.robodojo.eval_sequence import read_config
    assert read_config(path)['workers']==8


def test_collection_custom_teacher_and_limits_reach_runtime(tmp_path):
    from hybrid_rollout.robodojo.test_collection_suite import fixture
    from hybrid_rollout.robodojo.collection_suite import freeze_task,campaign_env
    from hybrid_rollout.robodojo.collection import runtime_settings
    cfg=recipe('collection/bootstrap.json','llm.rollout.model=custom-model','llm.rollout.effort=medium',
               'llm.rollout.auto_compact_token_limit=120000','limits.episode_seconds=7200')
    _,env,files=plan(cfg,tmp_path,inspect=True)
    native=files['stage_config.json']
    native['pi_force_hold']['config']['checkpoint_sha256']='a'*64
    panel=freeze_task(fixture(tmp_path),'fill_pen_holder',[10,7],max_seconds=native['max_seconds'])
    e=campaign_env(env,native,panel,tmp_path/'panel.json',tmp_path/'workers.json')
    assert e['ROLLOUT_MODEL']=='custom-model'
    assert e['ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT']=='120000'
    assert runtime_settings(e,panel)['collection_protocol']['max_seconds']==7200


def test_final_eval_has_distinct_default_checkpoint(tmp_path):
    final = recipe('eval/final.json', deployment='deployment.eval.example.json')
    base = recipe('eval/baseline.json', deployment='deployment.eval.example.json')
    assert final['run']['checkpoint'] == '/mnt/models/esi-final'
    assert base['run']['checkpoint'] != final['run']['checkpoint']
    _, env, _ = plan(final, tmp_path, inspect=True)
    assert env['ROLLOUT_PARALLEL_WORKERS'] == '1'
    assert env['CHECKPOINT'] == '/mnt/models/esi-final'


def test_ssh_user_global_and_slot_override(tmp_path):
    cfg = recipe('eval/final.json', 'deployment.paths.sim_ssh_user=robot')
    cfg['deployment']['collection']['slots'][0]['ROLLOUT_SIM_SSH_USER'] = 'worker'
    _, env, _ = plan(cfg, tmp_path, inspect=True)
    slots = explicit_slots(env)
    assert slots[0]['ROLLOUT_SIM_SSH_USER'] == 'worker'
    assert slots[1]['ROLLOUT_SIM_SSH_USER'] == 'robot'
    from hybrid_rollout.robodojo.parallel_collection import SLOT_KEYS
    assert 'ROLLOUT_SIM_SSH_USER' in SLOT_KEYS


@pytest.mark.parametrize('user', ['', '-option', 'a@host', 'a b', 'a;id'])
def test_invalid_ssh_user_is_rejected(tmp_path, user):
    cfg = recipe('eval/final.json')
    cfg['deployment']['paths']['sim_ssh_user'] = user
    with pytest.raises(ValueError, match='SSH username'):
        plan(cfg, tmp_path, inspect=True)
