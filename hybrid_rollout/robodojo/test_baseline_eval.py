import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import prompt_profile as pp
from .regression_eval import required_env


def baseline_env():
    return dict(required_env(), ROLLOUT_PROMPT_PROFILE=pp.BASELINE,
        ROLLOUT_ACTION_DOWNSAMPLING='off', ROBODOJO_PI_FORCE_HOLD_ENABLED='0',
        ROLLOUT_TASK_REFERENCE_IMAGES='0')


@pytest.mark.parametrize('task',[
    'pack_objects_into_box','arrange_largest_number','sort_nesting_dolls_by_size',
    'classify_objects','fill_pen_holder','organize_table','imitate_sorting_sequence',
    'stack_blocks_by_language','classify_objects_by_language','sweep_blocks'])
def test_baseline_thread_and_workspace_contain_no_added_guidance(tmp_path,monkeypatch,task):
    from .skill_auto_infer import run
    for k,v in baseline_env().items():monkeypatch.setenv(k,v)
    captured={}
    class Transport:
        def __init__(self,*a):self.process=SimpleNamespace(pid=0)
        def request(self,method,params,timeout):
            if method=='initialize':return {}
            assert method=='thread/start'
            captured.update(params)
            return dict(model=run.MODEL,reasoningEffort=run.EFFORT,thread={'id':'offline'})
        def notify(self,*a):pass
        def close(self):pass
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='offline'))
    worker=run.CodexPolicy(tmp_path/'controller/codex_workspace','unused',task=task,transport_factory=Transport)
    try:
        text=captured['developerInstructions']
        assert captured['ephemeral'] is True
        assert 'automatically infers pi05' in text
        assert 'execution_status=failed or intent_status=misaligned' in text
        assert '0.05 m' in text and 'link6' in text
        for forbidden in ('# Shared manipulation guidance','# Native task success conditions:',
                          'Rigid-object pick-and-place preferences','rigid-object pick-and-place preferences','cap end','task_context',
                          'For container placement','make_kong checks','subtask vocabulary'):
            assert forbidden not in text
        names=[t['name'] for t in captured['dynamicTools']]
        assert 'robodojo_history' in names and 'pi05_infer' not in names
        assert not worker.task_references
        for name in ('task_success_conditions.md','manipulation_guidance.md'):
            assert not (worker.workspace/name).exists()
        for p in worker.agent_workspace.rglob('*.md'):
            content=p.read_text()
            assert 'Rigid-object pick-and-place preferences' not in content
            assert 'shared guidance and task-specific grasp region' not in content
        metadata=json.loads((worker.workspace.parent/'controller_protocol.json').read_text())
        assert metadata['prompt_profile']==pp.BASELINE and metadata['task_context'] is False
    finally:worker.close()


def test_first_observation_omits_context_and_keeps_original_instruction(tmp_path,monkeypatch):
    from .robodojo_server.client import RoboDojoTools
    from .test_contract import BenchmarkSim,Student,invoke
    monkeypatch.setenv(pp.ENV_KEY,pp.BASELINE)
    tools=RoboDojoTools(tmp_path,'arrange_largest_number',Student(),rpc_factory=BenchmarkSim,max_decisions=0)
    obs=invoke(tools)
    assert 'task_context' not in obs
    assert obs['instruction']==tools.obs['instruction']


def test_profile_disabled_reference_cannot_be_reenabled_by_ambient_env(tmp_path,monkeypatch):
    from .task_references import prepare_task_references
    monkeypatch.setenv(pp.ENV_KEY,pp.BASELINE)
    monkeypatch.setenv('ROLLOUT_TASK_REFERENCE_IMAGES','1')
    assert prepare_task_references('store_tools_in_toolbox',tmp_path)==[]
    with pytest.raises(ValueError,match='ROLLOUT_TASK_REFERENCE_IMAGES'):
        pp.identity(dict(baseline_env(),ROLLOUT_TASK_REFERENCE_IMAGES='1'))


@pytest.mark.parametrize('key,value',[
    ('ROLLOUT_CONTROLLER_ROUTE','subtask_feedforward_v1'),('ROLLOUT_ACTION_DOWNSAMPLING','error'),
    ('ROBODOJO_PI_FORCE_HOLD_ENABLED','1'),('ROLLOUT_ENABLE_SCENE_QUERY','1'),
    ('ROLLOUT_ENABLE_TASK_MEMORY','1'),('ROLLOUT_ENABLE_TEACHER_ABANDON','1')])
def test_incompatible_baseline_settings_fail_closed(key,value):
    with pytest.raises(ValueError,match=key):pp.identity(dict(baseline_env(),**{key:value}))
    assert pp.identity({})=={}  # Existing guided runtime identities stay unchanged.


def test_exact_200_case_pairing_and_uniform_baseline():
    root=Path(__file__).parent
    cfg=json.loads((root/'eval_queue.baseline_full200.json').read_text())
    old=[json.loads((root/p).read_text()) for p in ['eval_queue.round3_pilot20.json','eval_queue.round3_remaining180.json']]
    expected={(m['task'],*c) for q in old for m in q['models'] for c in m['cases']}
    actual={(m['task'],*c) for m in cfg['models'] for c in m['cases']}
    assert actual==expected and len(actual)==200 and cfg['workers']==8
    for model in cfg['models']:
        assert model['prompt_profile']==pp.BASELINE and model['policy_input']=='task_only'
        assert model['downsampling']=={'mode':'off'}
        assert model['checkpoint'].endswith('/59999')


def test_baseline_and_guided_cannot_mix_in_one_queue():
    from .eval_queue import validate_queue
    models=[dict(task='a',checkpoint='same',policy_input='task_only',prompt_profile=pp.BASELINE),
            dict(task='b',checkpoint='same',policy_input='task_only',prompt_profile=pp.GUIDED)]
    with pytest.raises(ValueError,match='one policy'):validate_queue(dict(scheduling='cross_task_queue_v1',models=models))


def test_runtime_identity_records_profile_and_preserves_legacy_guided_identity(tmp_path):
    from .test_regression_eval import setup
    from .regression_eval import runtime_settings
    _, panel, env=setup(tmp_path)
    original=runtime_settings(env,panel)
    assert 'prompt_profile' not in original
    updated=runtime_settings(dict(env,**baseline_env()),panel)
    assert updated['prompt_profile']==pp.BASELINE
    assert {k:v for k,v in updated.items() if k in original}==original


def test_campaign_selects_and_resets_profile_without_environment_leakage(tmp_path):
    from .test_eval_sequence import config
    from .eval_sequence import campaign_env
    from .test_regression_eval import setup
    _,_,env=setup(tmp_path)
    cfg=config();model=dict(cfg['models'][0],prompt_profile=pp.BASELINE)
    child=campaign_env(env,cfg,model,env['ROLLOUT_EVAL_MANIFEST'])
    assert pp.identity(child)['prompt_profile']==pp.BASELINE
    plain=campaign_env(dict(env,ROLLOUT_PROMPT_PROFILE=pp.BASELINE),cfg,cfg['models'][0],env['ROLLOUT_EVAL_MANIFEST'])
    assert pp.identity(plain)=={}


def test_prepared_runtime_rejects_changed_simulator_or_checkpoint():
    from .eval_queue import validate_expected_runtime
    cfg=dict(expected_runtime=dict(native_a100='same',checkpoint_sha256='original'))
    child=dict(ROLLOUT_RESUME_IDENTITY=json.dumps(cfg['expected_runtime']))
    validate_expected_runtime(cfg,[(child,[])])
    child['ROLLOUT_RESUME_IDENTITY']=json.dumps(dict(native_a100='same',checkpoint_sha256='trained'))
    with pytest.raises(ValueError,match='checkpoint_sha256'):validate_expected_runtime(cfg,[(child,[])])
