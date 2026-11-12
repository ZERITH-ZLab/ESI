"""Offline multi-task selection, isolation, budgets and resume checks."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from . import collection_suite as suite
from .collection import runtime_settings
from .dual_campaign import configure
from .evaluation import canonical_sha256, freeze_panel, verify_assets
from .io import write_json
from .robodojo_server.step_budget import apply
from .task_success_conditions import render_task_success_conditions
from .test_evaluation import fake_native


def fixture(tmp_path):
    source=fake_native(tmp_path)
    for task in suite.TASKS:
        for row in suite.task_specs(task):
            p=source/'Assets/Eval_Layout/RoboDojo/arx_x5/0'/f'{row["runtime_task"]}_{row["layout_id"]}.json'
            p.write_text('{}')
            p=source/'task/RoboDojo/tasks'/f'{row["runtime_task"]}.py'
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('class Task:\n def __init__(self):\n  self.step_lim = 1600\n')
            if task=='imitate_sorting_sequence':
                p=source/'Assets/Traj/RoboDojo/imitate_sorting_sequence/0'/f'{row["layout_id"]}.pkl'
                p.write_bytes(b'native demonstration; never unpickle here')
    return source


def test_panel_selection_support_and_prompt_isolation(tmp_path):
    source=fixture(tmp_path);base=freeze_panel(source,'inventory');total=0
    for task in suite.TASKS:
        p=suite.freeze_task(source,task,[10,7],base)
        verify_assets(p,source)
        chosen=suite.pilot_cases(p);total+=len(chosen)
        assert len(p['cases'])==20
        expected=([(v,i) for v in ['standard','random'] for i in range(2)] if task in suite.RANDOM_TASKS
                  else [('standard',i) for i in range(3)])
        assert [(c['variant'],c['layout_id']) for c in chosen]==expected
        for c in p['cases']:
            brief=render_task_success_conditions(c['runtime_task'])
            assert ('Flip recovery:' in brief)==(task=='arrange_largest_number')
            assert bool(c['native_support_trajectories'])==(task=='imitate_sorting_sequence')
    assert total==40


@pytest.mark.parametrize("config_name,total,random_count", [
    ("collection_suite.pilot7.json", 24, 6),
    ("collection_suite.pilot7_revised.json", 25, 8),
])
def test_subset_preserves_original_case_ids_and_configures_grasp(tmp_path, config_name, total, random_count):
    from pathlib import Path
    from .dual_campaign import gripper_runtime_identity
    source = fixture(tmp_path)
    cfg = json.loads(Path(__file__).with_name(config_name).read_text())
    ckpt = tmp_path/'checkpoint'
    (ckpt/'params').mkdir(parents=True)
    (ckpt/'assets/arx_x5_sim').mkdir(parents=True)
    (ckpt/'assets/arx_x5_sim/norm_stats.json').write_text('{}')
    cfg['checkpoint'] = str(ckpt)
    path = tmp_path/'config.json'
    write_json(path, cfg)
    cfg = suite.read_config(path)
    cases = []
    for task in cfg['tasks']:
        panel = suite.freeze_task(source, task, [10, 7])
        selected = suite.pilot_cases(panel)
        assert all(c['replica_id'] == suite.TASKS.index(task) for c in selected)
        cases.extend(selected)
        env = suite.campaign_env(dict(ROLLOUT_SHARED_ROOT=str(tmp_path),
            ROBODOJO_PI_FORCE_HOLD_ENABLED='0'), cfg, panel, 'panel', 'control')
        assert env['ROLLOUT_CONTROLLER_ROUTE'] == 'auto_infer_v1'
        grasp = gripper_runtime_identity(env)['gripper_execution']
        assert grasp['pi_force_hold_enabled'] and grasp['contact_enabled']
        assert grasp['config']['max_extra_steps'] == 8
    assert len(cases) == total
    assert sum(c['variant'] == 'random' for c in cases) == random_count
    for tasks in ([], ['organize_table']*2, ['organize_table_random'], 'organize_table'):
        write_json(path, dict(cfg, tasks=tasks))
        with pytest.raises(ValueError, match='subset'):
            suite.read_config(path)


def test_grasp_resume_identity_detects_changes_and_preserves_legacy():
    from .dual_campaign import gripper_runtime_identity
    assert gripper_runtime_identity({}) == {}
    env = dict(ROBODOJO_GRIPPER_CONTACT_ENABLED='1', ROBODOJO_PI_FORCE_HOLD_ENABLED='1')
    baseline = gripper_runtime_identity(env)
    assert baseline != gripper_runtime_identity(dict(env, ROBODOJO_PI_FORCE_HOLD_ENABLED='0'))
    assert baseline != gripper_runtime_identity(dict(env, ROBODOJO_PI_FORCE_HOLD_CONFIG='{"stable_delta_m":0.0002}'))
    with pytest.raises(ValueError, match='requires'):
        gripper_runtime_identity(dict(env, ROBODOJO_GRIPPER_CONTACT_ENABLED='0'))


def test_round1_full20_queue_and_legacy_pilot_selection(tmp_path):
    from pathlib import Path
    from .parallel_collection import suite_jobs
    cfg = json.loads(Path(__file__).with_name('collection_suite.round1.json').read_text())
    assert len(cfg['tasks']) == len(set(cfg['tasks'])) == 10
    assert 'sweep_blocks' in cfg['tasks']
    assert not {'store_tools_in_toolbox', 'pick_from_conveyor_by_image'} & set(cfg['tasks'])
    source = fixture(tmp_path)
    base = freeze_panel(source, 'inventory')
    campaigns = []
    for task in cfg['tasks']:
        panel = suite.freeze_task(source, task, cfg['budget_multiplier'], base)
        path = tmp_path/f'{task}.json'
        write_json(path, panel)
        env = suite.campaign_env(dict(ROLLOUT_SHARED_ROOT=str(tmp_path)), cfg, panel, path, tmp_path/'workers.json')
        resolved, cases = configure(env)
        assert len(cases) == 20 and env['ROLLOUT_COUNT'] == '20'
        expected = ([(v, i) for v in ('standard', 'random') for i in range(10)]
                    if task in suite.RANDOM_TASKS else [('standard', i) for i in range(20)])
        assert [(c['variant'], c['layout_id']) for c in cases] == expected
        assert all(c['replica_id'] == suite.TASKS.index(task) for c in cases)
        assert suite.selected_cases(panel, {}) == suite.pilot_cases(panel)
        with pytest.raises(ValueError, match='case_selection'):
            suite.selected_cases(panel, {'case_selection': 'typo'})
        campaigns.append((resolved, cases))
    jobs = suite_jobs(campaigns)
    assert len(jobs) == 200
    assert len({case['case_id'] for _, (_, case) in jobs}) == 200
    assert sum(case['variant'] == 'random' for _, (_, case) in jobs) == 40
    assert [task for task, _ in jobs[:10]] == list(range(10))


def test_runtime_budget_selection_and_resume(tmp_path):
    source=fixture(tmp_path);p=suite.freeze_task(source,'imitate_sorting_sequence',[10,7])
    manifest=tmp_path/'panel.json';write_json(manifest,p)
    cfg=dict(name='pilot',workers=4,model='gpt-6-astra',reasoning_effort='xhigh',
             account_pool_file='/test/pool.json',checkpoint='/test/base')
    env=suite.campaign_env(dict(ROLLOUT_SHARED_ROOT=str(tmp_path),ROLLOUT_STRICT_EVAL='1',
        ROLLOUT_CONTROLLER_ROUTE='subtask_feedforward_v1',ROLLOUT_CASE_IDS='stale'),cfg,p,manifest,tmp_path/'workers.json')
    resolved,cases=configure(env)
    assert len(cases)==3 and resolved['ROBODOJO_TASK']=='imitate_sorting_sequence'
    assert env['ROLLOUT_STRICT_EVAL']=='0' and env['ROLLOUT_CONTROLLER_ROUTE']=='auto_infer_v1'
    assert env['ROLLOUT_RUN_MODE']=='fresh' and int(env['ROLLOUT_COLLECTION_STEP_LIMIT'])==2286
    assert runtime_settings(env,p)['collection_protocol']['native_step_limit']==1600
    native=SimpleNamespace(step_lim=1600)
    assert apply(native,2286,p)['effective_step_limit']==2286
    with pytest.raises(ValueError):apply(SimpleNamespace(step_lim=1050),2286,p)
    with pytest.raises(ValueError):apply(SimpleNamespace(step_lim=1600),1500,p)
    prior=tmp_path/'results/pilot__imitate_sorting_sequence/_replicas/replica_6/attempt_0'
    prior.mkdir(parents=True)
    assert suite.campaign_env(env,cfg,p,manifest,tmp_path/'workers.json')['ROLLOUT_RUN_MODE']=='resume'


def test_round2_feedforward_keeps_collection_features_and_all_200_cases(tmp_path):
    from pathlib import Path
    from .dual_campaign import gripper_runtime_identity, controller_route_identity
    from .parallel_collection import suite_jobs
    from .skill_auto_infer import task_memory, abandon
    cfg = json.loads(Path(__file__).with_name('collection_suite.round2.json').read_text())
    ckpt = tmp_path/'checkpoint'
    (ckpt/'params').mkdir(parents=True)
    (ckpt/'assets/arx_x5_sim').mkdir(parents=True)
    (ckpt/'assets/arx_x5_sim/norm_stats.json').write_text('{}')
    cfg['checkpoint'] = str(ckpt)
    path = tmp_path/'config.json'; write_json(path, cfg)
    cfg = suite.read_config(path)
    source = fixture(tmp_path); base = freeze_panel(source, 'inventory'); campaigns = []
    for task in cfg['tasks']:
        panel = suite.freeze_task(source, task, cfg['budget_multiplier'], base)
        manifest = tmp_path/f'{task}.json'; write_json(manifest, panel)
        env = suite.campaign_env(dict(ROLLOUT_SHARED_ROOT=str(tmp_path),
            ROLLOUT_CONTROLLER_ROUTE='auto_infer_v1'), cfg, panel, manifest, tmp_path/'control.json')
        resolved, cases = configure(env)
        assert resolved['ROLLOUT_CONTROLLER_ROUTE'] == 'subtask_feedforward_v1'
        assert resolved['CHECKPOINT'] == str(ckpt)
        assert task_memory.enabled(resolved) and abandon.enabled(resolved)
        assert resolved['ROLLOUT_ENABLE_SCENE_QUERY'] == '1'
        assert resolved['ROLLOUT_STRICT_EVAL'] == '0'
        assert controller_route_identity(resolved)['conditioning_version'] == 'subtask_feedforward_v1'
        hold = gripper_runtime_identity(resolved)['gripper_execution']['config']
        assert hold['tighten_mode'] == 'zero' and hold['flywheel_round'] == 2
        campaigns.append((resolved, cases))
    jobs = suite_jobs(campaigns)
    assert len(jobs) == 200
    assert len({case['case_id'] for _, (_, case) in jobs}) == 200
    assert sum(case['variant'] == 'random' for _, (_, case) in jobs) == 40
    write_json(path, dict(cfg, controller_route='auto_infer_v1'))
    with pytest.raises(ValueError, match='route'):
        suite.read_config(path)


def test_reject_changed_case_or_demo_mapping(tmp_path):
    source=fixture(tmp_path);p=suite.freeze_task(source,'imitate_sorting_sequence',[10,7])
    for change in ['case','demo']:
        bad=deepcopy(p)
        if change=='case':bad['cases'][0]['layout_id']=19
        else:bad['cases'][0]['native_support_trajectories']=[]
        bad['panel_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='panel_sha256'})
        with pytest.raises(ValueError):suite.validate_task_panel(bad)


def test_unified_queue_routes_every_case_to_its_own_task_budget_and_identity(tmp_path, monkeypatch):
    import os
    import time
    from pathlib import Path
    from . import parallel_collection as parallel
    from .deployment import source_identity
    from .test_parallel_collection import environment
    source=fixture(tmp_path)
    code=tmp_path/'code'
    (code/'hybrid_rollout/robodojo').mkdir(parents=True)
    (code/'hybrid_rollout/robodojo/sample.py').write_text('# source fixture\n')
    control=tmp_path/'workers.json';write_json(control,dict(desired_workers=8))
    cfg=dict(name='pilot',workers=8,model='gpt-6-astra',reasoning_effort='xhigh',
             account_pool_file='/test/pool.json',checkpoint='/test/base')
    campaigns=[]
    base=freeze_panel(source,'inventory')
    for task in suite.TASKS:
        p=suite.freeze_task(source,task,[10,7],base)
        manifest=tmp_path/f'{task}.json';write_json(manifest,p)
        e,cases=configure(suite.campaign_env(dict(environment(),ROLLOUT_SHARED_ROOT=str(tmp_path)),
                                           cfg,p,manifest,control))
        e.update(CODE_ROOT=str(code),ROLLOUT_RESUME_IDENTITY=json.dumps(
            dict(source_sha256=source_identity(code),task=task)))
        campaigns.append((e,cases))
    jobs=parallel.suite_jobs(campaigns)
    assert len(jobs)==40 and [t for t,j in jobs[:12]]==list(range(12))
    monkeypatch.setattr(parallel,'SUPERVISOR_POLL_SECONDS',.01)
    monkeypatch.setattr(parallel,'AccountPool',lambda env:SimpleNamespace())
    def fake_run(env,cases,*,pool,job_source,allocate_attempt,raise_on_retire):
        assert raise_on_retire
        index,(repeat,case)=next(job_source())
        assert case in cases and case['task']==env['ROBODOJO_TASK']
        panel=json.loads(Path(env['ROLLOUT_EVAL_MANIFEST']).read_text())
        assert panel['collection']['task']==case['task']
        assert env['ROLLOUT_COLLECTION_STEP_LIMIT']==str(panel['collection']['step_limit'])
        assert json.loads(env['ROLLOUT_RESUME_IDENTITY'])['task']==case['task']
        assert env['ROLLOUT_EXPERIMENT_ID']=='pilot__'+case['task']
        assert env['ROLLOUT_CASE_IDS']==','.join(c['case_id'] for c in cases)
        assert 'source_snapshot' in env['CODE_ROOT']
        assert repeat==1
        write_json(tmp_path/f'job_{index}.json',dict(attempt=allocate_attempt(),case=case['case_id'],
            host=env['ROLLOUT_SIM_SSH_PORT'],worker=env['ROLLOUT_WORKER_ID']))
        time.sleep(.02)
        return 0
    monkeypatch.setattr(parallel,'run_pool',fake_run)
    assert parallel.run_parallel(parallel.slots(campaigns[0][0]),campaigns[0][1],
                                 campaigns=campaigns,output=tmp_path/'queue')==0
    rows=[json.loads(p.read_text()) for p in tmp_path.glob('job_*.json')]
    assert len(rows)==len({r['case'] for r in rows})==len({r['attempt'] for r in rows})==40
    assert {r['host'] for r in rows}=={'4205','5727'}


def test_retired_suite_job_is_returned_without_ack(tmp_path, monkeypatch):
    import multiprocessing as mp
    from . import parallel_collection as p
    from .account_pool import WorkerRetired
    ctx=mp.get_context('fork')
    allocator=p.Allocation(ctx,[(0,(1,dict(case_id='case')))],0)
    source=allocator.source()
    def cancelled(*a,**kw):raise WorkerRetired('before starting')
    monkeypatch.setattr(p,'run_pool',cancelled)
    with pytest.raises(WorkerRetired):
        p.run_suite_jobs(dict(ROLLOUT_WORKER_ID='worker_0'),
            [(dict(ROBODOJO_TASK='task',ROLLOUT_EXPERIMENT_ID='synthetic-retirement'),
              [dict(case_id='case')])],source,allocator,SimpleNamespace())
    source.close()
    assert list(allocator.states)==[0]


def test_subset_dispatch_uses_canonical_panel_index_and_resumes(tmp_path):
    from contextlib import contextmanager
    from pathlib import Path
    from .account_pool import run_pool
    from .deployment import source_identity
    from .collection import sampling_job_index
    source=fixture(tmp_path);panel=suite.freeze_task(source,'arrange_largest_number',[10,7])
    manifest=tmp_path/'panel.json';write_json(manifest,panel)
    cfg=dict(name='pilot',workers=8,model='gpt-6-astra',reasoning_effort='xhigh',
             account_pool_file='/test/pool.json',checkpoint='/test/base')
    env,cases=configure(suite.campaign_env(dict(ROLLOUT_SHARED_ROOT=str(tmp_path)),
                                          cfg,panel,manifest,tmp_path/'workers.json'))
    code=tmp_path/'code';(code/'hybrid_rollout/robodojo').mkdir(parents=True)
    (code/'hybrid_rollout/robodojo/sample.py').write_text('# fixture\n')
    identity=dict(source_sha256=source_identity(code))
    env.update(CODE_ROOT=str(code),ROLLOUT_AUTH_ROOT=str(tmp_path/'auth'),ROLLOUT_RESUME_IDENTITY=json.dumps(identity))
    class Pool:
        config_path=tmp_path/'pool.json'
        @contextmanager
        def acquire(self,*a,**kw):
            yield dict(auth_profile='codex_a_5',account_fingerprint='test',credential_revision='test',fds=(1,2))
    called=[]
    def runner(child,fds):
        # Third selected case is panel case 10, despite global queue index 21.
        assert child['ROLLOUT_COLLECTION_JOB_INDEX']=='10'
        assert sampling_job_index(panel,child['ROLLOUT_CASE_ID'],1)==10
        called.append(child['ROLLOUT_CASE_ID'])
        archive=tmp_path/'archive';(archive/'dual_machine').mkdir(parents=True)
        (archive/'dual_machine/remote_cleanup_status.txt').write_text('0')
        receipt=dict(all_owned_processes_exited=True,ports_released=True)
        row=dict(case_id=child['ROLLOUT_CASE_ID'],repeat_id=1,archive=str(archive),reset=receipt,
                 state='completed',artifact_status='verified',result=dict(complete=True),
                 evaluation_outcome=dict(complete=True))
        root=tmp_path/'results'/env['ROLLOUT_EXPERIMENT_ID']/'_replicas'/f'replica_{env["ROLLOUT_REPLICA_ID"]}'/f'attempt_{child["ROLLOUT_ATTEMPT"]}'
        root.mkdir(parents=True)
        write_json(root/'batch.json',dict(final_reset=receipt,episodes=[row],resume_identity=identity))
        return 0
    source=lambda:iter([(21,(1,cases[2]))])
    assert run_pool(env,cases,pool=Pool(),runner=runner,job_source=source)==0
    assert run_pool(dict(env,ROLLOUT_RUN_MODE='resume'),cases,pool=Pool(),runner=runner,job_source=source)==0
    assert called==[cases[2]['case_id']]


@pytest.mark.parametrize("upgrade", [False, True])
@pytest.mark.parametrize('queue_state', ['completed', 'completed_with_pending_recollection',
                                       'dispatch_paused_connection_errors'])
def test_unified_launch_locks_all_tasks_and_checks_both_hosts(tmp_path, monkeypatch, upgrade, queue_state):
    from . import dual_campaign as dual, parallel_collection as parallel
    from .test_parallel_collection import environment
    source=fixture(tmp_path);base=freeze_panel(source,'inventory')
    cfg=dict(name='pilot',workers=8,model='gpt-6-astra',reasoning_effort='xhigh',
             account_pool_file='/test/pool.json',checkpoint='/test/base')
    env=dict(environment(),ROLLOUT_SHARED_ROOT=str(tmp_path))
    control=tmp_path/'workers.json';write_json(control,dict(desired_workers=8))
    prepared=[]
    for task in suite.TASKS:
        p=suite.freeze_task(source,task,[10,7],base)
        path=tmp_path/f'{task}.json';write_json(path,p)
        prepared.append((p,path,control))
    old_receipt = None
    if upgrade:
        panel = prepared[0][0]
        old = dict(task=suite.TASKS[0], source_sha256='old',
                   selected_case_ids=[c['case_id'] for c in suite.pilot_cases(panel)])
        old_receipt = tmp_path/'results'/('pilot__'+suite.TASKS[0])/'_account_pool/run_old/pool_run.json'
        old_receipt.parent.mkdir(parents=True)
        write_json(old_receipt, dict(resume_identity=old))
        old_bytes = old_receipt.read_bytes()
    keys=[];checked=[]
    def lock(root,values,*,wait):
        assert wait;keys.extend(values);return []
    def identity(e):
        checked.append((e['ROBODOJO_TASK'],e['ROLLOUT_SIM_SSH_PORT']))
        return dict(task=e['ROBODOJO_TASK'],source_sha256='fixture',
                    selected_case_ids=e['ROLLOUT_CASE_IDS'].split(','))
    def dispatch(environments,cases,*,campaigns,output):
        assert len(environments)==8 and len(campaigns)==12
        for e,_ in campaigns:
            expected = '1' if upgrade and e['ROBODOJO_TASK']==suite.TASKS[0] else '0'
            assert e['ROLLOUT_ALLOW_SOURCE_UPGRADE']==expected
        if upgrade:
            from .resume_upgrade import accepted_identities
            identity_value=json.loads(campaigns[0][0]['ROLLOUT_RESUME_IDENTITY'])
            assert old in accepted_identities(old_receipt.parents[2],identity_value)
            assert old_receipt.read_bytes()==old_bytes
        assert len(parallel.suite_jobs(campaigns))==40
        assert all(json.loads(e['ROLLOUT_RESUME_IDENTITY'])['task']==e['ROBODOJO_TASK'] for e,_ in campaigns)
        assert len({tuple(k) for k in keys if k[0]=='sim_gpu'})==4
        assert len({tuple(k) for k in keys if k[0]=='campaign'})==12
        assert len({tuple(k) for k in keys if k[0]=='local_port'})==16
        write_json(output.parent/'status.json', dict(state=queue_state,
                   connection_recovery=dict(pending_recollection_job_indices=[0])))
        return 1 if queue_state == 'dispatch_paused_connection_errors' else 0
    monkeypatch.setattr(dual,'acquire_resources',lock)
    monkeypatch.setattr(dual,'runtime_identity',identity)
    monkeypatch.setattr(parallel,'run_parallel',dispatch)
    assert suite.run_unified(cfg,env,prepared,tmp_path,allow_source_upgrade=upgrade)==(
        1 if queue_state == 'dispatch_paused_connection_errors' else 0)
    final_status = json.loads((tmp_path/'status.json').read_text())
    assert final_status['state'] == queue_state
    assert final_status['connection_recovery']['pending_recollection_job_indices'] == [0]
    assert len(checked)==2*len(suite.TASKS)
    assert {port for task,port in checked}=={'4205','5727'}
