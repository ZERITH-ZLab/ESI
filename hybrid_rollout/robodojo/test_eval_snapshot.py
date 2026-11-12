"""Offline tests: no simulator, credentials, checkpoint loads or network."""
import json
from pathlib import Path

import pytest

from .deployment import snapshot, source_identity, verify_snapshot, RUNTIME_SOURCE_SCOPE
from .eval_snapshot import experiment_source, bind_queue_config


def checkout(tmp_path):
    root = tmp_path/'checkout'
    files = {
        'hybrid_rollout/__init__.py': '',
        'hybrid_rollout/robodojo/robodojo_server/client.py': '# policy v1\n',
        'hybrid_rollout/robodojo/dual_machine_episode_entrypoint.sh': '#!/bin/bash\n',
        'hybrid_rollout/robodojo/manipulation_guidance.md': 'Observe the result.',
        'hybrid_rollout/robodojo/task_completion_briefs.md': 'Task constraints.',
        'hybrid_rollout/robodojo/skill_auto_infer/feedforward_prompt.md': 'Choose a subtask.',
        'hybrid_rollout/robodojo/skill/context/teacher_context.md': 'Robot contract.',
        'hybrid_rollout/robodojo/robodojo_server/motion_distribution_profiles.json': '{}',
        'hybrid_rollout/robodojo/robodojo_server/collision_assets/manifest.json': '{}',
        'hybrid_rollout/robodojo/README.md': 'Guide.',
        'hybrid_rollout/robodojo/eval_queue.old.json': '{"name":"unused"}',
        'hybrid_rollout/robodojo/collection_suite.round1.json': '{}',
        'hybrid_rollout/robodojo/.env.server': 'private',
    }
    for rel, text in files.items():
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    return root


def test_experiment_freezes_once_and_resume_needs_no_checkout(tmp_path):
    root=checkout(tmp_path);queue=tmp_path/'eval/demo';env={'CODE_ROOT':str(root)}
    frozen=experiment_source({},env,queue)
    original=verify_snapshot(frozen)
    bind_queue_config(queue,queue,{'config':{'checkpoint':'one'}},frozen)
    client='hybrid_rollout/robodojo/robodojo_server/client.py'
    (root/client).write_text('# policy v2')
    resumed=experiment_source({}, {'CODE_ROOT':'/unavailable'}, queue)
    assert resumed==frozen and verify_snapshot(resumed)==original
    assert (resumed/client).read_text()=='# policy v1\n'
    # Per-episode snapshots must inherit the queue's runtime scope and identity.
    episode=tmp_path/'episode'
    assert snapshot(resumed,episode)==original==source_identity(episode)
    assert json.loads((episode/'source_manifest.json').read_text())['source_scope']==RUNTIME_SOURCE_SCOPE


def test_runtime_scope_ignores_docs_and_unused_recipes_but_keeps_prompts(tmp_path):
    root=checkout(tmp_path)
    original=source_identity(root,scope=RUNTIME_SOURCE_SCOPE)
    for name in ['README.md','eval_queue.old.json','collection_suite.round1.json']:
        (root/'hybrid_rollout/robodojo'/name).write_text('changed')
    assert source_identity(root,scope=RUNTIME_SOURCE_SCOPE)==original
    (root/'hybrid_rollout/robodojo/manipulation_guidance.md').write_text('New policy.')
    assert source_identity(root,scope=RUNTIME_SOURCE_SCOPE)!=original
    frozen=experiment_source({},dict(CODE_ROOT=str(root)),tmp_path/'queue')
    assert not (frozen/'hybrid_rollout/robodojo/.env.server').exists()
    assert not (frozen/'hybrid_rollout/robodojo/README.md').exists()
    assert (frozen/'hybrid_rollout/robodojo/skill/context/teacher_context.md').is_file()


@pytest.mark.parametrize('mutation',['edit','delete','add','manifest','symlink'])
def test_snapshot_tampering_is_rejected(tmp_path,mutation):
    root=checkout(tmp_path);queue=tmp_path/'queue'
    frozen=experiment_source({},dict(CODE_ROOT=str(root)),queue)
    path=frozen/'hybrid_rollout/robodojo/robodojo_server/client.py'
    if mutation=='edit':path.write_text('changed')
    elif mutation=='delete':path.unlink()
    elif mutation=='add':path.with_name('unexpected.py').write_text('new')
    elif mutation=='manifest':
        p=frozen/'source_manifest.json';record=json.loads(p.read_text());record['sha256']='wrong';p.write_text(json.dumps(record))
    else:
        path.unlink();path.symlink_to(root/'hybrid_rollout/robodojo/robodojo_server/client.py')
    with pytest.raises(ValueError,match='snapshot changed'):
        experiment_source({},dict(CODE_ROOT=str(root)),queue)


def test_missing_old_or_incomplete_snapshot_does_not_silently_refreeze(tmp_path):
    root=checkout(tmp_path);queue=tmp_path/'queue';queue.mkdir()
    (queue/'queue_config.json').write_text('{}')
    with pytest.raises(ValueError,match='no experiment source_snapshot'):
        experiment_source({},dict(CODE_ROOT=str(root)),queue)
    (queue/'source_snapshot').mkdir()
    with pytest.raises(FileNotFoundError):
        experiment_source({},dict(CODE_ROOT=str(root)),queue)


def test_advanced_legacy_pin_is_copied_then_original_is_unneeded(tmp_path):
    root=checkout(tmp_path);pin=tmp_path/'pin';queue=tmp_path/'queue'
    snapshot(root,pin)
    # Simulate a historical manifest without a source_scope key.
    p=pin/'source_manifest.json';record=json.loads(p.read_text());record.pop('source_scope');p.write_text(json.dumps(record))
    cfg=dict(execution_source=str(pin));env=dict(CODE_ROOT=str(root))
    frozen=experiment_source(cfg,env,queue)
    assert frozen!=pin and verify_snapshot(frozen)==verify_snapshot(pin)
    p.unlink()
    assert experiment_source(cfg,env,queue)==frozen


def test_dry_run_does_not_initialize_experiment(tmp_path):
    root=checkout(tmp_path);queue=tmp_path/'queue';temporary=tmp_path/'temporary';temporary.mkdir()
    frozen=experiment_source({},dict(CODE_ROOT=str(root)),queue,dry_run=True,temporary=temporary)
    bind_queue_config(queue,queue,dict(config={}),frozen,dry_run=True)
    assert frozen.is_relative_to(temporary) and not queue.exists()


def test_failed_copy_does_not_publish_partial_snapshot(tmp_path,monkeypatch):
    from . import eval_snapshot
    root=checkout(tmp_path);queue=tmp_path/'queue'
    def interrupted(root,dest,scope):
        dest.mkdir();(dest/'partial.py').write_text('partial');raise RuntimeError('copy interrupted')
    monkeypatch.setattr(eval_snapshot,'freeze_source',interrupted)
    with pytest.raises(RuntimeError):experiment_source({},dict(CODE_ROOT=str(root)),queue)
    assert not (queue/'source_snapshot').exists()
    assert not list(queue.iterdir())


def test_retry_config_and_source_are_bound_to_parent(tmp_path):
    root=checkout(tmp_path);queue=tmp_path/'queue'
    source=experiment_source({},dict(CODE_ROOT=str(root)),queue)
    frozen=dict(config=dict(checkpoint='one',task='task',seed=0),teacher=dict(model='m'))
    bind_queue_config(queue,queue,frozen,source)
    child=queue/'retries/a';child.mkdir(parents=True)
    selected=dict(frozen,retry_selection=dict(cases=['case']))
    bind_queue_config(queue,child,selected,source,retry=True)
    for field in ('checkpoint','task','seed'):
        changed=dict(selected,config=dict(frozen['config'],**{field:'changed'}))
        with pytest.raises(ValueError,match='Parent eval queue'):
            bind_queue_config(queue,child,changed,source,retry=True)
    newer=tmp_path/'new-source'
    (root/'hybrid_rollout/robodojo/robodojo_server/client.py').write_text('# v2')
    snapshot(root,newer,scope=RUNTIME_SOURCE_SCOPE)
    with pytest.raises(ValueError,match='config or source changed'):
        bind_queue_config(queue,queue,frozen,newer)


def test_default_release_configs_do_not_require_private_snapshot():
    root=Path(__file__).parent
    for name in ('eval_queue.round3_final200_updated.json','eval_queue.baseline_full200.json'):
        assert 'execution_source' not in json.loads((root/name).read_text())


def test_queue_uses_same_snapshot_across_runs_and_subset_retry(tmp_path,monkeypatch):
    from . import eval_queue as q,task_eval,dual_campaign,parallel_collection
    root=checkout(tmp_path);shared=tmp_path/'shared'
    cfg=dict(name='demo',workers=1,models=[dict(experiment='task-eval',task='task',cases=[['standard',0]])])
    env=dict(CODE_ROOT=str(root),ROLLOUT_SHARED_ROOT=str(shared),ROBODOJO_SOURCE='unused')
    used=[];checked=[]
    monkeypatch.setattr(q,'validate_queue',lambda cfg:None)
    monkeypatch.setattr(task_eval,'freeze',lambda *args:{})
    monkeypatch.setattr(q,'read_panel',lambda path:json.loads(Path(path).read_text()))
    monkeypatch.setattr(q,'verify_assets',lambda *args,**kw:None)
    def prepare(cfg,env,prepared,source):
        used.append(Path(source))
        return [(dict(CODE_ROOT=str(source),ROLLOUT_EXPERIMENT_ID='task-eval',ROLLOUT_REPLICA_ID='3',ROBODOJO_TASK='task'),[dict(case_id='c')])]
    monkeypatch.setattr(q,'prepare_campaigns',prepare)
    slot=dict(ROLLOUT_WORKER_ID='worker_0',ROLLOUT_SIM_SSH_PORT='1',SIM_GPU='0',POLICY_PORT='2',SIM_PORT='3',ROLLOUT_SIM_REMOTE_PORT='4')
    monkeypatch.setattr(q,'queue_slots',lambda *args:[slot])
    monkeypatch.setattr(dual_campaign,'resource_keys',lambda *args:[])
    monkeypatch.setattr(dual_campaign,'acquire_resources',lambda *args,**kw:[])
    monkeypatch.setattr(q,'attach_identities',lambda *args:checked.append('identities'))
    monkeypatch.setattr(q,'validate_expected_runtime',lambda *args:checked.append('expected'))
    monkeypatch.setattr(parallel_collection,'run_parallel',lambda *args,**kw:0)
    monkeypatch.setattr(parallel_collection,'suite_jobs',lambda *args:[(0,(None,dict(case_id='c')))])
    monkeypatch.setattr(q,'save_report',lambda *args:dict(planned=1,completed=1,successes=1,failures=0))
    assert q.run(cfg,env)==0
    (root/'hybrid_rollout/robodojo/robodojo_server/client.py').write_text('# changed')
    assert q.run(cfg,env)==0
    assert q.run(cfg,env,selections=['task-eval=c'],retry_name='retry1')==0
    expected=shared/'evaluation/demo/source_snapshot'
    assert set(used)=={expected}
    assert (expected/'hybrid_rollout/robodojo/robodojo_server/client.py').read_text()=='# policy v1\n'
    assert checked==['identities','expected']*3
