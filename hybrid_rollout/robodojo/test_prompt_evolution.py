import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import pytest

from .prompt_evolution import bundle,schemas
from .prompt_evolution.config import settings
from .prompt_evolution.pipeline import support_check,after_collection,run
from .prompt_evolution.data import chunks,load_images


def change(scope='fill_pen_holder',operation='replace',ident='p2'):
    return dict(id='p1',scope=scope,operation=operation,rule_id=ident,title='Inspect the opening',
        text='If the opening is occluded, obtain a clear view before resuming insertion.',
        mechanism='lost observation',evidence_refs=['a','b'],success_refs=['c'],counterexample_refs=['c'],
        rationale='Repeated unresolved occlusion',regression_risks='Avoid unnecessary stops',
        why_existing_rule_insufficient='Trigger was unclear')


def dossier(key,case,category='strategy_failure',task='fill_pen_holder'):
    return dict(ref=key,verified=True,category=category,recovery='recovered' if category=='successful_recovery' else 'unresolved',
                task=task,case_id=case,why_strategy_failed='Repeated blind insertion')


def test_fixed_goals_and_runtime_aliases():
    base=bundle.seed();cfg=settings()
    with pytest.raises(ValueError,match='editable'):bundle.apply_patch(base,change(ident='p0'),cfg)
    with pytest.raises(ValueError,match='editable'):bundle.apply_patch(base,change(ident='p1'),cfg)
    c=change('sort_nesting_dolls_by_size',ident='p3');next_=bundle.apply_patch(base,c,cfg)
    assert next_['tasks']['sort_nesting_dolls_by_size'][3]['text']==c['text']
    assert next_['tasks']['sort_nesting_dolls_by_size_random'][3]['text']==c['text']
    altered=deepcopy(base);altered['tasks']['fill_pen_holder'][0]['text']='Skip the task'
    with pytest.raises(ValueError,match='semantics'):bundle.validate(bundle.seal(altered))
    assert base['sha256']!=next_['sha256']


def test_no_bundle_preserves_original_prompt_and_pinned_identity(monkeypatch):
    from .task_success_conditions import append_task_success_conditions,render_task_success_conditions,GUIDANCE
    from .dual_campaign import controller_route_identity
    monkeypatch.delenv(bundle.ENV,raising=False)
    old='BASE\n\n'+GUIDANCE.read_text()+'\n\n'+render_task_success_conditions('fill_pen_holder')
    assert append_task_success_conditions('BASE','fill_pen_holder')==old
    base=bundle.seed();monkeypatch.setenv(bundle.ENV,json.dumps(base))
    assert append_task_success_conditions('BASE','fill_pen_holder')==old
    env={'ROLLOUT_CONTROLLER_ROUTE':'subtask_feedforward_v1',bundle.ENV:json.dumps(base)}
    assert controller_route_identity(env)['guidance_bundle_sha256']==base['sha256']
    env.pop(bundle.ENV)
    assert 'guidance_bundle_sha256' not in controller_route_identity(env)


def test_easy_errors_frequency_not_sufficient_and_duplicate_cases_not_independent():
    ds={k:dossier(k,k) for k in ['a','b']};ds['c']=dossier('c','c','successful_recovery')
    support_check(change(),ds,settings())
    ds['a']['category']='easy_execution_error'
    with pytest.raises(ValueError,match='easy'):support_check(change(),ds,settings())
    ds['a']['category']='strategy_failure';ds['b']['case_id']='a'
    with pytest.raises(ValueError,match='independent'):support_check(change(),ds,settings())
    ds['b']['case_id']='b';ds['c']['recovery']='unknown'
    with pytest.raises(ValueError,match='alternative'):support_check(change(),ds,settings())


def test_global_requires_multiple_tasks():
    ds={k:dossier(k,k) for k in ['a','b','d']};ds['c']=dossier('c','c','successful_recovery')
    p=change('shared');p['evidence_refs'].append('d')
    with pytest.raises(ValueError,match='cross-task'):support_check(p,ds,settings())
    ds['d']['task']='organize_table';support_check(p,ds,settings())


def test_image_budget_sparse_and_no_repeat(tmp_path):
    from PIL import Image
    root=tmp_path/'controller/observations'
    for n in range(17):
        f=root/f'{n:03}';f.mkdir(parents=True)
        (f/'observation.json').write_text(json.dumps({'step_id':n*15}))
        Image.new('RGB',(900,600)).save(f/'cam_high.png')
    cfg=settings();seen=set();req=[dict(start_decision=0,end_decision=16,camera='cam_high',sampling='sparse',question='Did the object remain held?')]
    files,meta,denied=load_images({'archive':str(tmp_path)},req,cfg,tmp_path/'preview',seen,24,set(range(17)))
    assert [m['decision'] for m in meta]==[0,8,16]
    assert Image.open(files[0]).size[0]==640
    assert not load_images({'archive':str(tmp_path)},req,cfg,tmp_path/'preview',seen,24,set(range(17)))[0]
    req[0]['sampling']='dense'
    assert load_images({'archive':str(tmp_path)},req,cfg,tmp_path/'preview',set(),24,set(range(17)))[2]


def test_chunks_cover_all_records():
    records=[{'decision':i,'text':'x'*20} for i in range(40)]
    out=list(chunks(records,160));assert {r['decision'] for c in out for r in c}==set(range(40))
    assert all(sum(len(json.dumps(r,ensure_ascii=False)) for r in c)<=160 for c in out)




def test_hook_disabled_and_not_completed(tmp_path):
    assert after_collection({}, {},tmp_path,0) is None
    (tmp_path/'status.json').write_text(json.dumps({'state':'completed_with_pending_recollection'}))
    with patch('hybrid_rollout.robodojo.prompt_evolution.pipeline.run') as model:
        result=after_collection({}, {'ROLLOUT_PROMPT_EVOLUTION':'1'},tmp_path,0)
        assert result['state']=='deferred';model.assert_not_called()


def test_full_pipeline_without_model_calls_and_resume(tmp_path):
    suite=tmp_path/'collection_suites/demo';suite.mkdir(parents=True)
    cfg=dict(schema='robodojo.collection_suite.v1',name='demo',tasks=['fill_pen_holder'],account_pool_file='unused')
    (suite/'config.json').write_text(json.dumps(cfg))
    archive=tmp_path/'episode';(archive/'controller').mkdir(parents=True)
    history=[dict(decision=0,start_tick=0,end_tick=15,executed_steps=15,response={'mode':'student','reason':'Progress'})]
    (archive/'controller/history.json').write_text(json.dumps(history))
    row=dict(key='e',task='fill_pen_holder',case_id='case0',split='standard',archive=str(archive),
             history_sha256=bundle.digest(history),result={'success':True},failure_kind=None,receipt_state='completed')
    manifest=dict(episodes=[row],missing_cases=[],anomalies=[])
    count=[]
    from .test_prompt_conversation import answer
    def backend(session,prompt,payload,schema,checkpoint,images):
        count.append(1)
        if session.root.name=='lead' and payload.get('message'):
            return answer(reads=[dict(case_key='e',start_decision=0,end_decision=0,question='Verify ordinary progress')])
        return answer(session.root.name=='reviewer',action='finish')
    with patch('hybrid_rollout.robodojo.prompt_evolution.pipeline.inventory',return_value=manifest):
        result=run(suite,backend=backend)
        assert result['state']=='no_change' and len(count)==3
        assert run(suite,backend=backend)==result and len(count)==3
        assert not (Path(result['output'])/'published_bundle.json').exists()


def test_large_context_is_offline_only():
    cfg=settings();assert cfg['auto_compact_token_limit']==480000
    assert cfg['model_context_window']==544000
    from .skill_auto_infer.memory import compact_limit
    assert compact_limit({})==96000




def test_context_headroom():
    with pytest.raises(ValueError,match='headroom'):
        settings({'auto_compact_token_limit':480000,'model_context_window':272000})




def test_annotation_accounts_do_not_inherit_model_or_collection_pool(tmp_path):
    from .prompt_evolution.client import annotation_accounts
    path=tmp_path/'annotation.json'
    path.write_text(json.dumps(dict(backend='codex',profiles=['codex_c','codex_c_2'],workers=1,
        shared_root='/annotation/auth',codex_bin='/annotation/codex',proxy_file=None,
        model='gpt-6.1-sol',reasoning_effort='high')))
    result=annotation_accounts(path)
    assert result['profiles']==['codex_c']
    assert result['shared_root']=='/annotation/auth'
    assert 'model' not in result and 'reasoning_effort' not in result
    assert settings()['model']=='gpt-6-astra' and settings()['reasoning_effort']=='medium'


def test_annotation_lease_skips_busy_and_releases_on_call_error():
    from contextlib import contextmanager
    from .prompt_evolution.client import annotation_lease
    from .codex_backend import profiles
    released=[]
    @contextmanager
    def fake_lock(name,root):
        if name=='codex_c':raise ValueError('busy')
        try:yield 42
        finally:released.append(name)
    cfg=dict(profiles=['codex_c','codex_c_2'],shared_root='/annotation/auth')
    with patch.object(profiles,'account_lock',fake_lock),patch.object(profiles,'validate_credential'):
        with pytest.raises(RuntimeError,match='model failure'):
            with annotation_lease(cfg,1) as (name,fd):
                assert name=='codex_c_2' and fd==42
                raise RuntimeError('model failure')
    assert released==['codex_c_2']


def historical_seed(tmp_path):
    source=tmp_path/'historical';source.mkdir()
    for name in ('task_completion_briefs.md','manipulation_guidance.md'):
        text=(bundle.ROOT/name).read_text()
        if name=='task_completion_briefs.md':text=text.replace('Hold the holder with one hand','Historically hold the holder with one hand')
        (source/name).write_text(text)
    return source,bundle.seed(source)


def test_historical_seed_needs_explicit_isolated_validation(tmp_path):
    source,base=historical_seed(tmp_path)
    candidate=bundle.apply_patch(base,change(),settings(),original=base)
    assert bundle.render(candidate,'fill_pen_holder',original=base)[1]
    with pytest.raises(ValueError,match='seed mismatch'):bundle.validate(candidate)
    with pytest.raises(ValueError,match='seed mismatch'):bundle.from_env({bundle.ENV:json.dumps(candidate)})


def test_historical_pipeline_cannot_publish_even_with_override(tmp_path):
    source,base=historical_seed(tmp_path)
    suite=tmp_path/'suite';suite.mkdir()
    cfg=dict(schema='robodojo.collection_suite.v1',name='demo')
    (suite/'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='isolated'):
        run(suite,historical_source=source)
    with pytest.raises(ValueError,match='separate'):
        run(suite,historical_source=source,experiment_only=True,output=suite/'experiment')


def test_replay_frozen_baseline_and_resume(tmp_path):
    from .prompt_evolution import replay as module
    source,base=historical_seed(tmp_path)
    suite=tmp_path/'suite';suite.mkdir();(suite/'config.json').write_text('{}')
    # Model source must actually belong to a selected archive.
    archive=tmp_path/'records'/'episode'
    src=archive/'source_snapshot/hybrid_rollout/robodojo';src.mkdir(parents=True)
    for name in module.FILES:(src/name).write_bytes((source/name).read_bytes())
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(suite_root=str(suite),baseline_source=str(src),
        baseline_sha256=module.fingerprints(src),output_root=str(tmp_path/'output'),settings={'publish':'auto_verified'})))
    manifest=dict(episodes=[dict(case_id='a',archive=str(archive))],missing_cases=[],anomalies=[])
    def fake_run(*args,**kw):
        assert kw['experiment_only'] and kw['overrides']['publish']=='candidate_only'
        assert module.fingerprints(kw['historical_source'])==module.fingerprints(src)
        return dict(state='prepared',output=str(kw['output']))
    with patch.object(module,'inventory',return_value=manifest),patch.object(module,'run',side_effect=fake_run):
        a=module.replay(config,prepare_only=True);b=module.replay(config,prepare_only=True)
    assert a==b
    assert (Path(a['replay_root'])/'before/fill_pen_holder.md').exists()
    (src/module.FILES[0]).write_text('tampered')
    with pytest.raises(ValueError,match='hash changed'):module.replay(config,prepare_only=True)
