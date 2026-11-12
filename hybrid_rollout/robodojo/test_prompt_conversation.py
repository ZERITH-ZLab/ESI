import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import pytest

from .prompt_evolution import bundle
from .prompt_evolution.config import settings
from .prompt_evolution.session import Conversation
from .prompt_evolution.evidence import Evidence
from .prompt_evolution.conversation import run_conversation,validate_audit,final_bundle


def fixture_data(tmp_path):
    rows=[]
    for key in ('a','b','c'):
        archive=tmp_path/key;(archive/'controller').mkdir(parents=True)
        history=[dict(decision=i,start_tick=i*3,end_tick=(i+1)*3,executed_steps=3,
            response=dict(mode='eef' if i<2 else 'student',reason='Contact cleared; supported placement.',
                assessment=dict(execution_evidence='Object is resting on support.'))) for i in range(3)]
        (archive/'controller/history.json').write_text(json.dumps(history))
        rows.append(dict(key=key,task='fill_pen_holder',case_id='case_'+key,split='standard',archive=str(archive),
            history_sha256=bundle.digest(history),result=dict(success=key=='c'),failure_kind=None,receipt_state='completed'))
    return dict(episodes=rows,missing_cases=[],anomalies=[])


def issue(key):
    return dict(case_key=key,group_id='occluded_contact',id='recovery',start_decision=0,end_decision=2,
        category='successful_recovery' if key=='c' else 'strategy_failure',recovery='recovered' if key=='c' else 'unresolved',
        mechanism='Contact could not be observed',observed_facts='Changed viewpoint showed support.',
        existing_strategy='Keep inserting blindly',why_strategy_failed='Repeated blind insertion',
        alternative='Observe support before insertion',alternative_basis='observed_success',
        evidence_decisions=[0,2],counterexample_decisions=[1],online_observable_trigger='Blocked view',image_ids=[],verified=True)


def proposal():
    return dict(id='visibility',scope='fill_pen_holder',operation='replace',rule_id='p2',title='',
        text='When the opening is occluded, obtain a clear view before resuming insertion.',
        mechanism='occluded contact',evidence_refs=['a:recovery','b:recovery'],success_refs=['c:recovery'],
        counterexample_refs=['c:recovery'],rationale='Independent cases',regression_risks='Avoid excess pauses',
        why_existing_rule_insufficient='Unclear trigger')


def answer(audit=False,**kwargs):
    base=dict(action='continue',reason='Compare cases',memory='Compare unresolved contacts with recovered control.',queries=[],reads=[],images=[])
    base.update(dict(verdicts=[]) if audit else dict(updates=[],patches=[],deferred_issues=[]))
    return dict(base,**kwargs)


def reads():return [dict(case_key=k,start_decision=0,end_decision=2,question='Was the support observed?') for k in 'abc']


def vote():
    return dict(patch_id='visibility',verdict='approve',reason='Original evidence supports conditional guidance',
        evidence_supports_change=True,same_mechanism=True,online_observable=True,preserves_goals_and_protocol=True,
        does_not_target_easy_errors=True,counterexamples_reviewed=True,concise_nonredundant=True)


def run_fixture(tmp_path,backend,prepare=False):
    suite=tmp_path/'suite';suite.mkdir(exist_ok=True)
    manifest=fixture_data(tmp_path) if not (tmp_path/'a').exists() else json.loads((tmp_path/'input.json').read_text())
    (tmp_path/'input.json').write_text(json.dumps(manifest))
    cfg=dict(name='demo',schema='robodojo.collection_suite.v1')
    opts=settings(dict(workflow='conversation_v2',publish='candidate_only'))
    return run_conversation(suite,cfg,opts,manifest,bundle.seed(),dict(experiment_only=True),tmp_path/'result',
                            backend=backend,prepare_only=prepare)


def scripted(session,prompt,payload,schema,checkpoint,images):
    audit=session.root.name=='reviewer'
    if payload.get('message'):return answer(audit,reads=reads())
    if audit:return answer(True,action='finish',verdicts=[vote()])
    return answer(action='finish',updates=[issue(k) for k in 'abc'],patches=[proposal()])


def test_two_persistent_threads_and_independent_review(tmp_path):
    calls=[]
    def backend(session,*args):
        calls.append((session.root.name,session.state['thread_id']))
        return scripted(session,*args)
    result=run_fixture(tmp_path,backend)
    assert result['state']=='candidate_only' and result['whole_bundle_approved']
    assert calls==[('lead',None),('lead','test-thread-lead'),('reviewer',None),('reviewer','test-thread-reviewer')]
    root=tmp_path/'result'
    assert not (root/'published_bundle.json').exists()
    assert not (root/'next_collection_config.json').exists()
    candidate=json.loads((root/'candidate_bundle.json').read_text())
    assert candidate['tasks']['fill_pen_holder'][2]['text']==proposal()['text']
    assert run_fixture(tmp_path,backend)==result and len(calls)==4
    usage=json.loads((root/'usage_summary.json').read_text());assert usage['attempts']==4
    timing=json.loads(next((root/'lead/turns').glob('*/attempt_*/timing.json')).read_text())
    assert timing['started_at'] and timing['ended_at'] and timing['duration_seconds']>=0


def test_resume_at_failed_turn_keeps_prior_evidence_and_thread(tmp_path):
    def interrupted(session,prompt,payload,*args):
        if session.root.name=='lead' and not payload.get('message'):raise OSError('transport disconnected')
        return scripted(session,prompt,payload,*args)
    with pytest.raises(OSError):run_fixture(tmp_path,interrupted)
    state=json.loads((tmp_path/'result/checkpoint.json').read_text())
    assert state['turn']==1 and set(state['knowledge']['read'])==set('abc')
    calls=[]
    def resumed(session,*args):
        calls.append((session.root.name,session.state['thread_id']))
        return scripted(session,*args)
    assert run_fixture(tmp_path,resumed)['whole_bundle_approved']
    assert calls[0]==('lead','test-thread-lead') and len(calls)==3


def test_auditor_cannot_approve_only_analyst_claims(tmp_path):
    store=Evidence(tmp_path/'evidence',fixture_data(tmp_path)['episodes'],settings())
    issues={k+':recovery':dict(issue(k),episode_key=k,ref=k+':recovery') for k in 'abc'}
    state=dict(issues=issues,reviews=[dict(state='pending_independent_audit',patch=proposal())],audit_knowledge=dict(read={},images={}))
    with pytest.raises(ValueError,match='original cited'):validate_audit(dict(verdicts=[vote()]),state,store)
    state['audit_knowledge']['read']={k:[0,1,2] for k in 'abc'}
    validate_audit(dict(verdicts=[vote()]),state,store)
    state['issues']['a:recovery']['image_ids']=['0:cam_high']
    with pytest.raises(ValueError,match='cited images'):validate_audit(dict(verdicts=[vote()]),state,store)


def test_coverage_and_duplicate_occurrences_not_incidence(tmp_path):
    store=Evidence(tmp_path/'evidence',fixture_data(tmp_path)['episodes'],settings())
    state=dict(issues={},knowledge=dict(read={},images={}))
    with pytest.raises(ValueError,match='complete recovery span'):store.update([issue('a')],state)
    store.fetch(reads(),[],state['knowledge'],tmp_path/'images')
    store.update([issue('a')],state)
    duplicate=issue('a');duplicate['id']='another_window';store.update([duplicate],state)
    table=store.problem_table(state['issues'])[0]
    assert table['distinct_cases']==1 and table['executed_steps_union']==9 and table['teacher_steps_union']==6
    assert store.coverage(state['knowledge'])['by_task']['fill_pen_holder']['fully_read_cases']==3
    with pytest.raises(ValueError,match='Unknown case_key'):
        store.fetch([dict(reads()[0],case_key='../../private')],[],state['knowledge'],tmp_path)


def test_native_compaction_reanchors_without_new_thread(tmp_path):
    c=Conversation(tmp_path,settings());w=tmp_path/'attempt';w.mkdir()
    (w/'events.jsonl').write_text(json.dumps(dict(type='thread.started',thread_id='original'))+'\n'+
        json.dumps(dict(type='item.completed',item=dict(type='contextCompaction')))+'\n')
    c.recover(w)
    assert c.state['thread_id']=='original' and c.state['needs_anchor']
    command=c.command(dict(codex_bin='/codex'),w,w/'schema',w/'last')
    assert '--ephemeral' not in command
    assert command[command.index('resume')+1]=='original'


def test_partial_audit_never_publishes_changed_subset():
    good=dict(patch=proposal(),state='approved',audit=vote())
    bad=deepcopy(good);bad['state']='reject';bad['patch']['id']='other'
    result,ok=final_bundle(bundle.seed(),[good,bad])
    assert not ok and result['tasks']==bundle.seed()['tasks']


def test_prepare_only_does_not_call_model(tmp_path):
    assert run_fixture(tmp_path,lambda *args:pytest.fail('model invoked'),prepare=True)['state']=='prepared'


def test_real_subprocess_resume_increment_and_compaction_anchor(tmp_path):
    import os
    import sys
    from contextlib import contextmanager
    from .prompt_evolution import session as module
    from .codex_backend import profiles
    from .prompt_evolution.schemas import obj,B
    fake=tmp_path/'fake-codex'
    fake.write_text('#!'+sys.executable+'\n'+'''import sys,json,pathlib
args=sys.argv[1:];data=sys.stdin.read();out=pathlib.Path(args[args.index('--output-last-message')+1])
(out.parent/'received.json').write_text(json.dumps({'args':args,'input':data}))
out.write_text('{"ok": true}')
print(json.dumps({'type':'thread.started','thread_id':'persistent-123'}))
if 'compact_me' in data:print(json.dumps({'type':'item.completed','item':{'type':'contextCompaction'}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'cached_input_tokens':2,'output_tokens':1}}))
''');fake.chmod(0o755)
    home=tmp_path/'home';home.mkdir();(home/'auth.json').write_text(json.dumps({'tokens':{'account_id':'dummy'}}))
    @contextmanager
    def lease(*args,**kwargs):
        with (tmp_path/'lease').open('a') as f:yield 'codex_c',f.fileno()
    accounts=dict(profiles=['codex_c'],shared_root=str(tmp_path),codex_bin=str(fake),proxy_file=None)
    with patch.object(module,'annotation_accounts',return_value=accounts),patch.object(module,'annotation_lease',lease), \
         patch.object(profiles,'account_home',return_value=home),patch.object(profiles,'clean_login_env',return_value=dict(os.environ)):
        s=Conversation(tmp_path/'lead',settings());check=dict(memory='Recovery evidence is saved.')
        for n,payload in enumerate([{'step':0},{'compact_me':True},{'step':2}]):
            assert s.ask(n,'INITIAL SYSTEM',payload,obj(ok=B),check)=={'ok':True}
    values=[json.loads(p.read_text()) for p in sorted((tmp_path/'lead/turns').glob('*/attempt_*/received.json'))]
    assert 'resume' not in values[0]['args'] and 'INITIAL SYSTEM' in values[0]['input']
    assert all('persistent-123' in v['args'] for v in values[1:])
    assert 'INITIAL SYSTEM' not in values[1]['input'] and 'durable_checkpoint' not in values[1]['input']
    assert 'durable_checkpoint' in values[2]['input']
    assert all('--ephemeral' not in v['args'] for v in values)
    summary=module.usage_summary(tmp_path)
    assert summary['usage']['input_tokens']==30 and summary['attempts_without_usage']==0


def test_no_change_still_has_independent_final_review(tmp_path):
    names=[]
    def backend(session,prompt,payload,schema,checkpoint,images):
        names.append(session.root.name)
        if session.root.name=='reviewer':return answer(True,action='finish',verdicts=[])
        if payload.get('message'):return answer(reads=reads())
        return answer(action='finish',deferred_issues=['No strategy failure supported'])
    assert run_fixture(tmp_path,backend)['state']=='no_change'
    assert names==['lead','lead','reviewer']


def test_saved_problem_queries_are_paged_and_do_not_mark_raw_evidence_read(tmp_path):
    from .prompt_evolution.conversation import query_pages
    cfg=settings();store=Evidence(tmp_path/'evidence',fixture_data(tmp_path)['episodes'],cfg)
    state=dict(issues={},knowledge=dict(read={},images={}),reviews=[])
    pages=query_pages([dict(kind='cases',offset=0,limit=2,question='Choose comparison cases')],store.index(),state,store,cfg)
    assert pages[0]['total']==3 and pages[0]['next_offset']==2
    assert state['knowledge']['read']=={}
    with pytest.raises(ValueError,match='pagination'):
        query_pages([dict(kind='issues',offset=-1,limit=1,question='x')],store.index(),state,store,cfg)


def test_saved_summary_reproposal_reuses_claims_but_audits_raw_evidence(tmp_path):
    manifest=fixture_data(tmp_path);base=bundle.seed();opts=settings(dict(publish='candidate_only'))
    saved={k+':recovery':dict(issue(k),episode_key=k,ref=k+':recovery',task='fill_pen_holder',case_id='case_'+k)
           for k in 'abc'}
    reuse=dict(issues=saved,reviews=[dict(patch=proposal(),scope='fill_pen_holder',state='candidate',reason='old budget exceeded')])
    calls=[]
    def backend(session,prompt,payload,schema,checkpoint,images):
        calls.append(session.root.name)
        assert 'records' not in json.dumps(checkpoint)
        if session.root.name=='lead':
            if payload.get('message'):
                assert payload['summary_count']==3
                return answer(queries=[dict(kind='cited_issues',offset=0,limit=3,question='Review frozen summaries'),
                    dict(kind='prior_patches',offset=0,limit=1,question='Read old proposal')])
            assert len(payload['query_results'][0]['entries'])==3
            return answer(action='finish',patches=[proposal()])
        if payload.get('message'):
            assert 'patch_catalog' in payload and 'evidence' not in payload
            return answer(True,reads=reads())
        return answer(True,action='finish',verdicts=[vote()])
    result=run_conversation(tmp_path/'suite',dict(name='test'),opts,manifest,base,
        dict(experiment_only=True),tmp_path/'result',reuse_analysis=reuse,backend=backend)
    assert result['whole_bundle_approved']
    state=json.loads((tmp_path/'result/checkpoint.json').read_text())
    assert state['knowledge']['read']=={}  # Imported summaries never count as fresh raw reads.
    assert set(state['audit_knowledge']['read'])==set('abc')
    assert state['issues']==saved and calls==['lead','lead','reviewer','reviewer']
    assert not (tmp_path/'result/next_collection_config.json').exists()


def test_incremental_budget_denial_does_not_mark_evidence_read(tmp_path):
    store=Evidence(tmp_path/'evidence',fixture_data(tmp_path)['episodes'],settings())
    knowledge=dict(read={},images={})
    payload,images=store.fetch(reads(),[],knowledge,tmp_path/'images',text_budget=10)
    assert len(payload['denied'])==3 and not payload['evidence']
    assert knowledge['read']=={} and not images


def test_saved_review_entry_freezes_source_and_forces_isolation(tmp_path):
    from .prompt_evolution.review_saved import replay_saved
    manifest=fixture_data(tmp_path);source=tmp_path/'old/analysis';source.mkdir(parents=True)
    data={'manifest':dict(suite=str(tmp_path/'suite'),config=settings(),inventory=manifest),
          'base_bundle':bundle.seed(),'dossiers':[],'patch_reviews':[],'status':dict(state='no_change')}
    for name,value in data.items():(source/(name+'.json')).write_text(json.dumps(value))
    original={p.name:p.read_bytes() for p in source.iterdir()}
    out=tmp_path/'new'
    result=replay_saved(source,out,overrides=dict(publish='auto_verified'),prepare_only=True,
        backend=lambda *a:pytest.fail('prepare must not call model'))
    assert result['state']=='prepared'
    cfg=json.loads((out/'analysis/manifest.json').read_text())
    assert cfg['config']['publish']=='candidate_only' and cfg['experiment_only']
    assert original=={p.name:p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError,match='overlaps'):
        replay_saved(source,source/'new',prepare_only=True)


def test_invalid_add_id_repaired_in_lead_before_audit_and_verdict_saved(tmp_path):
    calls=[]
    bad=dict(proposal(),operation='add',rule_id='keyboard_contact_response')
    fixed=dict(bad,rule_id='learned_keyboard_contact_response')
    def backend(session,prompt,payload,schema,checkpoint,images):
        calls.append((session.root.name,session.state['thread_id']))
        if session.root.name=='lead':
            if payload.get('message'):return answer(reads=reads())
            if payload.get('precheck_errors'):
                assert payload['precheck_errors'][0]['patch_id']==bad['id']
                assert 'learned_' in payload['precheck_errors'][0]['error']
                assert all(name=='lead' for name,_ in calls)
                return answer(action='finish',updates=[issue(k) for k in 'abc'],patches=[fixed])
            return answer(action='finish',updates=[issue(k) for k in 'abc'],patches=[bad])
        assert checkpoint['audit_contract']['expected_patch_ids']==['visibility']
        if payload.get('message'):return answer(True,reads=reads())
        return answer(True,action='finish',verdicts=[dict(vote(),verdict='needs_closed_loop',
            reason='Plausible but benefit needs a closed-loop comparison',evidence_supports_change=False)])
    result=run_fixture(tmp_path,backend)
    assert result['approved_patches']==0 and not result['whole_bundle_approved']
    assert [name for name,_ in calls]==['lead','lead','lead','reviewer','reviewer']
    assert calls[1][1]==calls[2][1]=='test-thread-lead'
    reviews=json.loads((tmp_path/'result/patch_reviews.json').read_text())
    assert reviews[0]['patch']['rule_id']=='learned_keyboard_contact_response'
    assert reviews[0]['state']==reviews[0]['audit']['verdict']=='needs_closed_loop'
    assert len(list((tmp_path/'result/proposal_attempts').glob('*.json')))==2


def test_pending_audit_pages_and_errors_exclude_failed_prechecks(tmp_path):
    from .prompt_evolution.conversation import query_pages,audit_contract
    cfg=settings();store=Evidence(tmp_path/'evidence',fixture_data(tmp_path)['episodes'],cfg)
    issues={k+':recovery':dict(issue(k),episode_key=k,ref=k+':recovery') for k in 'abc'}
    good=dict(state='pending_independent_audit',patch=proposal())
    bad=dict(state='candidate',patch=dict(proposal(),id='invalid'),reason='Invalid rule ID')
    state=dict(phase='auditing',issues=issues,reviews=[bad,good],audit_knowledge=dict(read={},images={}))
    pages=query_pages([dict(kind='patches',offset=0,limit=20,question='Which patches need verdicts?')],[],state,store,cfg)
    assert pages[0]['total']==1 and pages[0]['entries']==[good]
    assert pages[0]['audit_contract']['expected_patch_ids']==['visibility']
    with pytest.raises(ValueError,match='unexpected.*invalid'):
        validate_audit(dict(verdicts=[dict(vote(),patch_id='invalid')]),state,store)
    state['reviews']=[bad]
    assert audit_contract(state)['expected_patch_ids']==[]
    assert query_pages([dict(kind='cited_issues',offset=0,limit=20,question='Citations?')],[],state,store,cfg)[0]['total']==0
    validate_audit(dict(verdicts=[]),state,store)


def test_incomplete_audit_cannot_publish_approved_subset():
    good=dict(patch=proposal(),state='approved',audit=vote())
    waiting=dict(patch=dict(proposal(),id='waiting'),state='pending_independent_audit')
    base=bundle.seed();candidate,approved=final_bundle(base,[good,waiting])
    assert not approved and candidate['tasks']==base['tasks']
