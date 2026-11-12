"""One continuing lead analyst + durable evidence + one independent final review thread."""
from copy import deepcopy
import difflib
import fcntl
import json
from pathlib import Path
import time

from ..io import write_json
from .bundle import apply_patch,word_count,seal,digest
from .data import read
from .evidence import Evidence
from .session import Conversation,utc,usage_summary
from . import conversation_schema as schema

HERE=Path(__file__).parent


class ProposalPrecheckError(ValueError):
    def __init__(self,reviews):
        self.errors=[dict(patch_id=r['patch']['id'],scope=r['scope'],rule_id=r['patch']['rule_id'],
                          error=r['reason']) for r in reviews if r['state']!='pending_independent_audit']
        super().__init__('Proposal precheck failed; repair and resubmit the COMPLETE patch set in this lead conversation: '
                         +json.dumps(self.errors,ensure_ascii=False))


def audit_contract(state):
    return dict(expected_patch_ids=[r['patch']['id'] for r in state.get('reviews',[])
                                   if r['state']=='pending_independent_audit'],
        excluded_prechecks=[dict(patch_id=r['patch']['id'],reason=r.get('reason'),state=r['state'])
                            for r in state.get('reviews',[]) if r['state']=='candidate'],
        instruction='Return exactly one verdict for each expected_patch_id only. Historical proposals and failed prechecks are NOT pending audits. If the expected list is empty, return verdicts=[].')


def brief_bundle(base):
    return {k:base[k] for k in ('shared_prefix','shared_rules','tasks')}


def compact_issues(issues):
    return [{k:i[k] for k in ('ref','episode_key','task','group_id','category','recovery',
        'mechanism','verified','evidence_decisions','counterexample_decisions','start_decision','end_decision')}
        for i in issues.values()]


def checkpoint(state,index,store,base):
    catalog=[{k:x[k] for k in ('case_key','task','case_id','first_decision','last_decision')} for x in index]
    table=[] if state.get('reuse_analysis') else store.problem_table(state['issues'])
    groups=[{k:g[k] for k in ('group_id','distinct_cases','tasks','observed_case_outcomes')} for g in table]
    return dict(phase=state['phase'],memory=state['memory'],case_catalog=catalog,
        issue_count=len(state['issues']),group_summary=groups,
        coverage=store.coverage(state['knowledge']),
        guidance=brief_bundle(base),proposal_limits={k:store.cfg[k] for k in
            ('max_patches_per_scope','max_rule_words','max_growth_words','max_shared_words','max_task_words')},
        proposal_state=[dict(id=r['patch']['id'],scope=r['scope'],state=r['state']) for r in state.get('reviews',[])],
        audit_contract=audit_contract(state) if state['phase']=='auditing' else None,
        evidence_note='Full evidence and dossiers remain in host storage; request case ranges to inspect them again.')


def query_pages(queries,index,state,store,cfg,*,max_chars=None):
    if len(queries)>2:raise ValueError('At most two index queries per turn')
    output=[];size=0
    for q in queries:
        if q['offset']<0 or not 1<=q['limit']<=20 or not q['question'].strip():
            raise ValueError('Invalid query pagination/question')
        if q['kind']=='cited_issues':
            reviews=state.get('reviews') if state['phase']=='auditing' else state.get('prior_reviews',[])
            if state['phase']=='auditing':
                reviews=[r for r in reviews or [] if r['state']=='pending_independent_audit']
            refs=list(dict.fromkeys(ref for r in reviews or []
                for field in ('evidence_refs','success_refs','counterexample_refs') for ref in r['patch'][field]))
            values=[state['issues'][ref] for ref in refs]
        elif q['kind']=='groups':
            values=[] if state.get('reuse_analysis') else store.problem_table(state['issues'])
        elif q['kind']=='patches' and state.get('phase')=='auditing':
            values=[r for r in state.get('reviews',[]) if r['state']=='pending_independent_audit']
        else:
            values={'cases':index,'issues':list(state['issues'].values()),
                    'patches':state.get('reviews',[]),'prior_patches':state.get('prior_reviews',[])}[q['kind']]
        page=values[q['offset']:q['offset']+q['limit']]
        item=dict(kind=q['kind'],offset=q['offset'],total=len(values),entries=page,
                  next_offset=q['offset']+len(page) if q['offset']+len(page)<len(values) else None)
        if state.get('phase')=='auditing':item['audit_contract']=audit_contract(state)
        size+=len(json.dumps(item,ensure_ascii=False))
        if size>(max_chars if max_chars is not None else cfg['max_input_chars']//4):
            raise ValueError('Index query too large; request a smaller page')
        output.append(item)
        if q['kind'] in ('issues','cited_issues'):
            state['summary_seen']=sorted(set(state.get('summary_seen',[]))|{i['ref'] for i in page})
        if q['kind']=='prior_patches':
            state['prior_seen']=sorted(set(state.get('prior_seen',[]))|set(range(q['offset'],q['offset']+len(page))))
    return output


def proposals(base,state,cfg):
    from .validation import support_check
    candidate=base;reviews=[];counts={};ids=set()
    for p in state['proposed']:
        outcome=dict(scope=p['scope'],patch=p,state='candidate')
        try:
            if p['id'] in ids:raise ValueError('Duplicate patch ID')
            ids.add(p['id']);scope=p['scope']
            counts[scope]=counts.get(scope,0)+1
            if counts[scope]>cfg['max_patches_per_scope']:raise ValueError('Per-scope patch budget exceeded')
            support_check(p,state['issues'],cfg)
            changed=apply_patch(candidate,p,cfg,original=base)
            if word_count(changed,scope)>word_count(base,scope)+cfg['max_growth_words']:
                raise ValueError('Per-round strategy growth exceeded')
            if word_count(changed,scope)>cfg['max_shared_words' if scope=='shared' else 'max_task_words']:
                raise ValueError('Absolute strategy length exceeded')
            candidate=changed;outcome['state']='pending_independent_audit'
        except (ValueError,KeyError) as exc:outcome['reason']=str(exc)
        reviews.append(outcome)
    return candidate,reviews


def validate_finish(state,store):
    coverage=store.coverage(state['knowledge'])['by_task']
    # Coverage is not incidence: require cross-task work and a normal success comparison,
    # but allow selective inspection rather than force re-reading every episode.
    for task,c in coverage.items():
        if c['read_cases']<min(2,c['total_cases']):
            raise ValueError('Before finishing inspect at least two distinct cases per task: '+task)
        success_keys=[k for k,r in store.rows.items() if r['task']==task and r['result'].get('success') is True]
        if success_keys and not any(state['knowledge']['read'].get(k) for k in success_keys):
            raise ValueError('Inspect at least one successful comparison case for task: '+task)


def validate_audit(value,state,store):
    expected={r['patch']['id']:r['patch'] for r in state['reviews'] if r['state']=='pending_independent_audit'}
    votes=value['verdicts']
    if len(votes)!=len(expected) or {v['patch_id'] for v in votes}!=set(expected):
        received=[v['patch_id'] for v in votes]
        raise ValueError('Audit verdict IDs mismatch: '+json.dumps(dict(expected=sorted(expected),received=received,
            missing=sorted(set(expected)-set(received)),unexpected=sorted(set(received)-set(expected)),
            duplicates=sorted({p for p in received if received.count(p)>1})),ensure_ascii=False))
    for v in votes:
        if v['verdict']!='approve':continue
        if not all(x is True for k,x in v.items() if k not in ('patch_id','verdict','reason')):
            raise ValueError('Approval requires all audit checks')
        p=expected[v['patch_id']]
        for ref in p['evidence_refs']+p['success_refs']+p['counterexample_refs']:
            issue=state['issues'][ref];key=issue['episode_key']
            needed=set(issue['evidence_decisions']+issue['counterexample_decisions'])
            if not needed<=set(state['audit_knowledge']['read'].get(key,[])):
                raise ValueError('Independent reviewer must read original cited decisions before approval: '+ref)
            if not set(issue['image_ids'])<=set(state['audit_knowledge']['images'].get(key,[])):
                raise ValueError('Independent reviewer must inspect cited images before approval: '+ref)


def final_bundle(base,reviews):
    # Patches are only published as the SAME whole bundle that the auditor saw. Partial
    # acceptance can change dependencies (delete+replacement etc.), so keep it candidate-only.
    pending=[r for r in reviews if 'audit' in r]
    accepted=bool(pending) and all(r['state']=='approved' for r in pending) and not any(
        r['state']=='pending_independent_audit' for r in reviews)
    result=base
    if accepted:
        for r in pending:result=apply_patch(result,r['patch'],{'max_rule_words':10**9},original=base)
    return seal(dict(result,parent_sha256=base['sha256'])),accepted


def run_conversation(suite,cfg,opts,manifest,base,identity,root,*,prepare_only=False,backend=None,reuse_analysis=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=dict(suite=str(suite),**identity)
        if (root/'manifest.json').exists() and read(root/'manifest.json')!=frozen:
            raise ValueError('Output directory belongs to different inputs/configuration')
        write_json(root/'manifest.json',frozen);write_json(root/'base_bundle.json',base)
        for name in ('proposal_attempts','audit_protocol_errors'):(root/name).mkdir(exist_ok=True)
        status_path=root/'status.json';state_path=root/'checkpoint.json'
        if status_path.exists() and read(status_path)['state'] in ('published','candidate_only','no_change'):
            return read(status_path)
        if manifest['missing_cases'] or manifest['anomalies']:
            status=dict(state='waiting_for_collection',missing_cases=manifest['missing_cases'],anomalies=manifest['anomalies'])
            write_json(status_path,status);return status
        if not manifest['episodes']:raise ValueError('No collection episodes')
        store=Evidence(root/'evidence',manifest['episodes'],opts)
        index=store.index();write_json(root/'case_index.json',index)
        # The first turn sees every case's outcome and burden. Checkpoints use a compact
        # catalog; full rows/dossiers are paged back in after compaction, never truncated.
        index=[{k:v for k,v in x.items() if k not in ('instruction','last_assessment_excerpt','excerpt_is_not_local_outcome')} for x in index]
        existing_state=state_path.exists()
        state=read(state_path) if existing_state else dict(phase='analyzing',turn=0,audit_turn=0,
            memory='',audit_memory='',issues={},knowledge=dict(read={},images={}),audit_knowledge=dict(read={},images={}),
            next_payload=dict(message='Compare this collection using this full case index; request evidence as needed.',case_index=index),
            next_images=[],proposed=[],reviews=[],deferred_issues=[],started_at=utc(),validation_errors=0)
        if reuse_analysis is not None and not existing_state:
            if not identity.get('experiment_only') or opts['publish']!='candidate_only':
                raise ValueError('Summary replay is restricted to isolated candidate-only experiments')
            state.update(reuse_analysis=True,issues=deepcopy(reuse_analysis['issues']),
                prior_reviews=deepcopy(reuse_analysis['reviews']),summary_seen=[])
            state['next_payload']=dict(message='Reuse the saved analysis. Reconsider and consolidate proposals only; do not re-summarize episodes.',
                summary_count=len(state['issues']),prior_patch_count=len(state['prior_reviews']),
                prior_patch_catalog=[dict(index=n,id=r['patch']['id'],scope=r['scope'],
                    rule_id=r['patch']['rule_id'],previous_reason=r.get('reason')) for n,r in enumerate(state['prior_reviews'])],
                summary_catalog=[dict(offset=n,ref=i['ref'],task=i['task'],category=i['category'])
                    for n,i in enumerate(state['issues'].values())])
        write_json(state_path,state)
        # A large summary catalog is paged, never pasted into the first model turn.
        if state.get('reuse_analysis') and not existing_state:
            write_json(root/'summary_catalog.json',state['next_payload'].pop('summary_catalog'))
            write_json(state_path,state)
        lead_prompt='reproposal.md' if state.get('reuse_analysis') else 'conversation.md'
        initial_size=len((HERE/lead_prompt).read_text())+len(json.dumps(dict(
            incremental_evidence=state['next_payload'],
            durable_checkpoint=checkpoint(state,index,store,base)),ensure_ascii=False))
        if state['turn']==0 and initial_size>opts['max_input_chars']:
            raise ValueError('Initial overview exceeds input budget; reduce index detail before starting')
        if prepare_only:
            status=dict(state='prepared',episodes=len(index),output=str(root),workflow='conversation_v2',
                        initial_input_chars=initial_size)
            write_json(status_path,status);return status
        lead=Conversation(root/'lead',opts,backend=backend)
        reviewer=Conversation(root/'reviewer',opts,backend=backend)
        invocation_start=time.monotonic()
        try:
            while state['phase'] in ('analyzing','auditing'):
                auditing=state['phase']=='auditing';turn_key='audit_turn' if auditing else 'turn';n=state[turn_key]
                if n>=opts['conversation_max_turns']:
                    raise ValueError('Conversation turn budget reached; checkpoint preserved')
                memory_key='audit_memory' if auditing else 'memory'
                status=dict(state=state['phase'],output=str(root),started_at=state['started_at'],updated_at=utc(),
                    lead_turns=state['turn'],audit_turns=state['audit_turn'],issues=len(state['issues']),
                    coverage=store.coverage(state['knowledge']))
                write_json(status_path,status)
                print(json.dumps(dict(event='prompt_conversation_progress',**status)),flush=True)
                anchor=checkpoint(state,index,store,base)
                if auditing:
                    anchor['memory']=state['audit_memory'];anchor['coverage']=store.coverage(state['audit_knowledge'])
                    # Same whole set each time reconstruction is necessary, not the analyst transcript.
                    anchor['candidate_guidance']=brief_bundle(read(root/'draft_bundle.json'))
                client=reviewer if auditing else lead
                system=(HERE/('conversation_audit.md' if auditing else lead_prompt)).read_text()
                response=client.ask(n,system,state['next_payload'],schema.REVIEW if auditing else schema.ANALYST,
                    anchor,images=state['next_images'])
                next_state=deepcopy(state);next_state[turn_key]+=1
                try:
                    if len(response['memory'])>opts['conversation_memory_chars']:
                        raise ValueError('Memory too long; preserve references, hypotheses and unresolved work concisely')
                    next_state[memory_key]=response['memory']
                    if not auditing:
                        if state.get('reuse_analysis') and response['updates']:
                            raise ValueError('Saved summaries are frozen in proposal replay; do not submit issue updates')
                        store.update(response['updates'],next_state)
                    if response['action']=='finish':
                        if response['reads'] or response['images'] or response['queries']:
                            raise ValueError('Finish cannot simultaneously request unseen evidence')
                        if auditing:
                            validate_audit(response,next_state,store)
                            votes={v['patch_id']:v for v in response['verdicts']}
                            for r in next_state['reviews']:
                                if r['state']=='pending_independent_audit':
                                    v=votes[r['patch']['id']];r['audit']=v
                                    r['state']='approved' if v['verdict']=='approve' else v['verdict']
                            next_state['phase']='complete'
                        else:
                            if state.get('reuse_analysis'):
                                if len(next_state.get('prior_seen',[]))!=len(next_state['prior_reviews']):
                                    raise ValueError('Read every prior candidate through prior_patches pages before final synthesis')
                                refs={ref for p in response['patches'] for field in
                                    ('evidence_refs','success_refs','counterexample_refs') for ref in p[field]}
                                if not refs<=set(next_state.get('summary_seen',[])):
                                    raise ValueError('Read the saved issue summaries cited by each proposal before finishing')
                            else:validate_finish(next_state,store)
                            if len(response['patches'])>(len(base['tasks'])+1)*opts['max_patches_per_scope']:
                                raise ValueError('Too many proposed patches')
                            next_state['proposed']=response['patches'];next_state['deferred_issues']=response['deferred_issues']
                            draft,reviews=proposals(base,next_state,opts);next_state['reviews']=reviews
                            # A format/evidence precheck is a repair request to the proposer,
                            # not an independent review and not a silent final exclusion.
                            write_json(root/'proposal_attempts'/f'{n:05d}.json',dict(
                                proposals=response['patches'],prechecks=reviews))
                            if any(r['state']!='pending_independent_audit' for r in reviews):
                                raise ProposalPrecheckError(reviews)
                            write_json(root/'draft_bundle.json',draft)
                            pending=[r for r in reviews if r['state']=='pending_independent_audit']
                            next_state['phase']='auditing'
                            refs={ref for r in pending for field in ('evidence_refs','success_refs','counterexample_refs') for ref in r['patch'][field]}
                            next_state['next_payload']=dict(message='Independently review the whole proposed bundle, including a no-change conclusion. Read original cited evidence.',
                                patch_catalog=[dict(id=r['patch']['id'],scope=r['scope'],state=r['state']) for r in reviews],
                                audit_contract=audit_contract(next_state),
                                dossier_count=len(refs),
                                evidence_access='Use cited_issues queries to page full cited dossiers and exact raw record/image locations. Do not paste all raw evidence at once.')
                        next_state['next_images']=[]
                    else:
                        if not auditing and response['patches']:raise ValueError('Proposals are accepted only at final synthesis')
                        if auditing and response['verdicts']:raise ValueError('Return the complete verdict set at finish')
                        knowledge=next_state['audit_knowledge' if auditing else 'knowledge']
                        # Reserve room for a full durable checkpoint even if native compaction
                        # requires one next turn. Deny oversize reads before consuming them.
                        room=opts['max_input_chars']-len(system)-len(json.dumps(anchor,ensure_ascii=False))-8192
                        if room<4096:raise ValueError('Durable checkpoint leaves insufficient incremental evidence space')
                        payload,images=store.fetch(response['reads'],response['images'],knowledge,root/'images',
                            text_budget=room*2//3)
                        payload.update(query_results=query_pages(response['queries'],index,next_state,store,opts,max_chars=room//3),
                            issue_count=len(next_state['issues']),
                            updated_issue_refs=[u['case_key']+':'+u['id'] for u in response.get('updates',[])],
                            coverage=store.coverage(knowledge),
                            reminder='Use queries(kind=issues/groups/cases, offset, limit, question) to recall saved evidence summaries and counts.')
                        next_state['next_payload']=payload;next_state['next_images']=[str(p) for p in images]
                    next_state['validation_errors']=0
                except (ValueError,KeyError) as exc:
                    next_state=deepcopy(state);next_state[turn_key]+=1;next_state['validation_errors']+=1
                    next_state['next_payload']=dict(validation_error=str(exc),instruction='Correct this in the SAME conversation; host state was not changed. Fetch missing evidence if necessary.')
                    if isinstance(exc,ProposalPrecheckError):
                        next_state['next_payload'].update(precheck_errors=exc.errors,
                            instruction='Repair the listed prechecks in this SAME lead conversation, or explicitly defer unsupported proposals. Resubmit the COMPLETE patch set (and any uncommitted issue updates). Do not start an audit or discard valid proposals silently.')
                    if auditing:
                        next_state['next_payload']['audit_contract']=audit_contract(state)
                        write_json(root/'audit_protocol_errors'/f'{n:05d}.json',dict(
                            response=response,error=str(exc),audit_contract=audit_contract(state)))
                    next_state['next_images']=[]
                state=next_state;write_json(state_path,state)
                if not state.get('reuse_analysis'):
                    write_json(root/'problem_table.json',store.problem_table(state['issues']))
                write_json(root/'dossiers.json',list(state['issues'].values()))
                write_json(root/'usage_summary.json',usage_summary(root))
                if state['validation_errors']>=opts['max_attempts']:
                    raise ValueError('Repeated protocol validation failures; see checkpoint next_payload')
            candidate,all_approved=final_bundle(base,state['reviews'])
            write_json(root/'candidate_bundle.json',candidate);write_json(root/'proposals.json',state['proposed'])
            write_json(root/'patch_reviews.json',state['reviews'])
            (root/'changes.diff').write_text(''.join(difflib.unified_diff(
                json.dumps(base,ensure_ascii=False,indent=2).splitlines(True),
                json.dumps(candidate,ensure_ascii=False,indent=2).splitlines(True),fromfile='base',tofile='candidate')))
            lines=['# Continuous prompt review','',f"Coverage: {json.dumps(store.coverage(state['knowledge']),ensure_ascii=False)}",
                'Selective evidence review does not measure population incidence or prove rollout improvement.',
                'A partially rejected bundle is not published; draft_bundle.json retains all proposed text.','']
            for r in state['reviews']:
                lines.extend([f"## {r['patch']['scope']} / {r['patch']['id']}: {r['state']}",r['patch']['text'],
                    r.get('reason',r.get('audit',{}).get('reason','')),''])
            lines+=['## Deferred',*state['deferred_issues']];(root/'report.md').write_text('\n'.join(lines))
            outcome='no_change' if not state['proposed'] else 'candidate_only'
            status=dict(state=outcome,output=str(root),workflow='conversation_v2',started_at=state['started_at'],ended_at=utc(),
                invocation_seconds=time.monotonic()-invocation_start,lead_turns=state['turn'],audit_turns=state['audit_turn'],
                approved_patches=sum(r['state']=='approved' for r in state['reviews']),whole_bundle_approved=all_approved,
                offline_evidence_only=True,coverage=store.coverage(state['knowledge']))
            if all_approved and opts['publish']=='auto_verified' and not identity.get('experiment_only'):
                path=root/'published_bundle.json';write_json(path,candidate)
                next_cfg=deepcopy(cfg);next_cfg.update(name=cfg['name']+'_prompt_'+candidate['sha256'][:8],
                    prompt_bundle=str(path),prompt_bundle_sha256=candidate['sha256'],prompt_evolution=opts)
                write_json(root/'next_collection_config.json',next_cfg)
                status.update(state='published',bundle=str(path),next_config=str(root/'next_collection_config.json'))
            write_json(status_path,status);return status
        except BaseException as exc:
            write_json(status_path,dict(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                output=str(root),started_at=state['started_at'],updated_at=utc(),phase=state['phase'],
                error=f'{type(exc).__name__}: {exc}',lead_turns=state['turn'],audit_turns=state['audit_turn']))
            raise
        finally:write_json(root/'usage_summary.json',usage_summary(root))
