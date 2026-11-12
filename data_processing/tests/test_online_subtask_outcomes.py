from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from data_processing.tests.test_subtasks import episode
from data_processing.subtasks.io import read, write
from data_processing.subtasks.client import APIClient
from data_processing.subtasks.pipeline import load_config, load_input
from data_processing.online_subtask_outcomes.prepare import prepare, prepare_batch, model_payload
from data_processing.online_subtask_outcomes.contract import (collect_errors, validate_final, merge_patch,
    ValidationError, response_schema)
from data_processing.online_subtask_outcomes.pipeline import annotate_one, protocol, run_batch, validate_run
from hybrid_rollout.robodojo.skill_auto_infer.subtask_instruction import render_instruction, format_prompt


@pytest.fixture
def online(episode):
    task = 'Move the object.'
    write(episode / 'sim/reset.json', dict(episode_id='test-episode', instruction=task))
    history = read(episode / 'controller/history.json')
    sub = dict(skill_type='grasp', arm='left', object_ref='object', destination_ref=None,
               target_relation='none', target_orientation_ref=None, custom_description=None,
               giver_arm='unknown', receiver_arm='unknown')
    for row in history:
        d = row['decision']
        value = sub if d < 2 else None
        text = render_instruction(value)
        row['prediction_id'] = f'p{d}'
        row['conditioning'] = dict(episode_id='test-episode', prediction_id=f'p{d}',step_id=row['start_tick'],
            episode_instruction=task, subtask=value, subtask_text=text,
            effective_prompt=format_prompt(task,text),current_intent='Grasp the object with the left arm')
    write(episode / 'controller/history.json', history)
    return episode


def label(d, outcome='success'):
    return dict(decision_id=d, attempt_id=d, outcome=outcome, outcome_basis='direct',
        outcome_rationale='The observation establishes retention or loss during this attempt.',
        unknown_reason=None, outcome_evidence_refs=[f'd{1 if d==0 else 3:04d}.assessment.execution_evidence'],
        instruction_action_alignment='aligned' if d==0 else 'not_applicable',
        alignment_rationale='The operation matches the goal; null input remains null.',
        alignment_evidence_refs=[], review_reasons=[])


def candidate():
    return dict(labels=[label(0,'failure'),label(2)])


def audit():
    return dict(reviews=[dict(decision_id=d,verdict='accept',explanation='Attribution checked against feedback') for d in (0,2)])


def test_preserve_inputs_holds_null_and_zero(online):
    history=read(online/'controller/history.json')
    history[0].update(force_hold_steps=3,tighten_steps=2,force_hold_events=[dict(closing_arms=['left'],steps=[2,3,4])])
    write(online/'controller/history.json',history)
    prepared=prepare(online)
    assert [d['conditioning'] for d in prepared['payload']['decisions']]==[r['conditioning'] for r in history]
    payload=model_payload(prepared)
    assert payload['required_decision_ids']==[0,2]
    assert 'd0000.automatic_grasp_assistance.hold_intervals' in payload['decisions'][0]['after']
    ann=validate_final(prepared,candidate(),audit())
    a,b=ann['segments']
    assert (a['start_step'],a['end_step'],b['start_step'],b['end_step'])==(0,4,4,10)
    assert not a['training_eligible'] and b['training_eligible']
    assert b['online_conditioning']['subtask'] is None and b['instruction']['text'] is None
    assert not b['training_eligibility']['high_level']
    assert a['outcome_evidence'][0]['value']=='5 slipped during lift'
    assert len(ann['events'])==3


@pytest.mark.parametrize('mutate',[
    lambda h:h[0].pop('conditioning'),
    lambda h:h[0]['conditioning'].update(prediction_id='wrong'),
    lambda h:h[0]['conditioning'].update(step_id=99),
    lambda h:h[2]['conditioning'].update(subtask_text='invented null'),
    lambda h:h[0]['conditioning'].update(effective_prompt='different input'),
])
def test_input_integrity(online,mutate):
    h=read(online/'controller/history.json');mutate(h);write(online/'controller/history.json',h)
    with pytest.raises(ValueError):prepare(online)


def test_malformed_input_preserved_for_review(online):
    h=read(online/'controller/history.json');h[0]['conditioning']['subtask']['skill_type']='made_up'
    write(online/'controller/history.json',h)
    p=prepare(online);c=candidate();c['labels'][0]['outcome']='success'
    a=validate_final(p,c,audit())['segments'][0]
    assert a['online_conditioning']['subtask']['skill_type']=='made_up'
    assert a['needs_review'] and not a['training_eligible'] and 'instruction_format' in a['review_reasons']


@pytest.mark.parametrize('mutate,code',[
    (lambda c:c['labels'].pop(),'coverage'),
    (lambda c:c['labels'].append(label(1)),'coverage'),
    (lambda c:c['labels'][0].update(start_step=1),'schema'),
    (lambda c:c['labels'][0].update(subtask_text='rewritten'),'schema'),
    (lambda c:c['labels'][0].update(attempt_id=2),'attempt'),
    (lambda c:c['labels'][0].update(outcome_evidence_refs=['not-real']),'reference'),
    (lambda c:c['labels'][0].update(outcome_evidence_refs=['d0000.assessment.execution_evidence']),'outcome_observation'),
    (lambda c:c['labels'][1].update(instruction_action_alignment='aligned'),'alignment'),
    (lambda c:c['labels'][0].update(outcome='unknown'),'unknown'),
])
def test_validator(online,mutate,code):
    c=candidate();mutate(c)
    assert code in {e['code'] for e in collect_errors(prepare(online),c)}


def test_later_inference_allowed_mismatch_excluded(online):
    p=prepare(online);c=candidate()
    c['labels'][0].update(outcome_basis='inferred',outcome_evidence_refs=['d0003.assessment.execution_evidence'],
                         instruction_action_alignment='mismatch',alignment_evidence_refs=['d0000.assessment.current_subgoal'])
    ann=validate_final(p,c,audit())
    assert ann['segments'][0]['needs_review'] and not ann['segments'][0]['training_eligible']
    c['labels'][1].update(outcome='unknown',outcome_basis='unresolved',unknown_reason='No retained-state observation',outcome_evidence_refs=[])
    assert validate_final(p,c,audit())['segments'][1]['needs_review']


def test_patch_scope_and_empty_outputs(online):
    with pytest.raises(ValidationError):merge_patch(candidate(),dict(labels=[label(0)]),[2])
    assert merge_patch(candidate(),dict(labels=[]),[0,2])==candidate()
    assert collect_errors(prepare(online),dict(labels=[]))[0]['code']=='coverage'
    with pytest.raises(ValidationError):validate_final(prepare(online),candidate(),dict(reviews=[]))


class Fake:
    def __init__(self,values):self.values=iter(values);self.calls=[]
    def complete(self,prompt,payload,correction=None):
        self.calls.append(deepcopy(correction))
        value=next(self.values)
        if isinstance(value,Exception):raise value
        return dict(text=json.dumps(value))


def test_repair_audit_resume_no_overwrite(online,tmp_path):
    p=prepare(online);c=candidate();bad=deepcopy(c);bad['labels'][0]['outcome_evidence_refs']=['bad']
    client=Fake([bad,dict(labels=[c['labels'][0]]),dict(labels=[],**audit())])
    dest=tmp_path/'labels';cfg=load_config()
    ann,state=annotate_one(p,dest,cfg,protocol(),client)
    assert state=='annotated' and len(client.calls)==3
    assert client.calls[1]['mode']=='patch' and client.calls[1]['editable_decision_ids']==[0]
    assert client.calls[2]['mode']=='audit'
    assert annotate_one(p,dest,cfg,protocol(),client)==(ann,'reused')
    ann['segments'][0]['outcome']='success';write(dest/'annotation.json',ann)
    with pytest.raises(ValueError,match='corrupted'):annotate_one(p,dest,cfg,protocol(),client)


def test_coverage_repair_and_audit_rejection(online,tmp_path):
    c=candidate();broken=dict(labels=[],reviews=[])
    client=Fake([dict(labels=[]),c,broken,dict(labels=[],**audit())])
    ann,_=annotate_one(prepare(online),tmp_path/'labels',load_config(),protocol(),client)
    assert client.calls[1]['mode']=='initial'
    assert client.calls[3]['mode']=='audit' and client.calls[3]['errors']
    assert len(ann['segments'])==2


def test_failure_retry_preserves_previous_invocations(online,tmp_path):
    p=prepare(online);dest=tmp_path/'labels';cfg=load_config()
    with pytest.raises(ValueError):annotate_one(p,dest,cfg,protocol(),Fake([ValueError('Network failed')]))
    ann,_=annotate_one(p,dest,cfg,protocol(),Fake([candidate(),dict(labels=[],**audit())]))
    assert len(list((dest/'attempts').iterdir()))==2
    assert (dest/'attempts/attempt_0001/request_00.json').is_file()
    assert ann['semantic_audit']['status']=='completed'


def test_schema_injected_in_shared_api(monkeypatch,online):
    monkeypatch.setenv('SUBTASK_API_KEY','test')
    def handler(request):
        body=json.loads(request.content)
        schema=body['text']['format']['schema']
        assert list(schema['properties'])==['labels']
        assert 'start_step' not in schema['properties']['labels']['items']['properties']
        return httpx.Response(200,json={})
    client=APIClient(load_config(),httpx.MockTransport(handler),schema_factory=response_schema)
    client.complete('prompt',model_payload(prepare(online)))


def test_prepare_append_and_batch_reuse_without_accounts(online,tmp_path,monkeypatch):
    import data_processing.online_subtask_outcomes.pipeline as pipe
    batch=tmp_path/'batch';m=prepare_batch([online],batch)
    assert prepare_batch([online],batch,append=True)==m
    p=load_input(batch,m['episodes'][0]);assert p['payload']['version'].startswith('online_')
    cfg=load_config();cfg.update(backend='api')
    # Fake ignores backend parsing by returning a Responses envelope.
    class Client:
        def __init__(self,*a,**kw):pass
        def complete(self,prompt,payload,correction=None):
            value=dict(labels=[],**audit()) if correction else candidate()
            return dict(status='completed',output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(value))])])
    monkeypatch.setattr(pipe,'APIClient',Client)
    result=run_batch(batch,cfg)
    assert result['failures']==0
    assert Path(result['named_run_root']).is_dir()
    monkeypatch.setattr(pipe,'APIClient',lambda *a,**k:pytest.fail('Must reuse without model calls'))
    assert run_batch(batch,cfg)['results'][0]['state']=='reused'
    assert validate_run(batch)['failures']==0
    assert (batch/'latest_run.json').exists()


def test_manual_interruption_is_not_success_evidence(online):
    summary = read(online/'sim/summary.json')
    summary.update(reason='controller_error', manual_truncation=dict(retained_steps=10))
    write(online/'sim/summary.json', summary)
    p = prepare(online)
    last_d = p['payload']['decisions'][-1]['decision_id']
    ref = f'd{last_d:04d}.archive.manual_truncation'
    assert ref in model_payload(p)['decisions'][-1]['after']
    c = candidate()
    c['labels'][-1].update(outcome='interrupted', outcome_evidence_refs=[ref])
    assert not collect_errors(p, c)
    for outcome in ('success', 'failure'):
        c['labels'][-1]['outcome'] = outcome
        assert 'outcome_observation' in {e['code'] for e in collect_errors(p, c)}
    c['labels'][0].update(outcome='interrupted', outcome_evidence_refs=[ref])
    assert any(e['decision_id'] == 0 and e['code'] == 'outcome_observation' for e in collect_errors(p, c))


def test_manual_interruption_endpoint_must_match(online):
    summary = read(online/'sim/summary.json')
    summary.update(reason='controller_error', manual_truncation=dict(retained_steps=9))
    write(online/'sim/summary.json', summary)
    with pytest.raises(ValueError, match='truncation endpoint'):
        prepare(online)


def test_wire_schema_compatibility_and_nonempty_contract(online):
    p=model_payload(prepare(online))
    schema=response_schema(None,p)
    assert 'uniqueItems' not in json.dumps(schema)
    assert schema['properties']['labels']['minItems']==1
    assert 'uniqueItems' not in json.dumps(response_schema(dict(mode='audit'),p))
