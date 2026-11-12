"""Offline alignment and corruption checks for inserted force-hold controls."""
import json

import numpy as np
import pytest

from .force_hold_archive import (json_diagnostics, observation_diagnostics,
                                 validate_hold_mapping, validate_proposal_mapping)
from .tools.test_extract_episode import fixture
from .tools.extract_episode import extract


def event_fixture(tmp_path, count=2, trigger=1):
    command = np.full(14, .1, np.float32).tolist()
    original = [dict(step_id=i+1, proposal_action_index=i,
                     student_prediction_id='p', executed_action=command) for i in range(trigger)]
    held = [dict(step_id=trigger+i+1, proposal_action_index=None,
                 trigger_proposal_action_index=trigger, hold_event_id='e',
                 student_prediction_id='p', executed_action=command) for i in range(count)]
    event = dict(schema='pi_force_hold_event_v1', event_id='e', status='ended',
                 end_reason='native_terminal', student_prediction_id='p',
                 trigger_proposal_action_index=trigger, insert_before_proposal_action_index=trigger,
                 insert_after_step_id=trigger, held_action=command,
                 steps=[row['step_id'] for row in held], actual_count=count,
                 discarded_proposal_range=[trigger, 50], original_action_count=50)
    path = tmp_path/'pi_force_hold_events.jsonl'
    path.write_text(json.dumps(dict(event, status='running', end_reason=None, steps=[], actual_count=0))+'\n'+json.dumps(event)+'\n')
    return original+held, event, path


@pytest.mark.parametrize('count,trigger', [(2,1),(0,1),(1,0),(0,0)])
def test_event_insertion_and_zero_ack(tmp_path, count, trigger):
    receipts, event, _ = event_fixture(tmp_path, count, trigger)
    rows, sealed = validate_hold_mapping(tmp_path, receipts)
    assert len(rows)==2 and sealed['e']==event
    validate_proposal_mapping(receipts, np.full((50,14),.1,np.float32))


@pytest.mark.parametrize('corruption', ['hold_index','hold_command','event_count','discard','unsealed','prefix','future_index'])
def test_mapping_corruption_rejected(tmp_path, corruption):
    receipts, event, path = event_fixture(tmp_path)
    if corruption=='hold_index': receipts[1]['proposal_action_index']=1
    elif corruption=='hold_command': receipts[1]['executed_action']=[.2]*14
    elif corruption=='event_count': event['actual_count']=9
    elif corruption=='discard': event['discarded_proposal_range']=[2,50]
    elif corruption=='unsealed': event.update(status='running',end_reason=None)
    elif corruption=='prefix': receipts[0]['proposal_action_index']=2
    else: receipts[0]['proposal_action_index']=50
    path.write_text(json.dumps(event)+'\n')
    with pytest.raises(ValueError):
        validate_hold_mapping(tmp_path, receipts)


def test_unknown_force_is_nan_null_and_validity_not_zero():
    values=observation_diagnostics({},0)
    assert np.isnan(values['force_world_mean_N']).all()
    assert not values['force_valid_mask'].any()
    encoded=json.dumps(json_diagnostics(values),allow_nan=False)
    assert 'null' in encoded
    with pytest.raises(ValueError,match='Reset observation'):
        observation_diagnostics(dict(gripper_contact_valid_mask=np.ones((2,2),bool),gripper_contact_step_id=0),0)


def test_force_window_timing_and_finger_feedback():
    obs=dict(finger_positions_available=True,finger_positions_step_id=2,
             finger_positions_m=np.full((2,2),.01),gripper_contact_step_id=2,
             gripper_contact_valid_mask=[[True,False],[False,False]],
             gripper_contact_left_link7_force_world_mean_N=[1,2,3],
             gripper_contact_left_link7_force_magnitude_peak_N=4,
             gripper_contact_left_link7_contact_ratio=.5)
    values=observation_diagnostics(obs,2)
    assert values['finger_positions_valid'] and values['force_valid_mask'][0,0]
    np.testing.assert_array_equal(values['force_world_mean_N'][0,0],[1,2,3])
    assert np.isnan(values['force_magnitude_peak_N'][0,1])
    with pytest.raises(ValueError,match='Misaligned'):
        observation_diagnostics(obs,1)


def test_original_action_must_match_immutable_proposal():
    receipt=dict(proposal_action_index=1,executed_action=[.2]*14)
    with pytest.raises(ValueError,match='immutable proposal'):
        validate_proposal_mapping([receipt],np.full((50,14),.1,np.float32))


def test_old_archive_extract_preserves_t_plus_one_unknowns(tmp_path):
    raw=fixture(tmp_path)
    report=extract(raw,tmp_path/'out')
    assert report['status']=='passed'
    with np.load(tmp_path/'out/steps.npz') as steps:
        assert steps['observation_force_world_mean_N'].shape==(3,2,2,3)
        assert np.isnan(steps['observation_force_world_mean_N']).all()
        assert not steps['observation_force_valid_mask'].any()
    assert json.loads((tmp_path/'out/force_hold_events.json').read_text())==[]


def test_legacy_pi_only_history_extract(tmp_path):
    raw=fixture(tmp_path)
    (raw/'controller/history.json').write_text(json.dumps([
        dict(prediction_id='p1',end_step=1,executed_steps=1),
        dict(prediction_id='p2',end_step=2,executed_steps=1)]))
    for path in (raw/'controller').glob('execution_*.npz'):
        path.unlink()
    report=extract(raw,tmp_path/'out')
    assert report['steps']==2 and report['decisions']==2


def test_multi_chunk_hold_mapping_keeps_independent_original_indices(tmp_path):
    receipts, first, path=event_fixture(tmp_path,count=1,trigger=1)
    command=receipts[0]['executed_action']
    second=dict(first,event_id='e2',student_prediction_id='p2',
                trigger_proposal_action_index=0,insert_before_proposal_action_index=0,
                insert_after_step_id=2,steps=[3],actual_count=1,discarded_proposal_range=[0,50])
    receipts.append(dict(step_id=3,proposal_action_index=None,
                         trigger_proposal_action_index=0,hold_event_id='e2',
                         student_prediction_id='p2',executed_action=command))
    receipts.append(dict(step_id=4,proposal_action_index=0,student_prediction_id='p3',executed_action=command))
    path.write_text(json.dumps(first)+'\n'+json.dumps(second)+'\n')
    _,events=validate_hold_mapping(tmp_path,receipts)
    assert set(events)=={'e','e2'}
    assert [row['proposal_action_index'] for row in receipts]==[0,None,None,0]


def test_client_actual_ack_counts_do_not_consume_original_prediction(tmp_path):
    from .robodojo_server.client import RoboDojoTools
    from .test_contract import Sim, Student, invoke, response

    class ForceSim(Sim):
        def request(self, op, **args):
            if op == 'chunk_step':
                assert args['original_action_count'] == 50
                assert len(args['actions']) == 2
                command = args['actions'][0]
                rows = []
                for t in range(3):
                    self.tick += 1
                    rows.append(dict(valid=True, executed_action=command,
                        obs=dict(states=np.zeros(14)), step_id=self.tick,
                        proposal_action_index=0 if t == 0 else None,
                        trigger_proposal_action_index=None if t == 0 else 1,
                        hold_event_id=None if t == 0 else 'h',
                        terminated=t == 2, truncated=False, success=False))
                return dict(steps=rows,step_id=self.tick,proposal_steps_consumed=1,
                    force_hold_steps=2,force_hold_events=[dict(event_id='h')],reinfer_required=True)
            result = super().request(op, **args)
            if op == 'metadata':
                result.update(max_episode_steps=3,pi_force_hold=dict(enabled=True))
            if op == 'reset':
                result['metadata']=dict(pi_force_hold=dict(enabled=True,allowed_object_paths_resolved=['/obj']))
            return result

    tools=RoboDojoTools(tmp_path,'arrange_largest_number',Student(),rpc_factory=ForceSim)
    invoke(tools);invoke(tools)
    before=tools.proposal_path.read_bytes()
    result=invoke(tools,response=response(tools.request))
    assert tools.proposal_path.read_bytes()==before
    assert tools.counters['student_steps']==3
    assert tools.counters['student_proposal_steps']==1
    assert tools.counters['force_hold_steps']==2
    assert tools.history[0]['discarded_student_steps']==49
    assert tools.history[0]['selected_student_steps']==2
    assert result['result']['force_hold_intervened'] is True
    assert json.loads((tmp_path/'run.json').read_text())['pi_force_hold']['allowed_object_paths_resolved']==['/obj']
    with np.load(tmp_path/'execution_000.npz') as execution:
        assert execution['actions'].shape==(3,14)
        assert execution['proposal_action_index'].tolist()==[0,-1,-1]


@pytest.mark.parametrize('corruption',['before_step','before_force_future','before_ref','trigger_action'])
def test_trigger_before_evidence_uses_only_current_observation(tmp_path,corruption):
    receipts,event,path=event_fixture(tmp_path)
    event.update(trigger_before=dict(step_id=1,force_window_step_id=1),
                 trigger_before_observation='observations/000001.npz',
                 trigger_action=[.1]*14)
    if corruption=='before_step':event['trigger_before']['step_id']=2
    elif corruption=='before_force_future':event['trigger_before']['force_window_step_id']=2
    elif corruption=='before_ref':event['trigger_before_observation']='observations/000002.npz'
    else:event['trigger_action']=[.2]*14
    path.write_text(json.dumps(event)+'\n')
    with pytest.raises(ValueError):
        validate_hold_mapping(tmp_path,receipts)
        validate_proposal_mapping(receipts,np.full((50,14),.1,np.float32),{'e':event},'p')


def test_v3_client_checks_actual_checkpoint_before_reset(tmp_path, monkeypatch):
    from dataclasses import asdict
    from .robodojo_server.client import RoboDojoTools
    from .robodojo_server.pi_force_hold import ForceHoldConfig, BOOTSTRAP_CHECKPOINT_SHA256
    from .test_contract import Sim, Student, invoke
    config = ForceHoldConfig(tighten_enabled=True, flywheel_round=1,
        controller_route='auto_infer_v1', checkpoint_sha256=BOOTSTRAP_CHECKPOINT_SHA256)
    class MarginSim(Sim):
        def request(self, op, **args):
            assert op != 'reset', 'Unapproved checkpoint must fail before simulator reset'
            result=super().request(op,**args)
            if op=='metadata':
                result['pi_force_hold']=dict(enabled=True,version='contact_margin_v3',config=asdict(config))
            return result
    monkeypatch.setenv('ROLLOUT_CONTROLLER_ROUTE','auto_infer_v1')
    tools=RoboDojoTools(tmp_path,'arrange_largest_number',Student(),rpc_factory=MarginSim)
    with pytest.raises(ValueError,match='Tightening checkpoint differs from the pinned checkpoint'):
        invoke(tools)
