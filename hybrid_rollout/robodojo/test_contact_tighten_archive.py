"""Stable v2 record contracts: variable commands and immutable model provenance."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from .force_hold_archive import validate_hold_mapping, resolve_base_model_targets


def fixture(tmp_path, *, opening=.4, counts=(1,)):
    base=np.zeros(14,np.float32);base[6]=opening;base[13]=1
    raw=np.tile(base,(50,1));raw[:,13]=1.2
    actions=raw.copy();actions[:,[6,13]]=np.clip(actions[:,[6,13]],0,1)
    path=tmp_path/'original.npz';np.savez(path,raw_actions=raw,actions=actions)
    fingers=[[.01,.01],[.04,.04]]
    receipts=[dict(step_id=1,source='student',student_prediction_id='old',proposal_action_index=0,
        executed_action=base.tolist(),obs=dict(states=base.tolist(),finger_positions_m=fingers))]
    samples=[];previous_norm=0;increases=0
    for offset,count in enumerate(counts):
        step=offset+2;command=base.copy();requested=count*.001
        command[6]=np.clip(float(base[6])-requested/.108,0,1)
        actual=float(base[6])-float(command[6]);increased=actual>previous_norm+1e-10
        previous_norm=actual;increases+=increased
        compensation={'left':dict(requested_cumulative_gap_reduction_m=requested,
            actual_cumulative_gap_reduction_m=actual*.108,requested_normalized_reduction=requested/.108,
            actual_normalized_reduction=actual,base_model_target=float(base[6]),issued_target=float(command[6]),increase_count=count)}
        before=receipts[-1]['obs']['states'];after=command.tolist()
        samples.append(dict(step_id=step,actual_command=after,compensation=compensation,force_window_step_id=step,
            pre_observation_step_id=step-1,post_observation_step_id=step,
            pre_observation_ref=f'observations/{step-1:06d}.npz',post_observation_ref=f'observations/{step:06d}.npz',
            pre_contact={'left':dict(step_id=step-1)},post_contact={'left':dict(step_id=step)},
            pre_internal_target_state=before,post_internal_target_state=after,
            pre_finger_positions_m=fingers,post_finger_positions_m=fingers))
        receipts.append(dict(step_id=step,source='student',student_prediction_id='new',proposal_action_index=None,
            trigger_proposal_action_index=0,hold_event_id='event',hold_base_action=base.tolist(),hold_compensation=compensation,
            hold_tightening_increased=bool(increased),executed_action=after,
            obs=dict(states=after,finger_positions_m=fingers)))
    event=dict(schema='pi_force_hold_event_v1',version='contact_tighten_v2',event_id='event',status='ended',end_reason='native_terminal',
        student_prediction_id='new',trigger_proposal_action_index=0,insert_before_proposal_action_index=0,insert_after_step_id=1,
        held_action=base.tolist(),base_held_action=base.tolist(),base_model_action=base.tolist(),
        base_prediction_id='old',base_proposal_action_index=0,base_model_step_id=1,
        raw_target_ref=dict(prediction_id='old',proposal_action_index=0,field='raw_actions',gripper_indices=[6,13]),
        base_clipped_model_target=dict(left=float(base[6]),right=1.),
        trigger_evidence={'left':dict(executed_model_step_ids=[1],executed_model_history=[float(base[6])])},
        closing_arms=['left'],settings=dict(gap_step_m=.001,max_gap_reduction_m=.005,max_extra_steps=8),single_finger_scale_m=.054,
        original_action_count=50,discarded_proposal_range=[0,50],steps=list(range(2,len(receipts)+1)),actual_count=len(samples),
        tighten_steps=int(increases),samples=samples)
    return path,receipts,event


def save_event(tmp_path,event):
    (tmp_path/'pi_force_hold_events.jsonl').write_text(json.dumps(event)+'\n')


def test_variable_commands_keep_old_base_and_reconstruct_raw_without_rewriting(tmp_path):
    path,receipts,event=fixture(tmp_path,counts=(1,2))
    before=hashlib.sha256(path.read_bytes()).hexdigest();save_event(tmp_path,event)
    _,events=validate_hold_mapping(tmp_path,receipts)
    resolved=resolve_base_model_targets(events,{'old':path},receipts)
    assert resolved[0]['raw_model_target']['right']>1
    assert resolved[0]['clipped_model_target']['right']==1
    assert resolved[0]['base_prediction_id']=='old' and event['student_prediction_id']=='new'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert receipts[1]['executed_action']!=receipts[2]['executed_action']


@pytest.mark.parametrize('corruption',['actual_command','compensation','pre_state','force_future','budget','increment_flag','tighten_count'])
def test_tighten_record_corruption_is_rejected(tmp_path,corruption):
    _,receipts,event=fixture(tmp_path)
    if corruption=='actual_command':
        event['samples'][0]['actual_command']=deepcopy(event['samples'][0]['actual_command'])
        event['samples'][0]['actual_command'][6]+=.01
    elif corruption=='compensation':event['samples'][0]['compensation']['left']['actual_cumulative_gap_reduction_m']=.005
    elif corruption=='pre_state':event['samples'][0]['pre_internal_target_state']=[.7]*14
    elif corruption=='force_future':event['samples'][0]['pre_contact']['left']['step_id']=2
    elif corruption=='budget':event['settings']['max_gap_reduction_m']=.006
    elif corruption=='increment_flag':receipts[1]['hold_tightening_increased']=False
    else:event['tighten_steps']=2
    save_event(tmp_path,event)
    with pytest.raises(ValueError):validate_hold_mapping(tmp_path,receipts)


@pytest.mark.parametrize('corruption',['base_prediction','raw_ref','aux_history','history_value'])
def test_original_model_provenance_rejects_auxiliary_or_wrong_prediction(tmp_path,corruption):
    path,receipts,event=fixture(tmp_path)
    if corruption=='base_prediction':event['base_prediction_id']='new'
    elif corruption=='raw_ref':event['raw_target_ref']['field']='actions'
    elif corruption=='aux_history':event['trigger_evidence']['left']['executed_model_step_ids']=[2]
    else:event['trigger_evidence']['left']['executed_model_history']=[.7]
    with pytest.raises(ValueError):resolve_base_model_targets({'event':event},{'old':path,'new':path},receipts)


def test_clipping_limits_actual_compensation_and_count(tmp_path):
    _,receipts,event=fixture(tmp_path,opening=.01,counts=(1,2,2))
    save_event(tmp_path,event);validate_hold_mapping(tmp_path,receipts)
    assert event['tighten_steps']==2
    assert event['samples'][-1]['compensation']['left']['requested_cumulative_gap_reduction_m']==.002
    assert event['samples'][-1]['compensation']['left']['actual_cumulative_gap_reduction_m']<.001081
    assert not receipts[-1]['hold_tightening_increased']


def test_base_reference_cannot_change_during_lifecycle(tmp_path):
    _,receipts,event=fixture(tmp_path);first=deepcopy(event);first.update(status='running',end_reason=None,steps=[],samples=[],actual_count=0)
    event['base_proposal_action_index']=1
    (tmp_path/'pi_force_hold_events.jsonl').write_text(json.dumps(first)+'\n'+json.dumps(event)+'\n')
    with pytest.raises(ValueError,match='immutable'):validate_hold_mapping(tmp_path,receipts)
