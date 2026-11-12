import numpy as np
import pytest
from .action_downsampling import ActionDownsampler,SelectionConfig,configuration_dict,algorithm_version
from .test_action_downsampling import sample,toy_fk

def sampler():
    return ActionDownsampler(SelectionConfig(gripper_epsilon=.005,closing_hold_epsilon=.02),toy_fk)

def test_small_jitter_keeps_local_guard_without_resetting_long_hold():
    s=sampler();initial=sample()[0];a=sample();a[5:,6]=.592
    indices,r=s.select(a,initial)
    assert set(range(3,7))<=set(indices)
    assert r['closing_hold_events']==[] and len(indices)<15
    s.acknowledge(a,initial)
    assert s.hold_remaining.tolist()==[0,0]
    assert s.closing_event_anchor[0]==.6

def test_slow_close_accumulates_across_local_events_and_windows():
    s=sampler();initial=sample()[0]
    for opening in [.594,.588,.582]:
        a=initial.copy();a[6]=opening;s.acknowledge([a],initial)
        assert s.hold_remaining[0]==0
    a=initial.copy();a[6]=.576
    window=np.repeat(a[None],15,axis=0)
    indices,r=s.select(window,initial)
    assert indices==list(range(15)) and r['closing_hold_events'][0]['index']==0
    assert s.closing_event_anchor[0]==.6  # Selection is pure.
    s.acknowledge([a],initial)
    assert s.hold_remaining[0]==15
    a[6]=.570;s.acknowledge([a],initial)
    assert s.hold_remaining[0]==14  # Small subsequent close cannot refresh hold.

def test_opening_updates_anchor_but_does_not_start_hold():
    s=sampler();initial=sample()[0];a=initial.copy();a[13]=.7
    s.acknowledge([a],initial)
    assert s.hold_remaining[1]==0 and s.closing_event_anchor[1]==.7
    a[13]=.67;s.acknowledge([a],initial)
    assert s.hold_remaining[1]==15

def test_no_absolute_accumulation_and_no_unexecuted_commit():
    s=sampler();initial=sample()[0];a=sample();a[:,6]=np.resize([.594,.606],15)
    s.acknowledge(a,initial)
    assert s.hold_remaining[0]==0 and s.closing_event_anchor[0]==.6
    a[-1,6]=.5;s.select(a,initial);s.acknowledge(a[:1],initial);s.acknowledge([],initial)
    assert s.hold_remaining[0]==0 and s.closing_event_anchor[0]==.6

def test_hold_does_not_expire_in_deleted_ticks_and_partial_ack():
    s=sampler();initial=sample()[0];a=initial.copy();a[6]=.5
    s.acknowledge([a],initial)
    w=np.repeat(a[None],15,axis=0);idx,_=s.select(w,a)
    assert idx==list(range(15))
    s.acknowledge(w[:2],a)
    assert s.hold_remaining[0]==13

def test_legacy_config_identity_unchanged_and_explicit_new_version():
    c=SelectionConfig(gripper_epsilon=.005)
    assert 'closing_hold_epsilon' not in configuration_dict(c)
    assert algorithm_version(c)=='joint_fk_rdp_significant_gripper_v2'
    assert algorithm_version(sampler().config)=='joint_fk_rdp_split_gripper_v3'
    for value in [.001, float('nan'),float('inf')]:
        with pytest.raises(ValueError):SelectionConfig(gripper_epsilon=.005,closing_hold_epsilon=value)
