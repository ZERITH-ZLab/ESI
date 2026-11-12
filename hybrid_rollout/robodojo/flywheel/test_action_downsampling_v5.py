import json
from pathlib import Path

import numpy as np
import pytest

from .action_downsampling import ActionDownsampler, SelectionConfig, algorithm_version, configuration_dict
from .test_action_downsampling import sample, toy_fk
from .test_action_downsampling_v4 import actions


def config():
    return SelectionConfig(**json.loads(Path(__file__).parents[1].joinpath('downsampling.gripper_v5.json').read_text()))


def sampler():
    return ActionDownsampler(config(), toy_fk)


def test_small_unconfirmed_drift_has_no_phase_protection():
    s = sampler(); a = actions(np.linspace(.6,.591,15))
    idx, r = s.select(a,sample()[0])
    assert r['closing_phase_protected_indices'] == []
    assert r['closing_phase_events'] == []
    assert len(idx) < 15
    s.acknowledge(a[idx],sample()[0])
    assert s.hold_remaining.tolist() == [0,0]


def test_confirmed_close_backfills_start_only_in_approved_prefix():
    s = sampler(); a = actions([.6,.594,.588,.582,.576] + [.576]*10)
    idx,r = s.select(a,sample()[0])
    assert idx == list(range(15))
    assert r['closing_phase_events'] == [dict(index=4,arm=0,event='closing_confirmed')]
    assert s.closing_phases is None
    s.acknowledge(a[:5],sample()[0]);assert s.hold_remaining[0] == 10
    s.acknowledge(a[5:7],a[4]);assert s.hold_remaining[0] == 8
    s.acknowledge(a[7:],a[6]);assert s.hold_remaining[0] == 0


def test_tiny_new_minima_cannot_extend_finite_tail():
    s = sampler(); initial=sample()[0]
    s.acknowledge(actions([.5]),initial)
    a=actions(.5-np.arange(1,16)*.0002)
    idx,r=s.select(a,initial)
    assert r['closing_phase_protected_indices'] == list(range(10))
    assert not r['closing_phase_events']
    s.acknowledge(a[:10],initial);assert s.hold_remaining[0] == 0
    idx,r=s.select(a[10:],a[9]);assert r['closing_phase_protected_indices'] == []


def test_progress_requires_cumulative_new_low_not_return_to_old_target():
    s=sampler();initial=sample()[0]
    s.acknowledge(actions([.5]*11),initial)
    for g in [.499,.498,.499,.5,.515,.5]:
        s.acknowledge(actions([g]),initial);assert s.hold_remaining[0] == 0
    for g in [.496,.492]:
        s.acknowledge(actions([g]),initial);assert s.hold_remaining[0] == 0
    s.acknowledge(actions([.488]),initial);assert s.hold_remaining[0] == 10


def test_opening_rearms_one_new_grasp_without_extending_old_tail():
    s=sampler();initial=sample()[0]
    s.acknowledge(actions([.5]*11),initial)
    s.acknowledge(actions([.525,.5]),initial)
    assert s.closing_phases.arms[0].mode == 'latched' and s.hold_remaining[0] == 0
    s.acknowledge(actions([.53,.53]),initial)
    assert s.closing_phases.arms[0].mode == 'ready' and s.hold_remaining[0] == 0
    s.acknowledge(actions([.5]),initial);assert s.hold_remaining[0] == 10


def test_zero_partial_ack_and_unapproved_tail_do_not_commit_future_close():
    s=sampler();initial=sample()[0];a=actions([.6]*15+[.4]*5)
    idx,r=s.select(a[:15],initial,a)
    assert not r['closing_phase_events']
    s.acknowledge(a[:1],initial);s.acknowledge([],initial)
    assert s.hold_remaining.tolist() == [0,0]
    b=actions([.6]*5+[.5]*10)
    s.select(b,initial);s.acknowledge(b[:5],initial)
    assert s.hold_remaining.tolist() == [0,0]


def test_cross_chunk_signed_close_and_right_arm_shared_retention():
    s=sampler();initial=sample()[0]
    s.acknowledge(actions([.594,.588,.582],arm=1),initial)
    assert s.hold_remaining[1] == 0
    a=actions([.576]*15,arm=1);idx,r=s.select(a,initial)
    assert r['closing_phase_protected_indices'] == list(range(11))
    s.acknowledge(a[:1],initial);assert s.hold_remaining.tolist() == [0,10]
    s.acknowledge(a[:2],initial);assert s.hold_remaining[1] == 8


def test_local_guard_and_skip_limit_do_not_change_targets():
    s=sampler();initial=sample()[0];a=actions([.6]*7+[.612]*8,arm=1)
    before=a.copy();idx,r=s.select(a,initial)
    assert r['protected_gripper_indices'] == [6,7]
    assert all(j-i-1<=4 for i,j in zip([-1,*idx[:-1]],idx))
    np.testing.assert_array_equal(a,before)


def test_config_identity_is_explicit_and_legacy_versions_unchanged():
    assert algorithm_version(config()) == 'joint_fk_rdp_significant_closing_v5'
    assert 'closing_onset_epsilon' not in configuration_dict(config())
    for c in [SelectionConfig(),SelectionConfig(closing_hold_epsilon=.02),
              SelectionConfig(closing_phase_enabled=True,closing_hold_epsilon=.02)]:
        assert 'closing_refresh_epsilon' not in configuration_dict(c)
    with pytest.raises(ValueError):SelectionConfig(closing_refresh_epsilon=.01)
    with pytest.raises(ValueError):SelectionConfig(**(configuration_dict(config())|dict(closing_refresh_epsilon=.03)))
