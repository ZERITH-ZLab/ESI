import json
from pathlib import Path

import numpy as np
import pytest

from .action_downsampling import ActionDownsampler, SelectionConfig, algorithm_version, configuration_dict
from .test_action_downsampling import sample, toy_fk


def config():
    return SelectionConfig(**json.loads(Path(__file__).parents[1].joinpath('downsampling.gripper_v4.json').read_text()))


def sampler():
    return ActionDownsampler(config(), toy_fk)


def actions(values, arm=0):
    a = np.repeat(sample()[:1], len(values), axis=0)
    a[:, 6 if arm == 0 else 13] = values
    return a


def test_slow_closure_whole_phase_and_ten_following_ticks_across_chunks():
    s = sampler(); initial = sample()[0]
    close = actions(.6 - np.arange(1, 31) * .006)
    for chunk in (close[:15], close[15:]):
        idx, report = s.select(chunk, initial)
        assert idx == list(range(15))
        s.acknowledge(chunk[idx], initial)
        initial = chunk[-1]
    assert s.hold_remaining.tolist() == [10, 0]
    tail = actions([close[-1, 6]] * 15)
    idx, report = s.select(tail, initial)
    assert report['closing_phase_protected_indices'] == list(range(10))
    assert len(idx) < 15
    assert s.closing_phases.arms[0].mode == 'closing'  # Pure selection.
    s.acknowledge(tail[:2], initial)
    assert s.closing_phases.arms[0].mode == 'latched'
    assert s.hold_remaining[0] == 8  # Confirmation does not add two ticks.
    s.acknowledge(tail[2:10], initial)
    assert s.hold_remaining[0] == 0


def test_latched_jitter_does_not_restart_tail_or_accumulate_absolute_motion():
    s = sampler(); initial = sample()[0]
    s.acknowledge(actions([.5] * 11), initial)
    assert s.hold_remaining[0] == 0
    for _ in range(5):
        a = actions([.515, .5, .508, .5, .501])
        _, r = s.select(a, a[0])
        assert r['closing_phase_protected_indices'] == []
        s.acknowledge(a, initial)
    assert s.hold_remaining[0] == 0
    assert s.closing_phases.arms[0].mode == 'latched'


def test_sub_deadband_monotonic_tail_is_still_closing_not_settled():
    s = sampler(); initial = sample()[0]
    s.acknowledge(actions([.5]), initial)
    tail = actions(.5 - np.arange(1, 46) * .0002)
    for chunk in np.split(tail, 3):
        idx, _ = s.select(chunk, initial)
        assert idx == list(range(15))
        s.acknowledge(chunk, initial)
    assert s.closing_phases.arms[0].mode == 'closing'
    assert s.hold_remaining[0] == 10


def test_pending_small_bounce_never_starts_ten_tick_tail():
    s = sampler(); initial = sample()[0]
    a = actions([.594, .6, .6] + [.6] * 12)
    idx, r = s.select(a, initial)
    assert r['closing_phase_protected_indices'] == [0, 1, 2]
    s.acknowledge(a[idx], initial)
    assert s.hold_remaining[0] == 0


@pytest.mark.parametrize('opening', [[.53, .53], [.6]])
def test_release_then_regrasp_rearms_protection(opening):
    s = sampler(); initial = sample()[0]
    s.acknowledge(actions([.5] * 11), initial)
    s.acknowledge(actions(opening), initial)
    assert s.closing_phases.arms[0].mode == 'ready'
    s.acknowledge(actions([.5]), initial)
    assert s.closing_phases.arms[0].mode == 'closing'
    assert s.hold_remaining[0] == 10


def test_single_small_opening_spike_does_not_rearm():
    s = sampler(); initial = sample()[0]
    s.acknowledge(actions([.5] * 11), initial)
    s.acknowledge(actions([.525, .5, .525, .5]), initial)
    assert s.closing_phases.arms[0].mode == 'latched'
    assert s.hold_remaining[0] == 0


def test_material_new_low_can_resume_closing_without_release():
    s = sampler(); initial = sample()[0]
    s.acknowledge(actions([.5] * 11), initial)
    s.acknowledge(actions([.47]), initial)
    assert s.hold_remaining[0] == 10


def test_opening_local_guard_is_exactly_previous_and_event_row():
    s = sampler(); initial = sample()[0]
    a = actions([.6] * 7 + [.612] * 8, arm=1)
    idx, r = s.select(a, initial)
    assert r['protected_gripper_indices'] == [6, 7]
    assert {6, 7} <= set(idx)
    assert r['closing_phase_protected_indices'] == []


def test_unexecuted_proposals_and_empty_ack_do_not_start_phase():
    s = sampler(); initial = sample()[0]
    a = actions([.6] * 14 + [.4])
    s.select(a, initial)
    assert s.closing_phases is None
    s.acknowledge(a[:1], initial)
    s.acknowledge([], initial)
    assert s.hold_remaining.tolist() == [0, 0]
    assert s.closing_phases.arms[0].mode == 'ready'


def test_unapproved_lookahead_does_not_start_phase():
    s = sampler(); initial = sample()[0]
    a = actions([.6] * 15 + [.4, .4])
    idx, r = s.select(a[:15], initial, lookahead=a)
    assert r['closing_phase_events'] == []
    s.acknowledge(a[idx], initial)
    assert s.hold_remaining.tolist() == [0, 0]


def test_dual_arm_common_mask_max_skip_and_exact_targets():
    s = sampler(); initial = sample()[0]
    a = actions([.6] * 4 + [.5] * 11, arm=1); original = a.copy()
    idx, r = s.select(a, initial)
    assert set(range(3, 15)) <= set(idx)
    np.testing.assert_array_equal(a, original)
    still, _ = sampler().select(sample(), initial)
    assert all(b-a-1 <= 4 for a,b in zip([-1,*still[:-1]],still))


def test_legacy_identity_and_v4_preset():
    assert algorithm_version(config()) == 'joint_fk_rdp_closing_phase_v4'
    for c in [SelectionConfig(), SelectionConfig(closing_hold_epsilon=.02)]:
        assert not any(k.startswith('closing_phase') or k in ('closing_onset_epsilon','closing_settle_ticks')
                       for k in configuration_dict(c))
    assert config().max_skipped == 4 and config().closing_hold_ticks == 10


@pytest.mark.parametrize('overrides', [dict(closing_hold_epsilon=None),
    dict(closing_settle_ticks=0), dict(closing_onset_epsilon=.03),
    dict(closing_hold_ticks=1), dict(closing_phase_enabled='yes')])
def test_invalid_phase_configuration(overrides):
    with pytest.raises(ValueError):
        SelectionConfig(**(configuration_dict(config()) | overrides))
