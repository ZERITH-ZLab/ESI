"""Real action/ACK boundary with fake physics; no grasp-success claim."""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest

from .pi_force_hold import PiForceHold, ForceHoldConfig, execution_version
from .session import RoboDojoSession
from .pi_force_hold import validate_tightening_context, validate_tightening_policy
from .test_pi_contact_tighten import bootstrap_config, action, receipt, ACTOR
from ..force_hold_archive import validate_hold_mapping, validate_proposal_mapping, requested_action


@pytest.fixture
def sim(tmp_path):
    class Env:
        def __init__(self):
            self.take_action_cnt, self.end_flag, self.success = [0], [False], [False]
            self.commands = []
        def take_action(self, cmd):
            self.commands.append(np.r_[cmd['left_arm_joint_state'], cmd['left_ee_joint_state'],
                                       cmd['right_arm_joint_state'], cmd['right_ee_joint_state']])
            self.take_action_cnt[0] += 1

    s = object.__new__(RoboDojoSession)
    s.env, s.episode_dir, s.episode_id, s.step_id = Env(), tmp_path, 'ep', 0
    s.finished = s.terminated = s.truncated = s.poisoned = s.pi05_collision_shadow = False
    s.source, s.control_epoch = 'student', 0
    s.metadata = dict(control_dt=.04, max_episode_steps=100)
    s.kinematics = SimpleNamespace(manager=SimpleNamespace(get_joint=lambda _: [np.zeros(6)]),
        robots=dict(left='L', right='R'), check_self_collision=lambda *args: {})
    s.gripper_contact = SimpleNamespace(begin=lambda _: None)
    s.gripper_contact_cache = receipt(0)
    s.gripper_contact_error = None
    s.force, s.actor = 1., ACTOR
    s.pi_force_hold = PiForceHold(bootstrap_config(tighten_mode='zero'))
    s.pi_force_hold.allowed_actors = {ACTOR, '/World/envs/env_0/rigid/other'}
    s.pi_force_hold.finger_closing_directions = dict(left=np.array([-1., -1.]), right=np.array([-1., -1.]))

    def observe(record=True):
        s.obs = dict(states=s.env.commands[-1].copy() if s.env.commands else action(.8),
            finger_positions_available=True, finger_positions_step_id=s.step_id,
            finger_positions_m=np.full((2, 2), .04-(.001 if s.step_id >= 2 else 0.)))
        return s.obs
    def record_contact():
        s.gripper_contact_cache = receipt(s.step_id, force=s.force, actor=s.actor)
    s._observe, s._record_gripper_contact = observe, record_contact
    s.record_native_stages = lambda: None
    observe()
    return s


def start_grasp(s):
    proposal = np.stack([action(.8), action(.7)])
    s.chunk_step(proposal, student_prediction_id='p0')
    result = s.chunk_step([action(.7)], original_action_count=50, student_prediction_id='p1')
    assert result['force_hold_steps'] == 3
    assert all(row['executed_action'][6] == 0 for row in result['steps'])
    assert result['tighten_steps'] == 1
    assert 'left' in s.pi_force_hold.zero_latch.active
    return result


def all_receipts(s):
    return [json.loads(p.read_text()) for p in sorted(s.episode_dir.glob('action_*.json'))]


def test_zero_hold_then_transport_keeps_original_proposal_and_actual_acks(sim):
    start_grasp(sim)
    proposal = np.stack([action(.7), action(.7)])
    proposal[:, 0] = [.01, .02]
    immutable = proposal.copy()
    result = sim.chunk_step(proposal, student_prediction_id='p2')
    np.testing.assert_array_equal(proposal, immutable)
    assert result['proposal_steps_consumed'] == 2 and result['force_hold_steps'] == 0
    for row, original in zip(result['steps'], proposal):
        np.testing.assert_array_equal(requested_action(row), original)
        assert row['executed_action'][6] == 0
        assert row['executed_action'][13] == original[13]
    np.testing.assert_array_equal(sim.pi_force_hold.last_model_action, proposal[-1])
    assert sim.pi_force_hold.last_action[6] == 0
    validate_proposal_mapping(result['steps'], proposal)
    receipts = all_receipts(sim)
    _, events = validate_hold_mapping(sim.episode_dir, receipts)
    assert len(events) == 1
    assert list(events.values())[0]['version'] == 'contact_zero_v4'
    np.testing.assert_array_equal(receipts[-1]['executed_action'], sim.env.commands[-1])


def test_opening_uses_original_low_not_zero_and_is_per_arm(sim):
    start_grasp(sim)
    noise = sim.chunk_step([action(.705, .85)], student_prediction_id='p2')['steps'][0]
    assert noise['executed_action'][6] == 0  # Opposite arm may open independently.
    opened = sim.chunk_step([action(.72, .85)], student_prediction_id='p3')['steps'][0]
    assert opened['executed_action'][6] == pytest.approx(.72)
    assert not sim.pi_force_hold.zero_latch.active
    assert not sim.pi_force_hold.arms['left']['used']  # Original opening rearms.


def test_handoff_retains_latch_and_does_not_treat_gpt_zero_as_pi_open_baseline(sim):
    start_grasp(sim)
    sim.source = 'gpt_eef'
    sim.pi_force_hold.control_switch(sim, sim.source)
    sim.chunk_step([action(0.)], teacher_request_id='keep')
    sim.source = 'student'
    sim.pi_force_hold.control_switch(sim, sim.source)
    r = sim.chunk_step([action(.7)], student_prediction_id='return')['steps'][0]
    assert r['executed_action'][6] == 0
    sim.source = 'gpt_eef'
    sim.pi_force_hold.control_switch(sim, sim.source)
    r = sim.chunk_step([action(1.)], teacher_request_id='release')['steps'][0]
    assert r['executed_action'][6] == 1 and not sim.pi_force_hold.zero_latch.active


@pytest.mark.parametrize('failure', ['lost', 'different', 'stale', 'invalid'])
def test_contact_failure_cancels_override_without_forced_open_or_retrigger(sim, failure):
    start_grasp(sim)
    if failure == 'lost':
        sim.gripper_contact_cache = receipt(sim.step_id, force=0.)
    elif failure == 'different':
        sim.gripper_contact_cache = receipt(sim.step_id, actor='/World/envs/env_0/rigid/other')
    elif failure == 'stale':
        sim.gripper_contact_cache['step_id'] -= 1
    else:
        sim.gripper_contact_cache['reports_valid'] = False
    r = sim.chunk_step([action(.7)], student_prediction_id='p2')['steps'][0]
    assert r['executed_action'][6] == pytest.approx(.7)  # Restore original, do not force 1.
    assert not sim.pi_force_hold.zero_latch.active
    assert sim.pi_force_hold.consider(sim, action(.7)) == []


def test_empty_grasp_never_activates_zero(sim):
    sim.force = 0.
    r = sim.chunk_step([action(.8), action(.7), action(.6)], student_prediction_id='p0')
    assert r['force_hold_steps'] == 0
    assert not sim.pi_force_hold.zero_latch.active
    assert sim.env.commands[-1][6] == pytest.approx(.6)


def test_gpt_closure_requires_real_movement_and_contact_then_survives_handoff(sim):
    sim.source = 'gpt_eef'
    sim.chunk_step([action(0.)], teacher_request_id='close1')
    assert not sim.pi_force_hold.zero_latch.active  # Contact alone is insufficient.
    sim.chunk_step([action(0.)], teacher_request_id='close2')
    assert 'left' in sim.pi_force_hold.zero_latch.active
    sim.source = 'student'
    sim.pi_force_hold.control_switch(sim, sim.source)
    row = sim.chunk_step([action(.7)], student_prediction_id='handoff')['steps'][0]
    assert row['executed_action'][6] == 0
    validate_proposal_mapping([row], [action(.7)])


def test_gpt_empty_close_does_not_latch(sim):
    sim.source, sim.force = 'gpt_eef', 0.
    sim.chunk_step([action(0.), action(0.), action(0.)], teacher_request_id='empty')
    assert not sim.pi_force_hold.zero_latch.active


def test_episode_initialize_clears_latch_and_source_baselines(sim, monkeypatch):
    start_grasp(sim)
    from . import pi_force_hold
    monkeypatch.setattr(pi_force_hold, 'registered_dynamic_actors', lambda _: {ACTOR})
    sim.kinematics.robots = {arm: SimpleNamespace(gripper_scale=[-.010, .044],
        gripper_move={'sign': 1, 'mimic': [1, 1]}) for arm in ('left', 'right')}
    sim.metadata['pi_force_hold'] = {}
    sim.pi_force_hold.initialize(sim)
    assert not sim.pi_force_hold.zero_latch.active
    assert not sim.pi_force_hold.zero_latch.student_latest
    assert not sim.pi_force_hold.zero_latch.gpt_closing


@pytest.mark.parametrize('corruption', ['arm', 'unlisted_finger', 'actor', 'future', 'original', 'nonzero'])
def test_archive_rejects_unexplained_override(sim, corruption):
    start_grasp(sim)
    row = deepcopy(sim.chunk_step([action(.7)], student_prediction_id='p2')['steps'][0])
    info = row['gripper_override']
    if corruption == 'arm': row['executed_action'][0] += .1
    elif corruption == 'unlisted_finger': row['executed_action'][13] = 0
    elif corruption == 'actor': info['arms']['left']['actor'] = '/wrong'
    elif corruption == 'future': info['observation_step_id'] += 1
    elif corruption == 'original': info['original_action'][6] = .8
    else: row['executed_action'][6] = .1
    with pytest.raises(ValueError):
        validate_proposal_mapping([row], [action(.7)])


def test_bootstrap_gate_and_legacy_mode():
    assert execution_version(ForceHoldConfig()) == 'contact_margin_v3'
    assert execution_version(bootstrap_config(tighten_mode='zero')) == 'contact_zero_v4'
    hold_only = ForceHoldConfig(tighten_mode='zero', tighten_enabled=False)
    assert PiForceHold(hold_only).zero_latch is None
    assert execution_version(hold_only) == 'contact_margin_v3'
    with pytest.raises(ValueError):
        bootstrap_config(tighten_mode='unsupported')


def test_holding_without_tightening_never_changes_gripper_target(sim):
    from dataclasses import replace
    config = replace(sim.pi_force_hold.config, tighten_enabled=False)
    sim.pi_force_hold = PiForceHold(config)
    sim.pi_force_hold.allowed_actors = {ACTOR}
    sim.pi_force_hold.finger_closing_directions = dict(
        left=np.array([-1., -1.]), right=np.array([-1., -1.]))
    sim.chunk_step([action(.8), action(.7)], student_prediction_id='p0')
    departure = action(.7)
    departure[0] = .02  # A stationary model action already supplies its own hold.
    result = sim.chunk_step([departure], student_prediction_id='p1')
    assert result['force_hold_steps'] > 0
    assert result['tighten_steps'] == 0
    assert all(row['executed_action'][6] == pytest.approx(.7) for row in result['steps'])
    assert all('gripper_override' not in row for row in result['steps'])


def test_zero_round2_pins_identity_without_bootstrap_restriction():
    cfg = ForceHoldConfig(tighten_enabled=True, tighten_mode='zero', flywheel_round=2,
        controller_route='subtask_feedforward_v1', checkpoint_sha256='a'*64)
    validate_tightening_context(cfg, controller_route='subtask_feedforward_v1', checkpoint_sha256='a'*64)
    validate_tightening_policy(cfg, 'pi05/'+'a'*16)
    validate_tightening_policy(cfg, 'a'*64)
    with pytest.raises(ValueError, match='route'):
        validate_tightening_context(cfg, controller_route='auto_infer_v1')
    with pytest.raises(ValueError, match='checkpoint'):
        validate_tightening_context(cfg, controller_route='subtask_feedforward_v1', checkpoint_sha256='b'*64)
    with pytest.raises(ValueError, match='policy identity'):
        validate_tightening_policy(cfg, 'pi05/'+'b'*16)


@pytest.mark.parametrize('overrides', [dict(flywheel_round=0), dict(flywheel_round=True),
    dict(checkpoint_sha256=None), dict(checkpoint_sha256='trained'), dict(controller_route='baseline')])
def test_zero_requires_explicit_valid_round_route_checkpoint(overrides):
    options = dict(tighten_enabled=True, tighten_mode='zero', flywheel_round=2,
        controller_route='subtask_feedforward_v1', checkpoint_sha256='a'*64)
    with pytest.raises(ValueError):
        ForceHoldConfig(**dict(options, **overrides))


def test_round2_zero_execution_is_idempotent_and_release_still_wins(sim):
    sim.pi_force_hold.config = ForceHoldConfig(tighten_enabled=True, tighten_mode='zero', flywheel_round=2,
        controller_route='subtask_feedforward_v1', checkpoint_sha256='a'*64)
    start_grasp(sim)
    proposal = action(0.)
    first, _ = sim.pi_force_hold.zero_latch.prepare(sim, proposal)
    second, _ = sim.pi_force_hold.zero_latch.prepare(sim, first)
    np.testing.assert_array_equal(first, proposal)
    np.testing.assert_array_equal(second, first)
    opened, _ = sim.pi_force_hold.zero_latch.prepare(sim, action(.3))
    assert opened[6] == pytest.approx(.3)
    assert not sim.pi_force_hold.zero_latch.active
