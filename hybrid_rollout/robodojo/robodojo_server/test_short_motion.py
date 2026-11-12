"""CPU execution contracts; these fixtures do not establish native performance."""

from types import SimpleNamespace

import numpy as np
import pytest

from . import joint_motion as motion, motion_boundary, short_motion
from .session import RoboDojoSession
from .test_joint_motion import simulator
from .test_self_collision import guard


def target(x=0.0, angle=0.0, opening=1.0):
    return dict(position=[x, 0.0, 0.0],
                quaternion_wxyz=[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)],
                gripper_closed=opening < .5, gripper_opening=opening)


def test_complete_rotation_goal_and_parallel_fractional_grippers(simulator):
    goals = dict(left=target(.03, .3, .23), right=target(opening=.37))
    result = short_motion.execute(simulator, [goals], 5, 0, "request",
        gripper_intent=dict(left="keep", right="keep"))
    receipt = result["correction_result"]
    assert receipt["status"] == "targets_reached_not_grasp_verified"
    assert receipt["final_arrival_verified"]
    assert 5 <= len(simulator.commands) <= 60
    assert receipt["executed_steps"] == len(simulator.commands)
    for arm in ("left", "right"):
        final = simulator.kinematics.fk[arm].matrix(simulator.q[arm])
        original = motion.transform(goals[arm]["position"], goals[arm]["quaternion_wxyz"])
        dp, dr = motion.pose_error(final, original)
        assert dp <= motion.DEFAULTS.ik_position_m
        assert dr <= motion.DEFAULTS.ik_angle_rad
    # The requested .3-radian goal must survive the former .1-radian DLS cap.
    assert simulator.q["left"][5] == pytest.approx(.3, abs=.005)
    actions = np.asarray(simulator.commands)
    np.testing.assert_array_equal(actions[:, 6], np.full(len(actions), np.float32(.23)))
    np.testing.assert_array_equal(actions[:, 13], np.full(len(actions), np.float32(.37)))
    assert actions[0, 0] > 0
    assert receipt["targets"] == [goals]


def test_remaining_budget_refuses_before_any_ack(simulator):
    simulator.metadata["max_episode_steps"] = 4
    result = short_motion.execute(simulator, [dict(left=target(.03), right=target())],
                                  5, 0, "request")
    assert result["correction_result"]["status"] == "planning_failed"
    assert "Insufficient remaining" in result["correction_result"]["error"]
    assert result["steps"] == [] and simulator.commands == []
    assert simulator.step_id == 0
    assert not getattr(simulator, "_motion_action_history", [])


def test_real_sphere_guard_refuses_before_any_ack(simulator, guard):
    # Reuse the actual one-joint URDF/sphere checker, projected onto this
    # deterministic Cartesian fixture's first coordinate. This tests refusal
    # wiring with genuine geometry, not the native X5 collision model.
    simulator.kinematics.check_self_collision = (
        lambda arm, joints: guard.validate_path(np.asarray(joints)[:, :1]))
    result = short_motion.execute(simulator, [dict(left=target(.03), right=target())],
                                  5, 0, "request")
    assert result["correction_result"]["status"] == "self_collision"
    assert "clearance" in result["correction_result"]["error"]
    assert result["steps"] == [] and simulator.commands == []
    assert simulator.step_id == 0
    assert not getattr(simulator, "_motion_action_history", [])


def test_real_session_chunk_keeps_pi_actions_and_records_ack_when_macro_disabled(tmp_path):
    class CountingEnvironment:
        """ACK protocol fixture only; no physics is advanced."""

        def __init__(self):
            self.take_action_cnt = [0]
            self.end_flag = [False]
            self.success = [True]
            self.step_lim = 100
            self.commands = []

        def take_action(self, command):
            self.commands.append(np.concatenate([
                command["left_arm_joint_state"], command["left_ee_joint_state"],
                command["right_arm_joint_state"], command["right_ee_joint_state"]]).copy())
            self.take_action_cnt[0] += 1

    session = object.__new__(RoboDojoSession)
    session.gripper_contact = session.gripper_contact_error = session.gripper_contact_cache = None
    session.pi_force_hold = None
    session.poisoned = False
    session.gripper_contact_enabled = True
    session.motion_enabled = False
    session.pi05_collision_shadow = False
    session.finished = session.terminated = session.truncated = False
    session.env = CountingEnvironment()
    session.episode_id, session.step_id = "episode", 0
    session.episode_dir = tmp_path
    session.metadata = dict(control_dt=.04, max_episode_steps=100)
    session.source, session.control_epoch = "student", 0
    session.obs = dict(states=np.zeros(14, np.float32))

    def observe(record):
        session.obs = dict(states=session.env.commands[-1].copy())
        return session.obs

    session._observe = observe
    session.record_native_stages = lambda: None
    actions = np.arange(42, dtype=np.float32).reshape(3, 14) / 100
    original = actions.copy()
    result = RoboDojoSession.chunk_step(session, actions, student_prediction_id="pi")
    np.testing.assert_array_equal(actions, original)
    np.testing.assert_array_equal(session.env.commands, original)
    np.testing.assert_array_equal([row["executed_action"] for row in result["steps"]], original)
    assert result["step_id"] == 3 and len(result["steps"]) == 3
    assert session.env.take_action_cnt == [3]
    assert all(row["student_prediction_id"] == "pi" for row in result["steps"])
    history = session._motion_action_history
    np.testing.assert_array_equal([row["joints"] for row in history], original)
    for arm, offset in (("left", 0), ("right", 7)):
        v, a, source = motion_boundary.initial_derivatives(session, arm)
        assert source == "issued_action_differences"
        q = original[:, offset:offset + 6].astype(float)
        np.testing.assert_allclose(v, (q[-1] - q[-2]) / .04)
        np.testing.assert_allclose(a, (q[-1] - 2 * q[-2] + q[-3]) / .04**2)


def test_duplicate_poses_preserve_gripper_index_sequence_and_extended_values(simulator):
    goals = [dict(left=target(opening=.2), right=target(opening=.7)),
             dict(left=target(opening=.4), right=target(opening=.9))]
    for decision, count in enumerate((2, 5)):
        prior = len(simulator.commands)
        requested = 1 if count == 2 else count
        result = short_motion.execute(simulator, goals, requested, decision, "request",
            gripper_intent=dict(left="keep", right="keep"))
        receipt = result["correction_result"]
        assert receipt["status"] == "targets_reached_not_grasp_verified"
        assert receipt["requested_minimum_steps"] == requested
        assert receipt["effective_minimum_steps"] == max(requested, len(goals))
        assert receipt["executed_steps"] == count
        actions = np.asarray(simulator.commands[prior:])
        assert len(actions) == count
        np.testing.assert_array_equal(actions[:, :6], np.zeros((count, 6)))
        np.testing.assert_array_equal(actions[:, 7:13], np.zeros((count, 6)))
        for arm, offset, first, final in (("left", 6, .2, .4), ("right", 13, .7, .9)):
            # Original N=2 is preserved literally; N=5 keeps first/last and
            # uses the intervening quarter-index values on the same arm ticks.
            expected = np.asarray([first, final] if count == 2 else
                [first, first + (final - first) / 4, (first + final) / 2,
                 first + 3 * (final - first) / 4, final], dtype=np.float32)
            np.testing.assert_array_equal(actions[:, offset], expected)
            channel = receipt["gripper_schedule"][arm]
            assert channel["original_openings"] == [first, final]
            assert channel["arrival_at_via_controls_gripper"] is False
            np.testing.assert_array_equal(channel["planned_openings_float32"], expected)
            np.testing.assert_array_equal(channel["acknowledged_openings"], expected)
