"""Executor safety and truthful receipts using a deterministic session fixture."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import joint_motion as m
from .action_edit_kinematics import transform, pose


class CartesianFK:
    # Full-rank numerical fixture, not a model of the real robot's performance.
    def matrix(self, q):
        out = np.eye(4)
        out[:3, 3] = q[:3]
        out[:3, :3] = Rotation.from_rotvec(q[3:]).as_matrix()
        return out


def point(x=0.1, angle=0.0, gripper="keep"):
    return dict(
        position=[x, 0, 0],
        quaternion_wxyz=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)],
        gripper=gripper,
    )


class FakeSession:
    def __init__(self, path, lag=0.0, terminal_at=1000):
        from .gripper_hold import HoldConfig
        self.gripper_hold_config = HoldConfig(3, 2, 8, 1e-6)
        self.finger_positions = np.full((2, 2), .04)
        self.q = {a: np.zeros(6) for a in ("left", "right")}
        self.kinematics = SimpleNamespace(
            fk={a: CartesianFK() for a in self.q},
            root=lambda arm: np.eye(4),
            check=lambda: None,
            check_self_collision=lambda arm, joints: dict(fixture=True),
        )
        self.episode_dir = path
        self.episode_id = "episode"
        self.step_id = 0
        self.metadata = dict(control_dt=0.04, max_episode_steps=1000)
        self.finished = self.terminated = self.truncated = False
        self.lag = lag
        self.terminal_at = terminal_at
        self.commands = []
        self.update_obs()

    def update_obs(self):
        mats = [self.kinematics.fk[a].matrix(self.q[a]) for a in ("left", "right")]
        self.obs = dict(
            states=np.r_[self.q["left"], 1, self.q["right"], 1],
            finger_positions_available=True,
            finger_positions_m=self.finger_positions.copy(),
            finger_positions_step_id=self.step_id,
            eef_positions=np.array([p[:3, 3] for p in mats]),
            eef_quaternions_wxyz=np.array([pose(p)["quaternion_wxyz"] for p in mats]),
        )

    def chunk_step(self, actions, **kw):
        a = np.asarray(actions[0])
        self.commands.append(a.copy())
        self.step_id += 1
        from .motion_boundary import record_action
        record_action(self, a)
        for arm, off in (("left", 0), ("right", 7)):
            self.q[arm] = self.lag * self.q[arm] + (1 - self.lag) * a[off : off + 6]
        self.finger_positions[:] = np.asarray([a[6], a[13]])[:, None] * .04
        self.update_obs()
        self.terminated = self.step_id >= self.terminal_at
        row = dict(
            valid=True,
            executed_action=a.copy(),
            obs=dict(states=self.obs["states"]),
            step_id=self.step_id,
            terminated=self.terminated,
            truncated=False,
            success=self.terminated,
        )
        return dict(step_id=self.step_id, steps=[row])


@pytest.fixture
def simulator(tmp_path, monkeypatch):
    monkeypatch.setattr(
        m,
        "robot_state",
        lambda kin, arm: (
            session.q[arm].copy(),
            np.array([[-3.0, 3.0]] * 6),
            np.ones(6) * 3.14,
        ),
    )
    session = FakeSession(tmp_path)
    monkeypatch.setattr(m, "robot_velocity", lambda kin, arm: np.zeros(6))
    return session


def run(session, points, limits=m.DEFAULTS):
    return m.execute(session, "left", points, 0, "request", limits)


def test_gripper_close_follows_pose_arrival_and_measured_stopping(simulator):
    result = run(simulator, [point(0.05, gripper="close")])
    report = result["motion_result"]
    assert report["status"] == "waypoints_reached_not_grasp_verified"
    assert simulator.commands[0][0] > 0 and simulator.commands[0][6] == 1
    closed = [action for action in simulator.commands if action[6] == 0]
    assert len(closed) == 3
    assert all(abs(action[0] - .05) < m.DEFAULTS.ik_position_m for action in closed)
    assert report['events'][0]['status'] == 'fingers_stopped_not_grasp_verified'
    assert report['events'][0]['samples'][-1]['stable_transitions']['left'] == 2
    archive = np.load(simulator.episode_dir / "motion/000_segments/000_00_plan.npz")
    assert np.all(archive["gripper_openings"] == 1)


def test_old_gripper_protocol_is_rejected_before_actions(simulator):
    waypoint = point(0.05, gripper="close")
    waypoint.update(gripper_timing="after_arrival", hold_steps=2)
    with pytest.raises(ValueError, match="fields"):
        run(simulator, [waypoint])
    assert not simulator.commands


def test_stall_stops_without_close_or_pullback(simulator):
    simulator.lag = 1.0
    result = run(simulator, [point(0.15)])
    assert result["motion_result"]["status"] == "tracking_error"
    assert np.all(np.diff([a[0] for a in simulator.commands]) >= -1e-8)
    assert all(a[6] == 1 for a in simulator.commands)
    assert all(
        row["opening"] == 1 for row in result["motion_result"]["gripper_commands"]
    )


def test_colliding_plan_is_rejected_before_motion_or_after_arrival_grip(simulator):
    def blocked(arm, joints):
        raise m.SelfCollisionError("fixture self-collision")

    simulator.kinematics.check_self_collision = blocked
    result = run(simulator, [point(0.1)])
    assert result["motion_result"]["status"] == "self_collision"
    assert result["motion_result"]["executed_steps"] == 0
    assert not simulator.commands
    assert all(
        row["opening"] == 1 for row in result["motion_result"]["gripper_commands"]
    )


def test_large_rotation_lag_stops_not_reverse(simulator, monkeypatch):
    original = simulator.chunk_step

    def step(actions, **kw):
        result = original(actions, **kw)
        simulator.q["left"][3:] = 0.0
        simulator.update_obs()
        return result

    monkeypatch.setattr(simulator, "chunk_step", step)
    result = run(simulator, [point(0.1, 0.8)])
    assert result["motion_result"]["status"] in (
        "tracking_error",
        "joint_tracking_error",
    )
    assert np.all(np.diff([a[0] for a in simulator.commands]) >= -1e-8)


@pytest.mark.parametrize("reason", ["budget", "terminal", "planning", "exception"])
def test_all_exit_paths_save_evidence(simulator, monkeypatch, reason):
    limits = m.DEFAULTS
    if reason == "budget":
        limits = replace(limits, max_steps=2)
    if reason == "terminal":
        simulator.terminal_at = 2
    if reason == "planning":

        def fail(*args, **kw):
            raise m.PlanningError("test unreachable")

        monkeypatch.setattr(m, "plan_path", fail)
    if reason == "exception":

        def fail(*args, **kw):
            raise RuntimeError("sim failure")

        monkeypatch.setattr(simulator, "chunk_step", fail)
        with pytest.raises(RuntimeError):
            run(simulator, [point()], limits)
    else:
        result = run(simulator, [point(0.1)], limits)
        assert (
            result["motion_result"]["status"]
            == dict(
                budget="planning_failed",
                terminal="native_terminal",
                planning="planning_failed",
            )[reason]
        )
        assert all(a[6] == 1 for a in simulator.commands)
    report = json.loads((simulator.episode_dir / "motion/000.json").read_text())
    assert report["status"] != "running"
    assert report["executed_steps"] == len(simulator.commands)


def test_disabled_session_rejects_new_rpc():
    from .session import RoboDojoSession

    session = object.__new__(RoboDojoSession)
    session.motion_enabled = False
    session.scene_query_enabled = False
    session.poisoned = False
    session.episode_id = "e"
    session.step_id = 0
    with pytest.raises(ValueError, match="Unsupported"):
        session.dispatch("execute_joint_motion", dict(episode_id="e", step_id=0))


def test_right_arm_order_and_other_arm_hold(simulator):
    result = m.execute(simulator, "right", [point(0.06)], 0, "request")
    assert result["motion_result"]["status"] == "waypoints_reached_not_grasp_verified"
    assert all(np.all(a[:6] == 0) for a in simulator.commands)
    assert simulator.commands[-1][7] == pytest.approx(0.06, abs=0.0005)


def test_gripper_schedule_switches_at_interval_start():
    points = [point(0.03, gripper="open"), point(0.06, gripper="close"), point(0.09)]
    openings, owners = m.gripper_schedule(points, [3, 7, 9], 9, 0.4)
    np.testing.assert_array_equal(openings, [1, 1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(owners, [0, 0, 0, 1, 1, 1, 1, 2, 2])


@pytest.mark.parametrize("via_steps", [[0, 0], [2, 2]])
def test_conflicting_same_tick_commands_are_rejected(via_steps):
    with pytest.raises(m.PlanningError, match="Conflicting"):
        m.gripper_schedule(
            [point(gripper="close"), point(gripper="open")],
            via_steps,
            via_steps[-1],
            1.0,
        )


def test_stationary_close_uses_one_bounded_hold_and_repeat_is_deduplicated(simulator):
    result = run(simulator, [point(0, gripper="close"), point(0, gripper="close")])
    assert result["motion_result"]["status"] == "waypoints_reached_not_grasp_verified"
    assert len(simulator.commands) == 3
    assert len(result['motion_result']['events']) == 1
    assert all(action[6] == 0 for action in simulator.commands)
    np.testing.assert_array_equal(np.asarray(simulator.commands)[:, :6], np.zeros((3, 6)))


def test_stationary_close_then_open_are_separate_pose_events(simulator):
    result = run(simulator, [point(0, gripper="close"), point(0, gripper="open")])
    assert result["motion_result"]["status"] == "waypoints_reached_not_grasp_verified"
    assert len(simulator.commands) == 4
    assert [action[6] for action in simulator.commands] == [0, 0, 0, 1]


def test_keep_uses_observed_fractional_opening(simulator):
    simulator.obs["states"][6] = 0.37
    result = run(simulator, [point(0.05)])
    assert all(action[6] == 0.37 for action in simulator.commands)
    assert all(
        row["opening"] == 0.37 for row in result["motion_result"]["gripper_commands"]
    )


def test_initial_motion_is_immediate_but_close_is_after_arrival(simulator, monkeypatch):
    monkeypatch.setattr(m, "robot_velocity",
                        lambda kin, arm: np.full(6, .1 if simulator.step_id == 0 else 0))
    result = run(simulator, [point(.05, gripper="close")])
    assert simulator.commands[0][0] > 0
    assert simulator.commands[0][6] == 1
    assert result['motion_result']['events'][0]['executed_steps'] == 3


def test_initial_duplicate_keep_has_no_new_gripper_obligation():
    openings, owners = m.gripper_schedule(
        [point(0), point(0.03, gripper="close")], [0, 3], 3, 0.37
    )
    np.testing.assert_array_equal(openings, [0, 0, 0])
    np.testing.assert_array_equal(owners, [1, 1, 1])


@pytest.mark.parametrize("offset,opening", [(6, float("nan")), (13, 1.1)])
def test_invalid_either_arm_opening_fails_before_native_action(
    simulator, offset, opening
):
    simulator.obs["states"][offset] = opening
    with pytest.raises(ValueError, match="gripper opening"):
        run(simulator, [point(0.03)])
    assert not simulator.commands
