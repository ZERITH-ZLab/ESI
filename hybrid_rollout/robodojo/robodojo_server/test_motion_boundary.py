"""Inherited physical derivatives, timing invariants, and acknowledged references."""

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from . import joint_motion as motion
from . import motion_boundary as boundary
from .test_continuous_motion import CartesianFK, point
from .test_joint_motion import simulator


def test_endpoint_derivatives_do_not_dilate_and_knots_remain_exact():
    path = CubicSpline([0., .3, .7, 1.], [[0., .1], [.04, .12], [.08, .09], [.12, .13]],
                       bc_type="natural")
    trajectory = boundary.BoundaryPath(path)
    velocity = np.array([.03, -.02])
    acceleration = np.array([.01, .02])
    end_velocity = np.array([-.01, .015])
    end_acceleration = np.array([-.02, .03])
    ends = dict(end_velocity=end_velocity, end_acceleration=end_acceleration)
    for duration in (.8, 1.7):
        np.testing.assert_allclose(trajectory.sample([0, 1], duration, velocity, acceleration, 1, **ends),
                                   np.stack([velocity, end_velocity]), atol=1e-12)
        np.testing.assert_allclose(trajectory.sample([0, 1], duration, velocity, acceleration, 2, **ends),
                                   np.stack([acceleration, end_acceleration]), atol=1e-11)
        np.testing.assert_allclose(trajectory.sample(trajectory.times, duration, velocity, acceleration, **ends),
                                   path(path.x), atol=1e-12)
        # Independent derivatives recovered from the sampled position polynomial.
        for endpoint in (0., 1.):
            t = endpoint + np.linspace(0, 1e-3, 9) * (1 if endpoint == 0 else -1)
            q = trajectory.sample(t, duration, velocity, acceleration, **ends)
            fit = np.polynomial.polynomial.polyfit((t - endpoint) * duration, q, 5)
            np.testing.assert_allclose(fit[1], velocity if endpoint == 0 else end_velocity, atol=1e-7)
            np.testing.assert_allclose(2 * fit[2], acceleration if endpoint == 0 else end_acceleration, atol=1e-4)
        for knot in trajectory.times[1:-1]:
            for order in (0, 1, 2):
                adjacent = trajectory.sample([knot - 1e-9, knot + 1e-9], duration,
                                             velocity, acceleration, order, **ends)
                np.testing.assert_allclose(adjacent[0], adjacent[1], atol=2e-6)
        position_bounds, peaks = trajectory.bounds(duration, velocity, acceleration, **ends)
        times = np.linspace(0, 1, 10001)
        positions = trajectory.sample(times, duration, velocity, acceleration, **ends)
        assert np.all(positions >= position_bounds[0] - 1e-8)
        assert np.all(positions <= position_bounds[1] + 1e-8)
        for order in (1, 2, 3):
            measured = np.max(np.abs(trajectory.sample(times, duration, velocity, acceleration, order, **ends)), axis=0)
            assert np.all(measured <= peaks[order - 1] + 1e-6)
            np.testing.assert_allclose(measured, peaks[order - 1], rtol=.002, atol=1e-6)


def test_nonzero_planned_boundary_and_hard_limits():
    v = np.array([.03, 0, 0, 0, 0, 0])
    a = np.array([.01, 0, 0, 0, 0, 0])
    result = motion.plan_path(CartesianFK(), np.eye(4), np.zeros(6),
                              np.array([[-3., 3.]] * 6), np.ones(6),
                              [point(.03), point(.06)], .04,
                              start_velocity=v, start_acceleration=a)
    np.testing.assert_array_equal(result["joint_velocities"][-1], np.zeros(6))
    np.testing.assert_array_equal(result["joint_accelerations"][-1], np.zeros(6))
    np.testing.assert_array_equal(result["start_velocity"], v)
    assert not np.array_equal(result["start_velocity"], result["end_velocity"])
    assert np.all(result["derivative_peaks"] <= np.array([motion.DEFAULTS.joint_velocity, motion.DEFAULTS.joint_acceleration, motion.DEFAULTS.joint_jerk])[:, None] * (1 + 1e-8))
    assert np.max(np.abs(result["dense_joints"][:, 1:])) < 1e-9


def test_shared_boundary_jerk_uses_one_sided_span_without_double_counting():
    path = CubicSpline([0., .5, 1.], [[0.], [.05], [.1]], bc_type="natural")
    trajectory = boundary.BoundaryPath(path)
    knot = trajectory.times[1]
    at_knot = trajectory.sample([knot], 1., np.array([.03]), np.array([.02]), 3)
    right = trajectory.sample([knot + 1e-10], 1., np.array([.03]), np.array([.02]), 3)
    np.testing.assert_allclose(at_knot, right, atol=1e-7)


@pytest.mark.parametrize("target,velocity", [(0., .03), (.05, 1.26)])
def test_soft_boundary_recovery_is_explicit_and_never_changes_ack_input(target, velocity):
    v = np.array([velocity, 0, 0, 0, 0, 0])
    original = v.copy()
    result = motion.plan_path(CartesianFK(), np.eye(4), np.zeros(6),
        np.array([[-3., 3.]] * 6), np.ones(6) * 3, [point(target)], .04,
        replace(motion.DEFAULTS, joint_velocity=1.25, urdf_velocity_multiplier=1.), start_velocity=v)
    recovery = result["boundary_recovery"]
    np.testing.assert_array_equal(v, original)
    assert 0 <= recovery["scale"] < 1
    assert recovery["rejected_attempts"]
    assert recovery["strict_continuity_guaranteed"] is False
    np.testing.assert_allclose(result["start_velocity"], v * recovery["scale"])
    np.testing.assert_array_equal(result["end_velocity"], np.zeros(6))


def test_action_reference_replaces_planned_state_and_checks_identity():
    session = SimpleNamespace(episode_id="episode", step_id=0, obs={"states": np.zeros(14)},
                              metadata={"control_dt": .04})
    v, a, source = boundary.initial_derivatives(session, "left")
    assert source == "initial_zero_no_previous_action" and not np.any(v) and not np.any(a)
    for x in (.004, .008, .012):
        action = np.zeros(14)
        action[0] = x
        session.step_id += 1
        boundary.record_action(session, action)
        session.obs["states"] = action.copy()
    v, a, source = boundary.initial_derivatives(session, "left")
    assert source == "issued_action_differences"
    np.testing.assert_allclose(v, [.1, 0, 0, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(a, 0, atol=1e-12)
    boundary.remember_derivatives(session, "left", v / 2, np.ones(6) * .01)
    assert boundary.initial_derivatives(session, "left")[2] == "issued_action_differences"
    np.testing.assert_allclose(boundary.initial_derivatives(session, "left")[0], v)
    session.step_id += 1
    boundary.record_action(session, action)
    assert boundary.initial_derivatives(session, "left")[2] == "issued_action_differences"
    np.testing.assert_allclose(boundary.initial_derivatives(session, "left")[0], 0)
    session.episode_id = "other_episode"
    with pytest.raises(ValueError, match="Stale"):
        boundary.initial_derivatives(session, "left")


def test_two_serial_chunks_inherit_ack_and_never_insert_settle(simulator):
    for decision, target in enumerate((.05, .1)):
        v, a, source = boundary.initial_derivatives(simulator, "left")
        result = motion.execute(simulator, "left", [point(target)], decision, "test")
        report = result["motion_result"]
        assert report["status"] == "waypoints_reached_not_grasp_verified"
        assert {t["stage"] for t in report["tracking"]} == {"move"}
        assert report["boundary_reference"]["source"] == source
        np.testing.assert_allclose(report["boundary_reference"]["inherited_velocity"], v)
        np.testing.assert_allclose(report["boundary_reference"]["inherited_acceleration"], a)
        scale = report["boundary_reference"]["scale"]
        np.testing.assert_allclose(report["boundary_reference"]["effective_start_velocity"], v * scale)
        np.testing.assert_allclose(report["boundary_reference"]["effective_start_acceleration"], a * scale)
        np.testing.assert_array_equal(report["boundary_reference"]["end_velocity"], np.zeros(6))
        history = np.asarray([x["joints"][:6] for x in simulator._motion_action_history])
        np.testing.assert_allclose(boundary.initial_derivatives(simulator, "left")[0],
                                   (history[-1]-history[-2])/.04)


def test_interrupted_chunk_preserves_last_issued_sample_not_unexecuted_end(simulator):
    simulator.terminal_at = 2
    result = motion.execute(simulator, "left", [point(.05)], 0, "test")
    assert result["motion_result"]["status"] == "native_terminal"
    plan = np.load(simulator.episode_dir / "motion/000_00_plan.npz")
    v, a, _ = boundary.initial_derivatives(simulator, "left")
    history = np.asarray([x["joints"][:6] for x in simulator._motion_action_history])
    np.testing.assert_allclose(v, (history[-1]-history[-2])/.04)
    np.testing.assert_allclose(a, (history[-1]-2*history[-2]+history[-3])/.04**2)
    assert not np.array_equal(v, plan["end_velocity"])


def test_zero_seed_remains_zero_not_claimed_as_nonzero_cruise(simulator):
    for decision, target in enumerate((.04, .08)):
        result = motion.execute(simulator, "left", [point(target)], decision, "test")
        assert not np.any(result["motion_result"]["boundary_reference"]["end_velocity"])
        assert not np.any(result["motion_result"]["boundary_reference"]["end_acceleration"])
