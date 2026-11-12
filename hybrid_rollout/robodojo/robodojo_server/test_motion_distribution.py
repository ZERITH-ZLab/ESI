"""Distribution preferences must not alter path geometry or hard limits."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import joint_motion as motion
from . import motion_distribution as distribution
from .test_continuous_motion import CartesianFK, point
from .test_joint_motion import simulator  # pytest fixture, native physics not claimed


def plan(points, profile=None, limits=motion.DEFAULTS):
    return motion.plan_path(CartesianFK(), np.eye(4), np.zeros(6),
                            np.array([[-3., 3.]] * 6), np.ones(6) * 3,
                            points, .04, limits, distribution_profile=profile)


def test_exact_task_lookup_units_and_unknown_task():
    profile, receipt = distribution.task_profile("deposit_coin", "left", .04)
    assert receipt["reference_episode_count"] == 20
    assert profile["linear_velocity_norm"]["p50"] == pytest.approx(.19592862057242588)
    assert profile["angular_velocity_norm"]["p50"] == pytest.approx(np.deg2rad(89.56727538806686))
    assert distribution.task_profile("deposit_coin_random", "left", .04)[0] is None
    assert distribution.task_profile("solve_equation", "left", .04)[1]["status"] == "unavailable_task_profile"
    with pytest.raises(ValueError, match="25 Hz"):
        distribution.task_profile("deposit_coin", "left", .02)


def test_discrete_derivatives_and_pure_rotation_metric_selection():
    t = np.arange(51) * .04
    poses = np.repeat(np.eye(4)[None], len(t), axis=0)
    poses[:, 0, 3] = t**2
    poses[:, :3, :3] = Rotation.from_rotvec(t[:, None] * [[0., 0., .5]]).as_matrix()
    statistics = distribution.sampled_quantiles(poses, .04)
    np.testing.assert_allclose(statistics["linear_acceleration_norm"], [2, 2], atol=1e-10)
    np.testing.assert_allclose(statistics["angular_velocity_norm"], [.5, .5], atol=1e-10)
    poses[:, 0, 3] = 0
    assert distribution.applicable_metrics(poses) == ["angular_velocity_norm", "angular_acceleration_norm"]


def test_slower_reference_improves_score_without_changing_geometry():
    points = [point(.04), point(.08), point(.12)]
    baseline = plan(points)
    poses = [CartesianFK().matrix(q) for q in np.vstack([np.zeros(6), baseline["joints"]])]
    statistics = distribution.sampled_quantiles(poses, .04)
    profile = {metric: dict(count=100, p50=values[0] / (2 if "velocity" in metric else 4),
                           p90=values[1] / (2 if "velocity" in metric else 4))
               for metric, values in statistics.items()}
    fitted = plan(points, profile)
    fit = fitted["distribution_fit"]
    assert fit["duration_changed"]
    assert fit["selected"]["score"] < fit["baseline"]["score"]
    assert fitted["duration"] > baseline["duration"]
    np.testing.assert_array_equal(fitted["dense_joints"], baseline["dense_joints"])
    np.testing.assert_allclose(fitted["joints"][-1], baseline["joints"][-1])
    assert np.max(np.abs(fitted["joints"][:, 1:])) < 1e-9
    assert np.all(fitted["derivative_peaks"] <= baseline["derivative_peaks"] + 1e-9)


def test_unreachable_fast_distribution_never_relaxes_limits():
    profile = {metric: dict(count=100, p50=1000., p90=2000.) for metric in distribution.METRICS}
    # Explicit tight caller limits remain binding even though DEFAULTS changed.
    limits = replace(motion.DEFAULTS, joint_velocity=1.25,
                     joint_acceleration=5., joint_jerk=50., urdf_velocity_multiplier=1.)
    baseline = plan([point(.12)], limits=limits)
    fitted = plan([point(.12)], profile, limits)
    assert fitted["execution_steps"] == baseline["execution_steps"]
    assert fitted["distribution_fit"]["status"] == "closest_under_constraints"
    assert np.all(fitted["derivative_peaks"] <= np.array([1.25, 5, 50])[:, None] * (1 + 1e-8))


def test_stationary_and_budget_behavior_with_profile():
    profile, _ = distribution.task_profile("deposit_coin", "left", .04)
    assert plan([point(0)], profile)["execution_steps"] == 0
    with pytest.raises(motion.PlanningError, match="budget"):
        plan([point(.1)], profile, replace(motion.DEFAULTS, max_steps=2))


def test_actual_executor_uses_task_and_preserves_after_arrival_gripper(simulator):
    simulator.metadata["task"] = "pour_liquid_into_cup"
    result = motion.execute(simulator, "left", [dict(point(.06), gripper="close")], 0, "test")
    report = result["motion_result"]
    assert report["distribution_reference"]["task"] == "pour_liquid_into_cup"
    assert report["segments"][0]["distribution_fit"]["selected"]["score"] is not None
    assert simulator.commands[0][6] == 1
    closed = [command for command in simulator.commands if command[6] == 0]
    assert len(closed) == simulator.gripper_hold_config.min_steps
    assert all(abs(command[0] - .06) < motion.DEFAULTS.ik_position_m for command in closed)
    assert report["status"] == "waypoints_reached_not_grasp_verified"


def test_bad_reference_fails_before_native_action(simulator, monkeypatch):
    def invalid(*args):
        raise ValueError("bad reference")
    monkeypatch.setattr(distribution, "task_profile", invalid)
    with pytest.raises(ValueError, match="bad reference"):
        motion.execute(simulator, "left", [point(.06)], 0, "test")
    assert simulator.commands == []
