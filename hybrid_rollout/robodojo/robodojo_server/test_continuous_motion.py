"""Numerical trajectory contracts; no simulated robot performance claim."""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from . import joint_motion as m


class CartesianFK:
    def matrix(self, q):
        t = np.eye(4)
        t[:3, 3] = q[:3]
        t[:3, :3] = Rotation.from_rotvec(q[3:]).as_matrix()
        return t


def point(x=0.0, angle=0.0):
    return dict(
        position=[x, 0, 0],
        quaternion_wxyz=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)],
        gripper="keep",
    )


def plan(points, limits=m.DEFAULTS):
    return m.plan_path(
        CartesianFK(),
        np.eye(4),
        np.zeros(6),
        np.array([[-3.0, 3.0]] * 6),
        np.ones(6) * 3.0,
        points,
        0.04,
        limits,
    )


@pytest.mark.parametrize(
    "points",
    [
        [point(0.1)],
        [point(0.04), point(0.08), point(0.12)],
        [point(0.04), point(0)],
        [point(0, 0.6)],
        [point(0.000001)],
        [point(0), point(0.1), point(0.1)],
    ],
)
def test_joint_dynamics_and_endpoint(points):
    p = plan(points)
    q = np.vstack([np.zeros(6), p["joints"]])
    v = np.diff(q, axis=0) / 0.04
    assert p["duration"] == pytest.approx(len(p["joints"]) * 0.04)
    assert np.max(np.abs(v)) <= m.DEFAULTS.joint_velocity + 1e-6
    assert (
        np.max(np.abs(np.diff(v, axis=0) / 0.04))
        <= m.DEFAULTS.joint_acceleration + 1e-5
    )
    assert (
        np.max(np.abs(np.diff(v, n=2, axis=0) / 0.04**2), initial=0)
        <= m.DEFAULTS.joint_jerk + 1e-3
    )
    assert np.all(
        p["derivative_peaks"] <= np.array([m.DEFAULTS.joint_velocity, m.DEFAULTS.joint_acceleration, m.DEFAULTS.joint_jerk])[:, None] * (1 + 1e-8)
    )
    assert (
        m.pose_error(
            CartesianFK().matrix(q[-1]),
            m.transform(points[-1]["position"], points[-1]["quaternion_wxyz"]),
        )[0]
        < 0.0005
    )


def test_no_stop_at_collinear_vias_and_return_path_not_erased():
    p = plan([point(0.04), point(0.08), point(0.12)])
    q = np.vstack([np.zeros(6), p["joints"]])
    for i in p["via_steps"][:-1]:
        assert np.min(np.diff(q[max(0, i - 2) : i + 2, 0])) > 0.002
    q = plan([point(0.04), point(0)])["joints"]
    assert np.max(q[:, 0]) > 0.039
    assert abs(q[-1, 0]) < 1e-6


def test_boundaries_and_extrema_against_independent_dense_derivatives():
    from scipy.interpolate import CubicSpline

    x = np.array([0.0, 0.23, 0.58, 1.0])
    y = np.array([[0, 0.1], [0.08, 0.14], [0.02, -0.1], [0.15, 0.05]])
    path = CubicSpline(x, y, bc_type="natural")
    peaks = m._continuous_bounds(path)
    t = np.linspace(0, 1, 100001)
    s = m._time_law(t)
    sd = 30 * t * t * (1 - t) ** 2
    sdd = 60 * t - 180 * t * t + 120 * t**3
    sddd = 60 - 360 * t + 360 * t * t
    derivatives = [
        path(s, 1) * sd[:, None],
        path(s, 2) * sd[:, None] ** 2 + path(s, 1) * sdd[:, None],
        path(s, 3) * sd[:, None] ** 3
        + 3 * path(s, 2) * (sd * sdd)[:, None]
        + path(s, 1) * sddd[:, None],
    ]
    for order, d in enumerate(derivatives):
        assert np.all(np.max(np.abs(d), axis=0) <= peaks[order] * (1 + 1e-7))
        assert np.allclose(np.max(np.abs(d), axis=0), peaks[order], rtol=2e-3)
    assert np.allclose(derivatives[0][[0, -1]], 0)
    assert np.allclose(derivatives[1][[0, -1]], 0)


def test_duplicate_and_infeasible_budget():
    assert plan([point(0), point(0)])["execution_steps"] == 0
    with pytest.raises(m.PlanningError, match="budget"):
        plan([point(0.1)], replace(m.DEFAULTS, max_steps=2))
    with pytest.raises(m.PlanningError):
        plan([point(10)])


def test_measured_velocity_is_reported_without_forcing_a_stop(tmp_path, monkeypatch):
    from .test_joint_motion import FakeSession

    session = FakeSession(tmp_path)
    monkeypatch.setattr(
        m,
        "robot_state",
        lambda kin, arm: (
            session.q[arm].copy(),
            np.array([[-3.0, 3.0]] * 6),
            np.ones(6) * 3,
        ),
    )
    monkeypatch.setattr(
        m,
        "robot_velocity",
        lambda kin, arm: np.zeros(6) if session.step_id == 0 else np.ones(6) * 0.1,
    )
    result = m.execute(session, "left", [point(0.05)], 0, "test")
    assert result["motion_result"]["status"] == "waypoints_reached_not_grasp_verified"
    assert result["motion_result"]["final_joint_speed_rad_s"] > 0.03
    assert not result["motion_result"]["stationary_required"]
    assert all(row["stage"] == "move" for row in result["motion_result"]["tracking"])


def test_between_tick_via_is_not_falsely_verified(tmp_path, monkeypatch):
    from .test_joint_motion import FakeSession

    session = FakeSession(tmp_path)
    monkeypatch.setattr(
        m,
        "robot_state",
        lambda kin, arm: (
            session.q[arm].copy(),
            np.array([[-3.0, 3.0]] * 6),
            np.ones(6) * 3,
        ),
    )
    monkeypatch.setattr(m, "robot_velocity", lambda kin, arm: np.zeros(6))
    result = m.execute(session, "left", [point(0.031234), point(0.06)], 0, "test",
                       replace(m.DEFAULTS, arrival_m=1e-7))
    assert result["motion_result"]["status"] == "via_unverified"
    assert (
        result["motion_result"]["via_observations"][0]["closest_normalized_error"] > 1
    )
    assert abs(session.q["left"][0] - 0.06) < 0.0005
    assert len(result["motion_result"]["segments"]) == 1
    assert not any(t["stage"] == "settle" for t in result["motion_result"]["tracking"])
    with pytest.raises(ValueError, match="replay"):
        m.execute(session, "left", [point(0.03)], 0, "test")


def test_corner_corridor_and_invalid_request():
    corner = point(0.02)
    endpoint = point(0.02)
    endpoint["position"][1] = 0.02
    result = plan([corner, endpoint])
    xyz = result["dense_joints"][:, :3]
    # Distance to either finite polyline leg, independent of path parameter.
    leg1 = np.column_stack(
        [np.clip(xyz[:, 0], 0, 0.02), np.zeros(len(xyz)), np.zeros(len(xyz))]
    )
    leg2 = np.column_stack(
        [np.full(len(xyz), 0.02), np.clip(xyz[:, 1], 0, 0.02), np.zeros(len(xyz))]
    )
    distances = np.minimum(
        np.linalg.norm(xyz - leg1, axis=1), np.linalg.norm(xyz - leg2, axis=1)
    )
    assert distances.max() <= m.DEFAULTS.path_position_m
    malformed = point()
    malformed["quaternion_wxyz"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="pose"):
        m.validate_request("left", [malformed])
