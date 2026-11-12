"""Continuous Cartesian via-path and shared jerk-bounded S-curve execution.

Planning never advances physics. Native controller and interpolation are unchanged.
The quintic path time law is conservative, not a time-optimal seven-phase profile.
"""

from dataclasses import asdict, dataclass, replace
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from numpy.polynomial import Polynomial
from scipy.spatial.transform import Rotation

from ..io import write_json
from .action_edit_kinematics import pose, transform
from .self_collision import SelfCollisionError, check_with_diagnostics
from . import motion_distribution, motion_boundary

VERSION = "link6_relaxed_quintic_ack_soft_boundary_v1"


@dataclass(frozen=True)
class Limits:
    """Planning/safety caps in SI units; angular quantities use radians."""

    max_steps: int = 60
    knot_m: float = 0.02
    knot_rad: float = 0.10
    joint_velocity: float = 5.0  # simulation planning cap, not a hardware rating
    urdf_velocity_multiplier: float = 6.0
    joint_acceleration: float = 200.0  # explicit planning limit, not a hardware rating
    joint_jerk: float = 20000.0  # rad/s^3; prototype planning cap, not hardware rating
    linear_velocity: float = 2.0
    angular_velocity: float = 12.0
    joint_step: float = 0.20
    cartesian_step_m: float = 0.08
    cartesian_step_rad: float = 0.48
    ik_position_m: float = 0.0005
    ik_angle_rad: float = 0.005
    path_position_m: float = 0.002
    path_angle_rad: float = 0.02
    ik_joint_jump: float = 0.35
    arrival_m: float = 0.005
    arrival_rad: float = 0.05
    tracking_m: float = 0.08
    tracking_rad: float = 0.60
    tracking_joint_rad: float = 0.60
    held_arm_m: float = 0.03
    held_arm_rad: float = 0.20


NATIVE_JOINT_VELOCITY = 5.0  # X5 velocity_limit_sim; the controller is unchanged.
DEFAULTS = Limits()


class PlanningError(ValueError):
    """Expected infeasibility: return observation, never replay or poison RPC."""


def pose_error(actual, target):
    """Return position norm (meters) and SO(3) angle (radians) for two poses."""
    return (
        float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()),
    )


def reference(start, end, fraction):
    """Straight translation and shortest-arc SLERP, including zero rotation."""
    out = np.eye(4)
    out[:3, 3] = start[:3, 3] + fraction * (end[:3, 3] - start[:3, 3])
    rv = Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).as_rotvec()
    out[:3, :3] = Rotation.from_rotvec(fraction * rv).as_matrix() @ start[:3, :3]
    return out


def solve_pose(fk, root, goal, seed, bounds, limits=DEFAULTS):
    """Solve one pose with converged, previous-solution-seeded bounded IK.

    Args:
        fk: Robot-only forward kinematics exposing matrix(joints).
        root: Environment-origin to robot-base 4x4 transform.
        goal: Environment-origin to link6 4x4 target transform.
        seed: Previous joint solution, in radians.
        bounds: Per-joint lower/upper position bounds, in radians.
        limits: Solver tolerances and local joint trust-region radius.

    Returns:
        Converged joint position array in radians.

    Raises:
        PlanningError: Invalid bounds or local IK cannot reach the target.
    """

    def residual(q):
        actual = root @ fk.matrix(q)
        return np.r_[
            (actual[:3, 3] - goal[:3, 3]) / limits.ik_position_m,
            Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).as_rotvec()
            / limits.ik_angle_rad,
        ]

    lower = np.maximum(bounds[:, 0], seed - limits.ik_joint_jump)
    upper = np.minimum(bounds[:, 1], seed + limits.ik_joint_jump)
    if np.any(lower >= upper) or np.any(seed < lower) or np.any(seed > upper):
        raise PlanningError("Invalid current joint position/limits")
    solution = least_squares(
        residual,
        seed,
        bounds=(lower, upper),
        method="trf",
        max_nfev=120,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    q = solution.x
    dp, dr = pose_error(root @ fk.matrix(q), goal)
    if (
        not np.isfinite(q).all()
        or dp > limits.ik_position_m
        or dr > limits.ik_angle_rad
    ):
        raise PlanningError(
            f"IK unreachable or local solve failed: position={dp:.6g}, angle={dr:.6g}"
        )
    return q


def _extrema(poly, lower=0.0, upper=1.0):
    """Evaluate endpoints and real derivative roots of a scalar Polynomial."""
    roots = poly.deriv().roots()
    points = [lower, upper] + [
        float(r.real) for r in roots if abs(r.imag) < 1e-7 and lower < r.real < upper
    ]
    values = poly(np.asarray(points))
    return float(np.min(values)), float(np.max(values))


def _time_law(t):
    return motion_boundary.time_law(t)


def _continuous_bounds(path):
    """Return normalized rest-boundary peaks for an initial duration estimate."""
    zeros = np.zeros(path.c.shape[-1])
    return motion_boundary.BoundaryPath(path).bounds(1.0, zeros, zeros)[1]


def _plan_path(fk, root, q0, bounds, velocity, waypoints, dt, limits=DEFAULTS,
              *, distribution_profile=None, start_velocity=None, start_acceleration=None,
              end_velocity=None, end_acceleration=None, minimum_steps=1, _geometry_cache=None):
    """Jointly plan all pose vias with a shared jerk-bounded quintic S time law.

    No stop is inserted at an ordinary via. Sharp reversals may necessarily have
    zero instantaneous velocity. Geometry uses C2 splines and bounded numerical
    corridor checks, not exact swept-volume collision certification. There is no
    TOPPRA, per-joint time law, online retiming, or measured-progress replanning.

    Args:
        fk: Robot-only forward kinematics exposing matrix(joints).
        root: Environment-origin to robot-base 4x4 transform.
        q0: Measured initial joint positions in radians.
        bounds: Per-joint lower/upper position bounds in radians.
        velocity: Per-joint URDF velocity caps in radians/second.
        waypoints: Ordered link6 pose dictionaries in meters and unit wxyz.
        dt: Native action sampling period in seconds.
        limits: Explicit geometry, timing, and safety limits.
        distribution_profile: Optional per-task/arm EE rate quantile preferences.
        start_velocity: Inherited joint reference velocity in rad/s; initial zero
            when no preceding action exists; independently chosen end defaults to zero.
        start_acceleration: Inherited joint reference acceleration in rad/s^2;
            initial zero when unavailable; independently chosen end defaults to zero.

    Returns:
        Dictionary containing executable joint targets, duration, sample counts,
        final pose, dense geometry-check joints, via sample indices, and numerical
        continuous reference velocity/acceleration/jerk maxima in SI units.
        The initial target is excluded from executable targets.

    Raises:
        PlanningError: Geometry, IK, dynamics, or the movement budget is infeasible.
    """
    q0, bounds = np.asarray(q0, float), np.asarray(bounds, float)
    velocity = np.minimum(np.asarray(velocity, float) * limits.urdf_velocity_multiplier, min(limits.joint_velocity, NATIVE_JOINT_VELOCITY))
    positive = [
        dt,
        limits.joint_velocity,
        limits.urdf_velocity_multiplier,
        limits.joint_step,
        limits.cartesian_step_m,
        limits.cartesian_step_rad,
        limits.joint_acceleration,
        limits.joint_jerk,
        limits.linear_velocity,
        limits.angular_velocity,
    ]
    if (
        not np.isfinite(positive).all()
        or min(positive) <= 0
        or not np.isfinite(velocity).all()
        or np.any(velocity <= 0)
    ):
        raise PlanningError("Invalid timing or dynamic limits")
    if (
        q0.ndim != 1
        or bounds.shape != (len(q0), 2)
        or not np.isfinite(q0).all()
        or np.any(q0 < bounds[:, 0])
        or np.any(q0 > bounds[:, 1])
    ):
        raise PlanningError("Invalid current joint position/limits")
    start_velocity = np.zeros_like(q0) if start_velocity is None else np.asarray(start_velocity, float)
    start_acceleration = np.zeros_like(q0) if start_acceleration is None else np.asarray(start_acceleration, float)
    end_velocity = np.zeros_like(q0) if end_velocity is None else np.asarray(end_velocity, float)
    end_acceleration = np.zeros_like(q0) if end_acceleration is None else np.asarray(end_acceleration, float)
    if (start_velocity.shape != q0.shape or start_acceleration.shape != q0.shape
            or not np.isfinite(start_velocity).all() or not np.isfinite(start_acceleration).all()
            or end_velocity.shape != q0.shape or end_acceleration.shape != q0.shape
            or not np.isfinite(end_velocity).all() or not np.isfinite(end_acceleration).all()):
        raise PlanningError("Invalid boundary derivative vectors")
    moving_boundary = bool(np.any(start_velocity != 0) or np.any(start_acceleration != 0))
    start = root @ fk.matrix(q0)
    poses = [start]
    # Duplicate poses retain receipt identity but add no artificial path segment.
    via_indices = []
    for waypoint in waypoints:
        target = transform(waypoint["position"], waypoint["quaternion_wxyz"])
        if max(pose_error(poses[-1], target)) > 1e-10:
            poses.append(target)
        via_indices.append(len(poses) - 1)
    if len(poses) == 1:
        if moving_boundary:
            raise PlanningError("Stationary path cannot preserve nonzero boundary derivatives")
        return dict(
            joints=np.empty((0, len(q0))),
            duration=0.0,
            planning_knots=1,
            execution_steps=0,
            endpoint=poses[-1],
            dense_joints=q0[None],
            via_steps=[0] * len(waypoints),
            derivative_peaks=np.zeros((3, len(q0))),
            distribution_fit=dict(status="stationary_path"),
            joint_velocities=np.empty((0, len(q0))),
            joint_accelerations=np.empty((0, len(q0))),
            start_velocity=start_velocity.copy(), end_velocity=end_velocity.copy(),
            start_acceleration=start_acceleration.copy(), end_acceleration=end_acceleration.copy(),
        )
    if _geometry_cache:
        lengths = _geometry_cache["lengths"]
        via_parameters = _geometry_cache["via_parameters"]
        start_rotation = _geometry_cache["start_rotation"]
        knot_parameters = _geometry_cache["knot_parameters"]
        knots = _geometry_cache["knots"]
        path = _geometry_cache["path"]
        samples = _geometry_cache["samples"]
        corridor = _geometry_cache["corridor"]
    else:
        lengths = [
            max(
                pose_error(a, b)[0] / limits.knot_m,
                pose_error(a, b)[1] / limits.knot_rad,
                1e-6,
            )
            for a, b in zip(poses[:-1], poses[1:])
        ]
        via_parameters = np.r_[0.0, np.cumsum(lengths)] / sum(lengths)
        start_rotation = start[:3, :3]
        for refinement in range(5):
            knot_parameters, pose_coordinates = [0.0], [np.r_[start[:3, 3], np.zeros(3)]]
            for i, (a, b) in enumerate(zip(poses[:-1], poses[1:])):
                subdivisions = max(2, math.ceil(lengths[i])) * 2**refinement
                for alpha in np.linspace(0, 1, subdivisions + 1)[1:]:
                    goal = reference(a, b, alpha)
                    knot_parameters.append(
                        via_parameters[i]
                        + alpha * (via_parameters[i + 1] - via_parameters[i])
                    )
                    pose_coordinates.append(
                        np.r_[
                            goal[:3, 3],
                            Rotation.from_matrix(
                                goal[:3, :3] @ start_rotation.T
                            ).as_rotvec(),
                        ]
                    )
            knot_parameters = np.asarray(knot_parameters)
            geometry = CubicSpline(knot_parameters, pose_coordinates, bc_type="natural")

            def desired(s):
                value = geometry(s)
                goal = np.eye(4)
                goal[:3, 3] = value[:3]
                goal[:3, :3] = Rotation.from_rotvec(value[3:]).as_matrix() @ start_rotation
                return goal

            def corridor(s):
                i = min(
                    len(poses) - 2, np.searchsorted(via_parameters, s, side="right") - 1
                )
                return reference(
                    poses[i],
                    poses[i + 1],
                    (s - via_parameters[i]) / (via_parameters[i + 1] - via_parameters[i]),
                )

            samples = np.unique(
                np.r_[knot_parameters, np.linspace(0, 1, len(knot_parameters) * 8)]
            )
            if any(
                dp > limits.path_position_m or dr > limits.path_angle_rad
                for dp, dr in (pose_error(desired(s), corridor(s)) for s in samples)
            ):
                continue
            knots = [q0]
            try:
                for s in knot_parameters[1:]:
                    knots.append(
                        solve_pose(fk, root, desired(s), knots[-1], bounds, limits)
                    )
            except PlanningError:
                if refinement == 4:
                    raise
                continue
            path = CubicSpline(knot_parameters, knots, bc_type="natural")
            within_joint_bounds = True
            for i in range(len(knot_parameters) - 1):
                for j in range(len(q0)):
                    lo, hi = _extrema(
                        Polynomial(path.c[:, i, j][::-1]),
                        0.0,
                        knot_parameters[i + 1] - knot_parameters[i],
                    )
                    within_joint_bounds &= lo >= bounds[j, 0] and hi <= bounds[j, 1]
            if not within_joint_bounds:
                continue
            if any(
                dp > limits.path_position_m or dr > limits.path_angle_rad
                for dp, dr in (
                    pose_error(root @ fk.matrix(path(s)), corridor(s)) for s in samples
                )
            ):
                continue
            break
        else:
            raise PlanningError(
                "Continuous path violates Cartesian corridor or joint limits"
            )
        if _geometry_cache is not None:
            _geometry_cache.update(lengths=lengths, via_parameters=via_parameters, start_rotation=start_rotation, knot_parameters=knot_parameters, knots=knots, path=path, samples=samples, corridor=corridor)
    timed_path = motion_boundary.BoundaryPath(path)
    _, peaks = timed_path.bounds(1.0, np.zeros_like(q0), np.zeros_like(q0))
    duration = (
        max(
            float(np.max(peaks[0] / velocity)),
            float(np.sqrt(np.max(peaks[1] / limits.joint_acceleration))),
            float(np.cbrt(np.max(peaks[2] / limits.joint_jerk))),
            dt,
        )
        * 1.001
    )
    # EE speed is additionally checked at dense, explicitly recorded resolution.
    probe_times = np.linspace(0, 1, max(1001, len(knot_parameters) * 16))
    probe_joints = path(_time_law(probe_times))
    probe_poses = [root @ fk.matrix(q) for q in probe_joints]
    pose_deltas = np.asarray(
        [pose_error(a, b) for a, b in zip(probe_poses[:-1], probe_poses[1:])]
    )
    duration = max(
        duration,
        float(np.max(pose_deltas[:, 0] / np.diff(probe_times)))
        / limits.linear_velocity
        * 1.001,
        float(np.max(pose_deltas[:, 1] / np.diff(probe_times)))
        / limits.angular_velocity
        * 1.001,
    )
    sample_cache = {}

    def sampled(steps):
        if steps not in sample_cache:
            times = np.arange(steps + 1) / steps
            qs = timed_path.sample(times, steps * dt, start_velocity, start_acceleration, end_velocity=end_velocity, end_acceleration=end_acceleration)
            sample_cache[steps] = (times, qs, [root @ fk.matrix(q) for q in qs])
        return sample_cache[steps]

    def sampled_geometry_ratio(steps):
        sample_times, joints, sample_poses = sampled(steps)
        pose_deltas = np.asarray(
            [pose_error(a, b) for a, b in zip(sample_poses[:-1], sample_poses[1:])]
        )
        ratio = max(
            float(np.max(np.abs(np.diff(joints, axis=0))) / limits.joint_step),
            float(np.max(pose_deltas[:, 0]) / limits.cartesian_step_m),
            float(np.max(pose_deltas[:, 1]) / limits.cartesian_step_rad),
        )
        for i in range(steps):
            for alpha in np.arange(1, 10) / 9:
                native = root @ fk.matrix(
                    (1 - alpha) * joints[i] + alpha * joints[i + 1]
                )
                # Nominal native reference assumes prior measured q == reference.
                # Actual measured-start chords are checked again before every send.
                dp, dr = pose_error(
                    native,
                    corridor(
                        (1 - alpha) * _time_law(sample_times[i])
                        + alpha * _time_law(sample_times[i + 1])
                    ),
                )
                ratio = max(
                    ratio,
                    math.sqrt(dp / limits.path_position_m),
                    math.sqrt(dr / limits.path_angle_rad),
                )
        return ratio

    feasibility_cache = {}

    def feasible(candidate_steps):
        if candidate_steps in feasibility_cache:
            return feasibility_cache[candidate_steps]["valid"]
        seconds = candidate_steps * dt
        entry = dict(valid=False)
        feasibility_cache[candidate_steps] = entry
        candidate_q = timed_path.sample(probe_times, seconds, start_velocity, start_acceleration, end_velocity=end_velocity, end_acceleration=end_acceleration)
        # Cheap dense rejection first; passing samples still requires polynomial extrema.
        caps = (velocity, limits.joint_acceleration, limits.joint_jerk)
        for order, cap in enumerate(caps, 1):
            derivative = timed_path.sample(probe_times, seconds, start_velocity, start_acceleration, order, end_velocity=end_velocity, end_acceleration=end_acceleration)
            if np.any(np.abs(derivative) > np.asarray(cap) * (1 + 1e-8)):
                return False
        position_bounds, candidate_peaks = timed_path.bounds(seconds, start_velocity, start_acceleration, end_velocity=end_velocity, end_acceleration=end_acceleration)
        if (np.any(position_bounds[0] < bounds[:, 0] - 1e-10)
                or np.any(position_bounds[1] > bounds[:, 1] + 1e-10)
                or any(np.any(candidate_peaks[i] > np.asarray(cap) * (1 + 1e-8))
                       for i, cap in enumerate(caps))):
            return False
        candidate_poses = [root @ fk.matrix(q) for q in candidate_q]
        for t, actual in zip(probe_times, candidate_poses):
            dp, dr = pose_error(actual, corridor(_time_law(t)))
            if dp > limits.path_position_m or dr > limits.path_angle_rad:
                return False
        rates = np.asarray([pose_error(a, b) for a, b in zip(candidate_poses[:-1], candidate_poses[1:])])
        rates /= (np.diff(probe_times) * seconds)[:, None]
        if (np.any(rates[:, 0] > limits.linear_velocity)
                or np.any(rates[:, 1] > limits.angular_velocity)
                or sampled_geometry_ratio(candidate_steps) > 1.000001):
            return False
        entry.update(valid=True, dense_joints=candidate_q, derivative_peaks=candidate_peaks)
        return True

    # Fixed physical derivatives do not scale down with duration. Search ticks
    # explicitly instead of assuming longer trajectories are always feasible.
    minimum_steps = max(minimum_steps, (max(1, math.ceil(float(np.max(np.abs(knots[-1] - q0) / velocity)) / dt))
                     if moving_boundary else max(1, math.ceil(duration / dt))))
    for steps in range(minimum_steps, limits.max_steps + 1):
        if feasible(steps):
            break
    else:
        raise PlanningError("No feasible inherited-boundary trajectory within step budget and hard constraints")
    distribution_fit = dict(status="not_requested")
    if distribution_profile is not None:
        ranked, distribution_fit = motion_distribution.rank_durations(
            distribution_profile, probe_poses, lambda n: sampled(n)[2],
            steps, limits.max_steps, dt,
        )
        baseline_steps = steps
        rejected = []
        for candidate in ranked:
            proposed = candidate["steps"]
            if proposed != baseline_steps and not feasible(proposed):
                rejected.append(proposed)
                continue
            steps = proposed
            distribution_fit["selected"] = candidate
            if candidate["score"] is not None:
                distribution_fit["status"] = (
                    "within_soft_tolerance" if candidate["score"] <= 1e-12
                    else "closest_under_constraints"
                )
            break
        distribution_fit["rejected_geometry_steps"] = rejected
        distribution_fit["minimum_hard_feasible_steps"] = baseline_steps
        distribution_fit["maximum_budget_steps"] = limits.max_steps
        distribution_fit["duration_changed"] = steps != baseline_steps
    sample_times, joints, sample_poses = sampled(steps)
    duration = steps * dt
    joint_velocities = timed_path.sample(sample_times, duration, start_velocity, start_acceleration, 1, end_velocity=end_velocity, end_acceleration=end_acceleration)
    joint_accelerations = timed_path.sample(sample_times, duration, start_velocity, start_acceleration, 2, end_velocity=end_velocity, end_acceleration=end_acceleration)
    if (not np.allclose(joint_velocities[[0, -1]], [start_velocity, end_velocity], rtol=0, atol=1e-8)
            or not np.allclose(joint_accelerations[[0, -1]], [start_acceleration, end_acceleration], rtol=0, atol=1e-7)):
        raise PlanningError("Boundary derivative verification failed")
    # Preserve the exact requested vectors after checking numerical evaluation.
    joint_velocities[[0, -1]] = [start_velocity, end_velocity]
    joint_accelerations[[0, -1]] = [start_acceleration, end_acceleration]
    via_steps = [
        int(np.searchsorted(_time_law(sample_times), via_parameters[i]))
        for i in via_indices
    ]
    return dict(
        joints=joints[1:],
        duration=duration,
        execution_steps=steps,
        planning_knots=len(knot_parameters),
        endpoint=poses[-1],
        dense_joints=feasibility_cache[steps]["dense_joints"],
        via_steps=via_steps,
        derivative_peaks=feasibility_cache[steps]["derivative_peaks"],
        joint_velocities=joint_velocities[1:],
        joint_accelerations=joint_accelerations[1:],
        start_velocity=start_velocity.copy(), end_velocity=end_velocity.copy(),
        start_acceleration=start_acceleration.copy(), end_acceleration=end_acceleration.copy(),
        geometry_check_samples=len(samples),
        ee_speed_check_samples=len(probe_times),
        distribution_fit=distribution_fit,
    )


def plan_path(fk, root, q0, bounds, velocity, waypoints, dt, limits=DEFAULTS,
              *, distribution_profile=None, start_velocity=None, start_acceleration=None,
              end_velocity=None, end_acceleration=None, minimum_steps=1):
    """Try the ACK boundary, then explicit soft recovery within unchanged hard geometry.

    Soft recovery changes only effective planning boundary derivatives. It never
    edits ACK history and does not establish strict physical or command continuity.
    The independent default terminal boundary is rest, so chunks may decelerate.
    """
    inherited_v = np.zeros_like(q0, dtype=float) if start_velocity is None else np.asarray(start_velocity, float)
    inherited_a = np.zeros_like(q0, dtype=float) if start_acceleration is None else np.asarray(start_acceleration, float)
    if (inherited_v.shape != np.asarray(q0).shape or inherited_a.shape != np.asarray(q0).shape
            or not np.isfinite(inherited_v).all() or not np.isfinite(inherited_a).all()):
        raise PlanningError("Invalid acknowledged boundary derivatives")
    attempts = []
    geometry_cache = {}
    scales = (1.0,) if not np.any(inherited_v) and not np.any(inherited_a) else (1.0, .25, .05, .01, 0.0)
    for scale in scales:
        try:
            segment = _plan_path(
                fk, root, q0, bounds, velocity, waypoints, dt, limits,
                distribution_profile=distribution_profile,
                start_velocity=inherited_v * scale, start_acceleration=inherited_a * scale,
                end_velocity=end_velocity, end_acceleration=end_acceleration, minimum_steps=minimum_steps,
                _geometry_cache=geometry_cache)
        except PlanningError as exc:
            attempts.append(dict(scale=scale, error=str(exc)))
            # Geometry and IK do not depend on boundary scaling.
            if "IK " in str(exc) or "Cartesian corridor" in str(exc) or "Invalid" in str(exc):
                raise
            continue
        segment["boundary_recovery"] = dict(
            inherited_velocity=inherited_v.tolist(), inherited_acceleration=inherited_a.tolist(),
            effective_start_velocity=segment["start_velocity"].tolist(),
            effective_start_acceleration=segment["start_acceleration"].tolist(), scale=scale,
            rejected_attempts=attempts, strict_continuity_guaranteed=False,
            terminal_boundary="independent_rest" if end_velocity is None and end_acceleration is None else "explicit")
        return segment
    raise PlanningError(f"No feasible ACK/soft-recovery quintic within budget: {attempts}")


def robot_state(kinematics, arm):
    """Return measured joints, native position bounds, and URDF velocity caps.

    Args:
        kinematics: Simulator-local DualKinematics instance.
        arm: "left" or "right".

    Returns:
        Tuple of joint radians, Nx2 radian limits, and radian/second limits.
    """
    robot = kinematics.robots[arm]
    key = kinematics.manager.robot_key[kinematics.manager.robot_list.index(robot)]
    q = np.asarray(kinematics.manager.get_joint(robot)[0], float)
    bounds = key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy()
    xml = {
        j.get("name"): j for j in ET.parse(robot.urdf_path).getroot().findall("joint")
    }
    velocity = [
        float(xml[n].find("limit").get("velocity")) for n in kinematics.fk[arm].names
    ]
    return q, bounds, velocity


def robot_velocity(kinematics, arm):
    """Read actual native arm velocity in radians/second."""
    robot = kinematics.robots[arm]
    key = kinematics.manager.robot_key[kinematics.manager.robot_list.index(robot)]
    return key.data.joint_vel[0, robot.arm_joint_indices].cpu().numpy()


def validate_request(arm, waypoints):
    """Validate pose and synchronous gripper syntax without advancing physics.

    Args:
        arm: "left" or "right".
        waypoints: One to five pose/gripper dictionaries in the public tool schema.

    Raises:
        ValueError: An arm, pose, or gripper-command field is invalid.
    """
    if (
        arm not in ("left", "right")
        or not isinstance(waypoints, list)
        or not 1 <= len(waypoints) <= 5
    ):
        raise ValueError("Expected arm and 1..5 waypoints")
    for p in waypoints:
        if set(p) != {
            "position",
            "quaternion_wxyz",
            "gripper",
        }:
            raise ValueError("Invalid waypoint fields")
        xyz, quat = (
            np.asarray(p["position"], float),
            np.asarray(p["quaternion_wxyz"], float),
        )
        if (
            xyz.shape != (3,)
            or quat.shape != (4,)
            or not np.isfinite(np.r_[xyz, quat]).all()
            or abs(np.linalg.norm(quat) - 1) > 1e-4
        ):
            raise ValueError("Invalid waypoint pose")
        if p["gripper"] not in ("keep", "open", "close"):
            raise ValueError("Unknown gripper command")


def gripper_schedule(waypoints, via_steps, execution_steps, initial_opening):
    """Resolve synchronous gripper targets on the existing 25 Hz action grid.

    Args:
        waypoints: Ordered pose dictionaries with keep/open/close commands.
        via_steps: First action sample at or beyond each reference via.
        execution_steps: Number of planned joint actions (possibly zero).
        initial_opening: Observed opening at macro start, in [0, 1].

    Returns:
        Opening targets and associated waypoint indices for each joint action.
        A stationary explicit gripper command requires exactly one joint action.

    Raises:
        PlanningError: Two commands require conflicting openings in one tick.
    """
    if not np.isfinite(initial_opening) or not 0 <= initial_opening <= 1:
        raise PlanningError("Invalid initial gripper opening")
    count = execution_steps
    if count == 0 and any(point["gripper"] != "keep" for point in waypoints):
        count = 1
    openings = np.full(count, np.nan)
    owners = np.full(count, -1, dtype=int)
    opening = float(initial_opening)
    previous_end = 0
    for index, (point, end) in enumerate(zip(waypoints, via_steps)):
        if point["gripper"] != "keep":
            opening = float(point["gripper"] == "open")
        if count == 0:
            continue
        if end == previous_end and point["gripper"] == "keep":
            # No new target to issue on a zero-length interval.
            continue
        # A zero-length interval still has an explicit command obligation. Map
        # it to its boundary tick, never invent a hold or silently erase a change.
        ticks = (
            range(previous_end, end)
            if end > previous_end
            else [max(0, min(count - 1, end - 1))]
        )
        for tick in ticks:
            if np.isfinite(openings[tick]) and openings[tick] != opening:
                raise PlanningError(
                    f"Conflicting gripper commands at action sample {tick + 1}; "
                    "give the poses enough path interval for separate samples"
                )
            openings[tick] = opening
            if owners[tick] < 0:
                owners[tick] = index
        previous_end = end
    if count and (not np.isfinite(openings).all() or np.any(owners < 0)):
        raise PlanningError("Incomplete gripper action schedule")
    return openings, owners


def _execute_segment(session, arm, waypoints, decision, teacher_request_id, limits=DEFAULTS,
                     *, _directory=None):
    """Execute one serial continuous-motion chunk with native safety receipts.

    Args:
        session: Live simulator-local session; owns physics and observations.
        arm: "left" or "right"; the other arm keeps its measured joint targets.
        waypoints: Validated link6 pose/gripper dictionaries.
        decision: Nonnegative decision index, also used for replay protection.
        teacher_request_id: Request identifier attached to every native step.
        limits: Planning, execution, settling, and safety caps.

    Returns:
        Native step rows and a motion_result receipt distinguishing reference
        progress, observed via arrival, final arrival, and measured velocity without imposing a stop.

    Raises:
        ValueError: Invalid/replayed request or already-ended episode.
        RuntimeError: Unexpected session execution error; partial receipt is saved.
    """
    validate_request(arm, waypoints)
    profile, profile_receipt = motion_distribution.task_profile(
        session.metadata.get("task"), arm, session.metadata["control_dt"]
    )
    if type(decision) is not int or decision < 0:
        raise ValueError("Invalid motion decision")
    if session.finished or session.terminated or session.truncated:
        raise ValueError("Episode already ended")
    kin = session.kinematics
    kin.check()
    starts = {
        a: kin.root(a) @ kin.fk[a].matrix(robot_state(kin, a)[0])
        for a in ("left", "right")
    }
    last = starts[arm]
    distance = angle = 0.0
    for p in waypoints:
        goal = transform(p["position"], p["quaternion_wxyz"])
        d, r = pose_error(last, goal)
        distance += d
        angle += r
        last = goal
    if distance > 0.205 or angle > 1.01:
        raise ValueError("Motion exceeds total path limit from measured start")
    directory = _directory if _directory is not None else session.episode_dir / "motion"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{decision:03d}.json"
    if path.exists():
        raise ValueError("Motion decision already exists; refusing replay")
    report = dict(
        version=VERSION,
        status="running",
        arm=arm,
        waypoints=waypoints,
        executed_steps=0,
        completed_waypoints=0,
        active_waypoint=None,
        gripper_commands=[],
        tracking=[],
        segments=[],
        parameters=asdict(limits),
        via_observations=[
            dict(
                waypoint=i,
                observed=False,
                closest_normalized_error=None,
                position_error_m=None,
                rotation_error_rad=None,
            )
            for i in range(len(waypoints))
        ],
        via_observation_frequency_hz=1 / session.metadata["control_dt"],
        control_dt=session.metadata["control_dt"],
        privileged_planning=False,
        ik_backend="scipy_least_squares_trf",
        time_parameterizer="shared_quintic_independent_end_soft_ack_v1",
        distribution_reference=profile_receipt,
    )
    rows = []
    action = np.asarray(session.obs["states"], float).copy()
    for a, offset in (("left", 0), ("right", 7)):
        if not np.isfinite(action[offset + 6]) or not 0 <= action[offset + 6] <= 1:
            raise ValueError(f"Invalid observed {a} gripper opening")
        action[offset : offset + 6] = robot_state(kin, a)[0]
    from .gripper_hold import issued_openings
    initial_openings = issued_openings(session)
    for a, off in (("left", 0), ("right", 7)):
        action[off + 6] = initial_openings[a]
    offset = 0 if arm == "left" else 7
    other = "right" if arm == "left" else "left"
    candidate_path = directory / f"{decision:03d}_00_candidate.npz"
    max_steps = min(
        limits.max_steps, session.metadata["max_episode_steps"] - session.step_id
    )

    def save():
        report["executed_steps"] = len(rows)
        write_json(path, report)

    def measured(a):
        i = 0 if a == "left" else 1
        return transform(
            session.obs["eef_positions"][i], session.obs["eef_quaternions_wxyz"][i]
        )

    def observe_vias():
        actual = measured(arm)
        for i, p in enumerate(waypoints):
            dp, dr = pose_error(actual, transform(p["position"], p["quaternion_wxyz"]))
            normalized = max(dp / limits.arrival_m, dr / limits.arrival_rad)
            evidence = report["via_observations"][i]
            if (
                evidence["closest_normalized_error"] is None
                or normalized < evidence["closest_normalized_error"]
            ):
                evidence.update(
                    closest_normalized_error=normalized,
                    position_error_m=dp,
                    rotation_error_rad=dr,
                    closest_step_id=session.step_id,
                )
            # Ordered observation: a later point cannot be verified before all
            # earlier points. Crossing between 25Hz samples is not proof.
            if i == report["completed_waypoints"] and normalized <= 1:
                evidence.update(observed=True, observed_step_id=session.step_id)
                report["completed_waypoints"] += 1

    def send(q, stage):
        if len(rows) >= max_steps:
            report["status"] = "step_budget"
            return False
        action[offset : offset + 6] = q
        actual_q = robot_state(kin, arm)[0]
        pre_send_error = float(np.max(np.abs(q - actual_q)))
        report.setdefault("pre_send_tracking", []).append(dict(
            step_id=session.step_id + 1, joint_error_rad=pre_send_error,
            original_tracking_would_stop=bool(pre_send_error > .12)))
        if pre_send_error > limits.tracking_joint_rad:
            report["status"] = "joint_tracking_error"
            return False
        try:
            for a, off in (("left", 0), ("right", 7)):
                check_with_diagnostics(kin, a,
                    np.vstack([robot_state(kin, a)[0], action[off : off + 6]]),
                    report, 'actual_start_command_chord', candidate_path,
                    next_step_id=session.step_id + 1, execution_index=len(rows))
        except SelfCollisionError as exc:
            report.update(status="self_collision", error=str(exc))
            return False
        reference_pose = kin.root(arm) @ kin.fk[arm].matrix(q)
        current_pose = kin.root(arm) @ kin.fk[arm].matrix(actual_q)
        # Local actual-start-to-command chord only; this does not establish
        # a 2mm global physical tracking corridor in the presence of PD lag.
        for alpha in np.arange(1, 10) / 9:
            native_pose = kin.root(arm) @ kin.fk[arm].matrix(
                (1 - alpha) * actual_q + alpha * q
            )
            dp, dr = pose_error(
                native_pose, reference(current_pose, reference_pose, alpha)
            )
            if dp > limits.path_position_m or dr > limits.path_angle_rad:
                report["status"] = "native_chord_geometry_error"
                return False
        target = {}
        for a, off in (("left", 0), ("right", 7)):
            target[a] = dict(
                pose(kin.root(a) @ kin.fk[a].matrix(action[off : off + 6])),
                gripper_closed=bool(action[off + 6] < 0.5),
                gripper_opening=float(action[off + 6]),
            )
        command_difference = motion_boundary.command_differences(session, arm, q)
        report.setdefault("command_boundary_differences", []).append(command_difference)
        result = session.chunk_step(
            [action.copy()], teacher_request_id=teacher_request_id
        )
        command_difference["acknowledged"] = bool(result["steps"])
        command_difference["proposed_not_acknowledged"] = not bool(result["steps"])
        for row in result["steps"]:
            row["edited_target"] = {a: dict(target[a],
                gripper_opening=float(row['executed_action'][off+6]),
                gripper_closed=bool(row['executed_action'][off+6] < .5))
                for a, off in (("left", 0), ("right", 7))}
            row["ik_diagnostics"] = dict(
                method=VERSION, stage=stage, physical_tracking_verified=False
            )
        for row in result["steps"]:
            report["gripper_commands"].append(
                dict(
                    step_id=row["step_id"],
                    opening=float(row["executed_action"][offset + 6]),
                    waypoint=report["active_waypoint"],
                    stage=stage,
                    verified="issued_action_only",
                )
            )
        rows.extend(result["steps"])
        observe_vias()
        dp, dr = pose_error(measured(arm), reference_pose)
        hd, hr = pose_error(measured(other), starts[other])
        dq = float(np.max(np.abs(robot_state(kin, arm)[0] - q)))
        report["tracking"].append(
            dict(
                step_id=session.step_id,
                stage=stage,
                position_error_m=dp,
                rotation_error_rad=dr,
                joint_error_rad=dq,
                original_tracking_would_stop=bool(dp > .025 or dr > .20 or dq > .12),
                held_arm_position_error_m=hd,
                held_arm_rotation_error_rad=hr,
            )
        )
        if session.terminated or session.truncated:
            report["status"] = "native_terminal"
        elif hd > limits.held_arm_m or hr > limits.held_arm_rad:
            report["status"] = "held_arm_tracking_error"
        elif (
            dp > limits.tracking_m
            or dr > limits.tracking_rad
            or dq > limits.tracking_joint_rad
        ):
            report["status"] = "tracking_error"
        save()
        return report["status"] == "running"

    save()
    try:
        q0, bounds, velocity = robot_state(kin, arm)
        inherited_velocity, inherited_acceleration, boundary_source = motion_boundary.initial_derivatives(session, arm)
        report["boundary_reference"] = dict(
            source=boundary_source, frame="joint", velocity_units="rad/s", acceleration_units="rad/s^2",
            inherited_velocity=inherited_velocity.tolist(),
            inherited_acceleration=inherited_acceleration.tolist(),
            position_anchor="measured_joints", actual_derivative_continuity_guaranteed=False,
        )
        if report["status"] == "running":
            began = time.monotonic()
            try:
                segment = plan_path(
                    kin.fk[arm],
                    kin.root(arm),
                    q0,
                    bounds,
                    velocity,
                    waypoints,
                    session.metadata["control_dt"],
                    replace(limits, max_steps=max_steps - len(rows)),
                    distribution_profile=profile,
                    start_velocity=inherited_velocity,
                    start_acceleration=inherited_acceleration,
                )
                report["boundary_reference"].update(segment["boundary_recovery"])
                report["boundary_reference"].update(end_velocity=segment["end_velocity"].tolist(), end_acceleration=segment["end_acceleration"].tolist())
                openings, gripper_waypoints = gripper_schedule(
                    waypoints,
                    segment["via_steps"],
                    segment["execution_steps"],
                    action[offset + 6],
                )
                if len(openings) and not segment["execution_steps"]:
                    segment["joints"] = q0[None].copy()
                    segment["execution_steps"] = 1
                    segment["duration"] = session.metadata["control_dt"]
                    segment["joint_velocities"] = np.zeros((1, len(q0)))
                    segment["joint_accelerations"] = np.zeros((1, len(q0)))
                if segment["execution_steps"] > max_steps - len(
                    rows
                ):
                    raise PlanningError(
                        "Trajectory exceeds remaining step budget"
                    )
                np.savez_compressed(candidate_path, joints=segment['joints'], start_joints=q0,
                    dense_joints=segment['dense_joints'], control_dt=session.metadata['control_dt'],
                    joint_velocities=segment['joint_velocities'],
                    joint_accelerations=segment['joint_accelerations'],
                    start_velocity=segment['start_velocity'], start_acceleration=segment['start_acceleration'],
                    end_velocity=segment['end_velocity'], end_acceleration=segment['end_acceleration'],
                    gripper_openings=openings, arm=np.asarray(arm),
                    candidate_collision_checked=np.asarray(False))
                report['candidate_plan_npz'] = str(candidate_path)
                collision = check_with_diagnostics(kin, arm, segment['dense_joints'],
                    report, 'planned_dense_joint_chords', candidate_path)
                check_with_diagnostics(kin, arm, np.vstack([q0, segment['joints']]),
                    report, 'planned_control_joint_chords', candidate_path)
            except (PlanningError, SelfCollisionError) as exc:
                report["segments"].append(
                    dict(planning_seconds=time.monotonic() - began, error=str(exc))
                )
                report.update(
                    status="self_collision"
                    if isinstance(exc, SelfCollisionError)
                    else "planning_failed",
                    error=str(exc),
                )
            else:
                np.savez_compressed(
                    directory / f"{decision:03d}_00_plan.npz",
                    joints=segment["joints"],
                    start_joints=q0,
                    dense_joints=segment["dense_joints"],
                    control_dt=session.metadata["control_dt"],
                    derivative_peaks=segment["derivative_peaks"],
                    joint_velocities=segment["joint_velocities"],
                    joint_accelerations=segment["joint_accelerations"],
                    start_velocity=segment["start_velocity"], end_velocity=segment["end_velocity"],
                    start_acceleration=segment["start_acceleration"], end_acceleration=segment["end_acceleration"],
                    gripper_openings=openings,
                    gripper_waypoint_indices=gripper_waypoints,
                )
                report["segments"].append(
                    dict(
                        planning_seconds=time.monotonic() - began,
                        boundary_recovery=segment["boundary_recovery"],
                        duration=segment["duration"],
                        planning_knots=segment["planning_knots"],
                        execution_steps=segment["execution_steps"],
                        self_collision=collision,
                        derivative_peaks=segment["derivative_peaks"].tolist(),
                        via_steps=segment["via_steps"],
                        continuous_collision_guarantee=False,
                        distribution_fit=segment["distribution_fit"],
                    )
                )
                observe_vias()
                save()
                for step, q in enumerate(segment["joints"], 1):
                    report["active_waypoint"] = int(gripper_waypoints[step - 1])
                    action[offset + 6] = openings[step - 1]
                    before_count = len(rows)
                    continuing = send(q, "move")
                    if len(rows) > before_count:
                        motion_boundary.remember_derivatives(
                            session, arm, segment["joint_velocities"][step - 1],
                            segment["joint_accelerations"][step - 1])
                    if not continuing:
                        break
                if report["status"] == "running":
                    goal = segment["endpoint"]
                    if report["status"] == "running":
                        dp, dr = pose_error(measured(arm), goal)
                        speed = float(np.max(np.abs(robot_velocity(kin, arm))))
                        report["final_joint_speed_rad_s"] = speed
                        report["final_arrival_verified"] = bool(
                            dp <= limits.arrival_m and dr <= limits.arrival_rad
                        )
                        report["final_velocity_tracking_error_rad_s"] = float(np.max(np.abs(
                            robot_velocity(kin, arm) - segment["end_velocity"])))
                        report["stationary_required"] = False
                        report["final_position_error_m"] = dp
                        report["final_rotation_error_rad"] = dr
                        if dp > limits.arrival_m or dr > limits.arrival_rad:
                            report["status"] = "arrival_not_reached"
                        elif report["completed_waypoints"] < len(waypoints):
                            report["status"] = "via_unverified"
                        else:
                            report.update(
                                status="waypoints_reached_not_grasp_verified",
                                active_waypoint=None,
                            )
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted"
        save()
    return dict(
        episode_id=session.episode_id,
        step_id=session.step_id,
        steps=rows,
        motion_result=report,
    )


def execute(session, arm, waypoints, decision, teacher_request_id, limits=DEFAULTS):
    """Execute GPT close commands only after their measured pose arrival."""
    validate_request(arm, waypoints)
    if type(decision) is not int or decision < 0:
        raise ValueError('Invalid motion decision')
    if session.finished or session.terminated or session.truncated:
        raise ValueError('Episode already ended')
    if not any(point['gripper'] == 'close' for point in waypoints):
        return _execute_segment(session, arm, waypoints, decision, teacher_request_id, limits)
    from .gripper_hold import execute_events
    return execute_events(session, 'motion', [{arm: point} for point in waypoints],
                          1, decision, teacher_request_id, limits, _execute_segment)
