"""Task-specific soft EE rate preferences; hard trajectory limits stay separate.

Durations are ranked; the caller rechecks boundary-dependent geometry and dynamics.
The data describe sampled command poses, not native PD dynamics or safety limits.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PROFILE_FILE = Path(__file__).with_name("motion_distribution_profiles.json")
VERSION = "task_ee_quantiles_v1"
METRICS = (
    "linear_velocity_norm", "angular_velocity_norm",
    "linear_acceleration_norm", "angular_acceleration_norm",
)
QUANTILES = (.5, .9)
RELATIVE_TOLERANCE = 1.25
MAX_CANDIDATES = 13


@lru_cache(maxsize=1)
def _catalog():
    data = PROFILE_FILE.read_bytes()
    catalog = json.loads(data)
    source = catalog["source"]
    if (catalog["schema"] != "ee_motion_distribution_v1"
            or source["fps"] != 25 or source["field"] != "action"
            or source["subset"] != "moving" or source["angular_units"] != "rad"
            or source["linear_units"] != "m" or source["quantiles"] != list(QUANTILES)):
        raise ValueError("Unsupported motion-distribution reference semantics")
    for task in catalog["tasks"].values():
        for arm in ("left", "right"):
            for metric in METRICS:
                row = task["arms"][arm][metric]
                values = [row["p50"], row["p90"]]
                if (not np.isfinite(values).all() or values[0] < 0
                        or values[1] < values[0] or row["count"] < 0):
                    raise ValueError("Invalid motion-distribution reference quantiles")
    return catalog, hashlib.sha256(data).hexdigest()


def task_profile(task, arm, dt):
    """Resolve an exact task/arm profile without substituting other tasks.

    Unknown tasks, including uncalibrated random-layout names, explicitly use
    hard-constraint-only timing. Corrupt or missing catalog files fail loudly.
    """
    if arm not in ("left", "right"):
        raise ValueError("Invalid profile arm")
    catalog, identity = _catalog()
    receipt = dict(version=VERSION, profile_sha256=identity, task=task, arm=arm,
                   reference_field="action", reference_subset="moving",
                   quantiles=list(QUANTILES), relative_tolerance=RELATIVE_TOLERANCE,
                   source_statistics_sha256=catalog["source"]["statistics_sha256"])
    if not np.isclose(dt, 1 / catalog["source"]["fps"], rtol=0, atol=1e-9):
        raise ValueError("Motion distribution reference requires native 25 Hz control")
    entry = catalog["tasks"].get(task)
    if entry is None:
        receipt.update(status="unavailable_task_profile", timing="hard_constraints_only")
        return None, receipt
    receipt.update(status="available", task_index=entry["task_index"],
                   reference_episode_count=len(entry["episode_ids"]))
    return entry["arms"][arm], receipt


def sampled_quantiles(poses, dt):
    """Estimate rate magnitudes using the same finite differences as the audit.

    Linear units are m/s and m/s^2. Angular units are rad/s and rad/s^2.
    Rotation increments and their derivatives are in the fixed environment frame.
    """
    poses = np.asarray(poses, dtype=float)
    if len(poses) < 3:
        return {}
    velocity = np.diff(poses[:, :3, 3], axis=0) / dt
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    omega = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt
    moving = (np.linalg.norm(velocity, axis=1) > .001) | (
        np.linalg.norm(omega, axis=1) > np.deg2rad(.5))
    accel_moving = moving[:-1] | moving[1:]
    values = (
        np.linalg.norm(velocity, axis=1)[moving],
        np.linalg.norm(omega, axis=1)[moving],
        np.linalg.norm(np.diff(velocity, axis=0) / dt, axis=1)[accel_moving],
        np.linalg.norm(np.diff(omega, axis=0) / dt, axis=1)[accel_moving],
    )
    return {metric: np.quantile(v, QUANTILES).tolist()
            for metric, v in zip(METRICS, values) if len(v)}


def applicable_metrics(poses):
    """Ignore intentionally fixed position/orientation instead of adding motion."""
    poses = np.asarray(poses, dtype=float)
    translation = np.max(np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1))
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    angle = np.max((rotations * rotations[0].inv()).magnitude())
    return [metric for metric in METRICS
            if (translation > .001 if metric.startswith("linear") else angle > np.deg2rad(.5))]


def distribution_score(statistics, profile, metrics):
    """Score relative quantile differences with a factor-1.25 dead band."""
    residuals, terms = [], []
    for metric in metrics:
        if metric not in statistics or profile[metric]["count"] == 0:
            continue
        for quantile, actual, key in zip(QUANTILES, statistics[metric], ("p50", "p90")):
            target = profile[metric][key]
            if target <= 1e-10:
                continue
            deviation = abs(float(np.log(max(actual, 1e-12) / target)))
            residuals.append(max(0., deviation - np.log(RELATIVE_TOLERANCE)) ** 2)
            terms.append(dict(metric=metric, quantile=quantile, reference=target, actual=actual))
    return (float(np.mean(residuals)) if residuals else None), terms


def rank_durations(profile, dense_poses, poses_at_steps, minimum_steps, maximum_steps, dt):
    """Rank a bounded, deterministic duration grid; never exceed hard timing caps.

    All candidate durations are at least the hard-feasible baseline. The caller
    must still validate dynamics and sampled/native geometry before adoption.
    """
    metrics = applicable_metrics(dense_poses)
    maximum_steps = max(minimum_steps, int(maximum_steps))
    grid = np.unique(np.r_[minimum_steps, np.rint(np.geomspace(
        minimum_steps, maximum_steps, MAX_CANDIDATES)).astype(int)])
    candidates = []
    for steps in grid:
        stats = sampled_quantiles(poses_at_steps(int(steps)), dt)
        score, terms = distribution_score(stats, profile, metrics)
        candidates.append(dict(steps=int(steps), duration=float(steps * dt),
                               score=score, quantiles=stats, terms=terms))
    baseline = candidates[0]
    # Unknown/zero reference dimensions cannot define an objective.
    if baseline["score"] is None:
        return [baseline], dict(status="no_applicable_reference", metrics=metrics,
                                 baseline=baseline, candidates=candidates)
    ranked = sorted((c for c in candidates if c["score"] is not None),
                    key=lambda c: (round(c["score"], 12), c["steps"]))
    return ranked, dict(status="ranked", metrics=metrics, baseline=baseline,
                        candidate_count=len(candidates), candidates=candidates)
