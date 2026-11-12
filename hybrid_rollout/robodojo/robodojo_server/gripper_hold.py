"""GPT pose events and measured-finger stopping evidence; never grasp proof.

The release RoboDojo preset specifies X5 measured-finger stability thresholds.
Retune these in a new environment; they are separate from automatic pi holding
and tightening, and remain active in the RoboDojo eval routes.
"""

from copy import deepcopy
from ..force_hold_archive import requested_action
from dataclasses import asdict, dataclass, replace
import json
import os

import numpy as np

from ..io import write_json
from .action_edit_kinematics import pose, transform
from . import motion_boundary
from .self_collision import SelfCollisionError, check_with_diagnostics

ARMS = ('left', 'right')
SUCCESS = {'waypoints_reached_not_grasp_verified', 'targets_reached_not_grasp_verified'}


@dataclass(frozen=True)
class HoldConfig:
    """Explicit experiment settings; no unmeasured runtime defaults."""

    min_steps: int
    stable_transitions: int
    max_steps: int
    delta_m: float

    def __post_init__(self):
        if (any(type(value) is not int for value in
                (self.min_steps, self.stable_transitions, self.max_steps))
                or not 1 <= self.stable_transitions < self.min_steps <= self.max_steps
                or isinstance(self.delta_m, bool)
                or not np.isfinite(self.delta_m) or self.delta_m <= 0):
            raise ValueError('Invalid measured-finger hold configuration')


# User-selected fast rule: the first post-close ACK is a baseline, so two
# differences naturally need three samples; there is no extra fixed wait.
# max_steps is a failure timeout, never a mandatory duration.
DEFAULTS = HoldConfig(min_steps=3, stable_transitions=2, max_steps=8, delta_m=.0001)


def configured(value=None):
    """Resolve one measured default or an explicit experiment override."""
    if value is None:
        raw = os.environ.get('ROBODOJO_GRIPPER_HOLD_CONFIG')
        value = json.loads(raw) if raw else DEFAULTS
    return value if isinstance(value, HoldConfig) else HoldConfig(**value)


def issued_openings(session):
    """Read last genuine ACK commands, never the gripper target-position echo."""
    history = getattr(session, '_motion_action_history', [])
    if history:
        if (history[-1]['episode_id'] != session.episode_id
                or history[-1]['step_id'] != session.step_id):
            raise ValueError('Stale gripper ACK command history')
        action = history[-1]['joints']
    else:
        if session.step_id != 0:
            raise ValueError('Missing acknowledged gripper command history')
        action = session.obs['states']
    return {arm: float(action[i * 7 + 6]) for i, arm in enumerate(ARMS)}


def telemetry(env, kin, step_id):
    """Return independent PhysX finger positions or an explicit unavailable receipt."""
    try:
        positions = np.asarray([env.robot_manager.get_end_effector_real_val(
            kin.robots[arm], env_idx_list=[0])[0] for arm in ARMS], float)
        if positions.shape != (2, 2) or not np.isfinite(positions).all():
            raise ValueError('Expected two finite measured finger positions per arm')
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
        return dict(finger_positions_available=False,
                    finger_positions_error=f'{type(error).__name__}: {error}',
                    finger_positions_step_id=step_id)
    return dict(finger_positions_available=True, finger_positions_m=positions,
                finger_positions_step_id=step_id)


def sample(session):
    """Require a fresh measured sample belonging to this exact native ACK."""
    obs = session.obs
    positions = np.asarray(obs.get('finger_positions_m', []), float)
    if (obs.get('finger_positions_available') is not True
            or obs.get('finger_positions_step_id') != session.step_id
            or positions.shape != (2, 2) or not np.isfinite(positions).all()):
        raise ValueError('Missing or stale PhysX measured finger positions')
    return positions


def observed(session, arm):
    i = ARMS.index(arm)
    return transform(session.obs['eef_positions'][i],
                     session.obs['eef_quaternions_wxyz'][i])


def hold(session, action, closing, goals, config, remaining, limits,
         teacher_request_id, receipt, save):
    """Keep both arm poses and close setpoints until fingers stop, or fail loudly.

    Stable transitions use only successive post-close ACK samples. A motionless
    empty or blocked gripper can satisfy this test; no grasp claim is made.
    """
    from . import joint_motion as motion
    rows = []
    initial_step_id = session.step_id
    receipt.update(status='running', settings=asdict(config), closing_arms=list(closing),
                   source='PhysX robot_manager.get_end_effector_real_val',
                   control_dt=session.metadata['control_dt'], samples=[],
                   stability_semantics='finger_motion_stopped_not_grasp_verified')
    previous = None
    stable = {arm: 0 for arm in closing}
    held_action = np.asarray(action, np.float32).copy()
    try:
        sample(session)  # Refuse before the first closing action when unavailable.
        for arm in ARMS:
            dp, dr = motion.pose_error(observed(session, arm), goals[arm])
            if dp > limits.arrival_m or dr > limits.arrival_rad:
                receipt.update(status='arrival_not_reached', failed_arm=arm,
                               position_error_m=dp, rotation_error_rad=dr)
                return rows
        for tick in range(config.max_steps):
            if len(rows) >= remaining or session.step_id >= session.metadata['max_episode_steps']:
                receipt['status'] = 'step_budget'
                break
            if session.terminated or session.truncated:
                receipt['status'] = 'native_terminal'
                break
            diagnostics = {}
            for arm, offset in (('left', 0), ('right', 7)):
                q = held_action[offset:offset + 6]
                actual = motion.robot_state(session.kinematics, arm)[0]
                if np.max(np.abs(q - actual)) > limits.tracking_joint_rad:
                    receipt.update(status='joint_tracking_error', failed_arm=arm)
                    return rows
                check_with_diagnostics(session.kinematics, arm, np.vstack([actual, q]),
                    receipt, 'hold_actual_start_command_chord', None,
                    next_step_id=session.step_id + 1, execution_index=tick)
                start = session.kinematics.root(arm) @ session.kinematics.fk[arm].matrix(actual)
                end = session.kinematics.root(arm) @ session.kinematics.fk[arm].matrix(q)
                for alpha in np.arange(1, 10) / 9:
                    native = session.kinematics.root(arm) @ session.kinematics.fk[arm].matrix(
                        (1 - alpha) * actual + alpha * q)
                    dp, dr = motion.pose_error(native, motion.reference(start, end, alpha))
                    if dp > limits.path_position_m or dr > limits.path_angle_rad:
                        receipt['status'] = 'native_chord_geometry_error'
                        return rows
                diagnostics[arm] = motion_boundary.command_differences(session, arm, q)
            before = session.step_id
            result = session.chunk_step([held_action.copy()], teacher_request_id=teacher_request_id)
            acknowledged = result['steps']
            rows.extend(acknowledged)  # Preserve even malformed/partial native ACK evidence.
            for row in acknowledged:
                row['edited_target'] = {arm: dict(pose(goals[arm]),
                    gripper_closed=bool(row['executed_action'][offset + 6] < .5),
                    gripper_opening=float(row['executed_action'][offset + 6]))
                    for arm, offset in (('left', 0), ('right', 7))}
                row['ik_diagnostics'] = dict(method=motion.VERSION, stage='gripper_hold',
                    command_differences=diagnostics, physical_tracking_verified=False)
            if (len(acknowledged) != 1 or session.step_id != before + 1
                    or acknowledged[0].get('step_id') != session.step_id
                    or not acknowledged[0].get('valid', False)
                    or not np.array_equal(requested_action(acknowledged[0]), held_action)):
                receipt['status'] = 'ack_missing_or_mismatched'
                break
            positions = sample(session)
            errors = {}
            arrived = True
            for arm in ARMS:
                dp, dr = motion.pose_error(observed(session, arm), goals[arm])
                errors[arm] = dict(position_m=dp, angle_rad=dr)
                arrived &= dp <= limits.arrival_m and dr <= limits.arrival_rad
            if previous is not None:
                for arm in closing:
                    delta = float(np.max(np.abs(positions[ARMS.index(arm)] - previous[ARMS.index(arm)])))
                    stable[arm] = stable[arm] + 1 if delta < config.delta_m else 0
            receipt['samples'].append(dict(step_id=session.step_id,
                finger_positions_m=positions.tolist(), stable_transitions=dict(stable),
                pose_errors=errors))
            previous = positions.copy()
            if not arrived:
                receipt['status'] = 'hold_pose_error'
                break
            if session.terminated or session.truncated:
                receipt['status'] = 'native_terminal'
                break
            if len(rows) >= config.min_steps and min(stable.values()) >= config.stable_transitions:
                receipt['status'] = 'fingers_stopped_not_grasp_verified'
                break
            save(rows)
        else:
            receipt['status'] = 'finger_stability_timeout'
    except SelfCollisionError as error:
        receipt.update(status='self_collision', error=str(error))
    except ValueError as error:
        # Only telemetry errors are classified here; planner/collision errors
        # propagate to the caller's normal safety receipt handling.
        if 'PhysX measured finger' not in str(error):
            raise
        receipt.update(status='finger_telemetry_unavailable', error=str(error))
    finally:
        receipt['executed_steps'] = session.step_id - initial_step_id
        receipt['returned_ack_rows'] = len(rows)
        receipt['ack_rows_missing'] = receipt['executed_steps'] - len(rows)
        if receipt['status'] == 'running':
            receipt['status'] = 'interrupted'
        if receipt['status'] != 'fingers_stopped_not_grasp_verified':
            receipt.setdefault('error', (f"{receipt['status']}: executed {receipt['executed_steps']} native ACK steps; "
                f"require min_steps={config.min_steps}, stable_transitions={config.stable_transitions}, "
                f"both-finger delta<{config.delta_m:g}m, timeout={config.max_steps} steps; "
                f"EEF arrival <= {limits.arrival_m:g}m / {limits.arrival_rad:g}rad"))
        save(rows)
    return rows


def execute_events(session, mode, targets, minimum_steps, decision, teacher_request_id,
                   limits, segment_executor, *, gripper_intent=None):
    """Run only GPT macros through pose events, keeping original native budgets."""
    from . import joint_motion as motion
    is_short = mode == 'short_motion'
    directory = session.episode_dir / mode
    directory.mkdir(exist_ok=True)
    report_path = directory / f'{decision:03d}.json'
    if report_path.exists():
        raise ValueError('Motion decision already exists; refusing replay')
    segment_directory = directory / f'{decision:03d}_segments'
    rows = []
    initial_step_id = session.step_id
    session.kinematics.check()
    # Segmentation must not turn the existing whole-request path cap into a per-event cap.
    for arm in ARMS:
        previous = observed(session, arm)
        distance = angle = 0.0
        for target in targets:
            if arm not in target:
                continue
            goal = transform(target[arm]['position'], target[arm]['quaternion_wxyz'])
            dp, dr = motion.pose_error(previous, goal)
            distance += dp
            angle += dr
            previous = goal
        if distance > .205 or angle > 1.01:
            raise motion.PlanningError('Whole-request Cartesian path budget exceeded')
    opening = issued_openings(session)
    events, start = [], 0
    close_seen = {arm: False for arm in ARMS}
    rewritten = deepcopy(targets)
    for index, target in enumerate(targets):
        closing = []
        changed = False
        for arm in ARMS:
            if is_short:
                intent = (gripper_intent[arm] if gripper_intent is not None else
                          ('closed' if target[arm]['gripper_closed'] else 'open'))
                original = target[arm].get('gripper_opening', float(not target[arm]['gripper_closed']))
                value = original if intent == 'keep' else float(intent == 'open')
                close = intent == 'closed' and (value < opening[arm] or not close_seen[arm])
                # Keep preserves the original continuous pi channel. Discrete
                # commands use pose events, never a thresholded pi opening.
                changed |= intent != 'keep' and value != opening[arm]
                if intent != 'keep':
                    rewritten[index][arm]['gripper_opening'] = opening[arm] if close else value
                    rewritten[index][arm]['gripper_closed'] = rewritten[index][arm]['gripper_opening'] < .5
                if close:
                    closing.append(arm)
            elif arm in target:
                command = target[arm]['gripper']
                value = opening[arm] if command == 'keep' else float(command == 'open')
                close = command == 'close' and (value < opening[arm] or not close_seen[arm])
                if close:
                    closing.append(arm)
                if command == 'close':
                    rewritten[index][arm]['gripper'] = 'keep'
            else:
                continue
            if close:
                close_seen[arm] = True
            elif value > opening[arm]:
                close_seen[arm] = False
            opening[arm] = value
        if closing or (is_short and changed):
            events.append(dict(start=start, end=index + 1, closing=closing,
                               openings=dict(opening)))
            start = index + 1
    if start < len(targets):
        events.append(dict(start=start, end=len(targets), closing=[], openings=dict(opening)))
    report = dict(version=motion.VERSION, status='running', executed_steps=0,
                  requested_minimum_steps=minimum_steps, targets=targets,
                  events=[], segments=[], final_arrival_verified=False, parameters=asdict(limits),
                  control_dt=session.metadata['control_dt'])
    if not is_short:
        report.update(arm=next(iter(targets[0])),
                      waypoints=[next(iter(item.values())) for item in targets],
                      completed_waypoints=0,
                      via_observations=[dict(waypoint=i, observed=False) for i in range(len(targets))])
    maximum = min(limits.max_steps, session.metadata['max_episode_steps'] - session.step_id)
    report['maximum_steps'] = maximum

    def save(extra=()):
        report['executed_steps'] = session.step_id - initial_step_id
        report['returned_ack_rows'] = len(rows) + len(extra)
        report['ack_rows_missing'] = report['executed_steps'] - report['returned_ack_rows']
        write_json(report_path, report)

    save()
    try:
        config = configured(getattr(session, 'gripper_hold_config', None))
        report['gripper_hold_config'] = asdict(config) if config else None
        for number, event in enumerate(events):
            remaining = maximum - len(rows)
            if remaining < 1:
                report['status'] = 'step_budget'
                break
            subset = rewritten[event['start']:event['end']]
            reserve = config.min_steps if event['closing'] else 0
            if remaining < reserve + (1 if is_short else 0):
                report.update(status='step_budget', required_hold_minimum_steps=reserve)
                break
            kwargs = dict(_directory=segment_directory, limits=replace(limits, max_steps=remaining - reserve))
            if is_short:
                # The total requested minimum belongs to this macro, not every event.
                requested = max(1, minimum_steps - len(rows) - reserve) if number == len(events) - 1 else 1
                result = segment_executor(session, subset, requested, number,
                                          teacher_request_id, **kwargs)
                receipt = result['correction_result']
            else:
                arm = next(iter(subset[0]))
                result = segment_executor(session, arm, [item[arm] for item in subset], number,
                                          teacher_request_id, **kwargs)
                receipt = result['motion_result']
            rows.extend(result['steps'])
            entry = dict(receipt['segments'][0]) if receipt.get('segments') else {}
            entry.update(target_start=event['start'], target_end=event['end'], receipt=receipt)
            report['segments'].append(entry)
            if 'distribution_reference' in receipt:
                report.setdefault('distribution_reference', receipt['distribution_reference'])
            if not is_short:
                report['completed_waypoints'] += receipt.get('completed_waypoints', 0)
                for evidence in receipt.get('via_observations', []):
                    index = event['start'] + evidence['waypoint']
                    report['via_observations'][index] = dict(evidence, waypoint=index)
            if receipt['status'] not in SUCCESS:
                report['status'] = receipt['status']
                if 'error' in receipt:
                    report['error'] = receipt['error']
                if 'collision_failure' in receipt:
                    report['collision_failure'] = receipt['collision_failure']
                break
            if event['closing']:
                action = np.asarray(rows[-1]['executed_action'] if rows else session.obs['states'], np.float32).copy()
                goals = {arm: (transform(targets[event['end'] - 1][arm]['position'],
                              targets[event['end'] - 1][arm]['quaternion_wxyz'])
                              if is_short or arm in targets[event['end'] - 1]
                              else session.kinematics.root(arm) @ session.kinematics.fk[arm].matrix(
                                  action[ARMS.index(arm) * 7:ARMS.index(arm) * 7 + 6])) for arm in ARMS}
                for arm in event['closing']:
                    action[ARMS.index(arm) * 7 + 6] = event['openings'][arm]
                hold_receipt = dict(target_index=event['end'] - 1)
                report['events'].append(hold_receipt)
                held_rows = hold(session, action, event['closing'], goals, config,
                                 maximum - len(rows), limits, teacher_request_id,
                                 hold_receipt, save)
                rows.extend(held_rows)
                if hold_receipt['status'] != 'fingers_stopped_not_grasp_verified':
                    report['status'] = hold_receipt['status']
                    report['error'] = hold_receipt['error']
                    break
            save()
        else:
            final = targets[-1]
            report['final_errors'] = {arm: dict(zip(('position_m', 'angle_rad'), motion.pose_error(
                observed(session, arm), transform(final[arm]['position'], final[arm]['quaternion_wxyz']))))
                for arm in ARMS if arm in final}
            report['final_arrival_verified'] = all(
                error['position_m'] <= limits.arrival_m and error['angle_rad'] <= limits.arrival_rad
                for error in report['final_errors'].values())
            report['status'] = (('targets_reached_not_grasp_verified' if is_short
                                 else 'waypoints_reached_not_grasp_verified')
                                if report['final_arrival_verified'] else 'arrival_not_reached')
    except (motion.PlanningError, motion.SelfCollisionError) as error:
        report.update(status='self_collision' if isinstance(error, motion.SelfCollisionError)
                      else 'execution_refused', error=str(error))
    except Exception as error:
        report.update(status='execution_error_after_ack' if session.step_id > initial_step_id
                      else 'execution_error', error_type=type(error).__name__,
                      error=f'{type(error).__name__}: {error}; native ACK steps={session.step_id - initial_step_id}')
        raise
    finally:
        if report['status'] == 'running':
            report['status'] = 'interrupted'
        if report['status'] not in SUCCESS:
            report.setdefault('error', (f"{report['status']}: executed {session.step_id - initial_step_id} "
                f"native ACK steps within macro budget {maximum}; EEF arrival requires "
                f"<={limits.arrival_m:g}m / {limits.arrival_rad:g}rad; no automatic lift"))
        save()
    return dict(episode_id=session.episode_id, step_id=session.step_id, steps=rows,
                **{('correction_result' if is_short else 'motion_result'): report})
