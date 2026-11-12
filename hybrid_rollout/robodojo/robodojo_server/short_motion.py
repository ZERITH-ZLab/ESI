"""Complete dual-arm short EEF targets through bounded-budget quintic planning."""

from dataclasses import asdict, replace
import json

import numpy as np

from ..io import write_json
from .action_edit_kinematics import pose, transform
from . import joint_motion as motion, motion_boundary
from .self_collision import SelfCollisionError, check_with_diagnostics


ARMS = ('left', 'right')


def gripper_schedule(targets, arm, count):
    """Resample the original pi target-index channel alongside arm actions.

    Equal counts preserve every original opening. Extra motion ticks linearly
    interpolate the ordered opening sequence, including its first and final
    values. Pose duplicates never merge gripper events; Cartesian via arrival
    does not determine the gripper timing. A single EEF target stays constant.
    """
    original = np.asarray([target[arm].get('gripper_opening',
        float(not target[arm]['gripper_closed'])) for target in targets], float)
    if (type(count) is not int or count < len(original) or len(original) == 0
            or not np.isfinite(original).all() or np.any(original < 0)
            or np.any(original > 1)):
        raise motion.PlanningError('Invalid short-motion gripper index schedule')
    openings = np.interp(np.linspace(0, len(original) - 1, count),
                         np.arange(len(original)), original)
    return openings, dict(method='pi_target_index_linear_resampling_parallel',
        original_openings=original.tolist(), arrival_at_via_controls_gripper=False,
        acknowledged_openings=[])


def _execute_segment(session, targets, minimum_steps, decision, teacher_request_id,
                     limits=motion.DEFAULTS, *, _directory=None):
    """Plan complete Cartesian targets, execute parallel grippers, verify final goal.

    ``minimum_steps`` is the requested minimum; additional GPT action ticks can
    be used within the existing 60-tick and remaining-episode budgets. Student
    action chunks never enter this function. Native physics is not advanced by
    IK, retiming, collision checking, or refused planning.
    """
    if (type(minimum_steps) is not int or not 1 <= minimum_steps <= 15
            or type(decision) is not int or decision < 0
            or not isinstance(targets, list) or not targets
            or any(set(target) != set(ARMS) for target in targets)):
        raise ValueError('Invalid short-motion request')
    if session.finished or session.terminated or session.truncated:
        raise ValueError('Episode already ended')
    kin = session.kinematics
    kin.check()
    maximum = min(limits.max_steps, session.metadata['max_episode_steps'] - session.step_id)
    directory = _directory if _directory is not None else session.episode_dir / 'short_motion'
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / f'{decision:03d}.json'
    candidate_path = directory / f'{decision:03d}_candidate.npz'
    if report_path.exists():
        raise ValueError('Short-motion decision already exists; refusing replay')
    report = dict(version=motion.VERSION, status='planning', requested_minimum_steps=minimum_steps,
                  maximum_steps=maximum, parameters=asdict(limits), targets=targets,
                  executed_steps=0, tracking=[], pre_send_tracking=[], command_boundary_differences=[],
                  gripper_schedule={},
                  final_arrival_verified=False, strict_continuity_guaranteed=False)
    rows = []
    plans, initial, points, inherited, sources = {}, {}, {}, {}, {}
    dt = session.metadata['control_dt']

    def save():
        report['executed_steps'] = len(rows)
        write_json(report_path, report)

    def observed(arm):
        index = ARMS.index(arm)
        return transform(session.obs['eef_positions'][index],
                         session.obs['eef_quaternions_wxyz'][index])

    save()
    try:
        for arm in ARMS:
            initial[arm] = motion.robot_state(kin, arm)
            points[arm] = []
            previous = kin.root(arm) @ kin.fk[arm].matrix(initial[arm][0])
            distance = angle = 0.0
            for target in targets:
                item = target[arm]
                if type(item['gripper_closed']) is not bool:
                    raise ValueError('Explicit boolean gripper required')
                opening = item.get('gripper_opening', float(not item['gripper_closed']))
                if not np.isfinite(opening) or not 0 <= opening <= 1:
                    raise ValueError('Invalid gripper opening')
                point = dict(position=item['position'], quaternion_wxyz=item['quaternion_wxyz'],
                             gripper='close' if item['gripper_closed'] else 'open')
                goal = transform(point['position'], point['quaternion_wxyz'])
                dp, dr = motion.pose_error(previous, goal)
                distance += dp
                angle += dr
                previous = goal
                points[arm].append(point)
            for point in points[arm]:
                motion.validate_request(arm, [point])
            if distance > .205 or angle > 1.01:
                raise motion.PlanningError('Short-motion total Cartesian path budget exceeded')
            v, a, sources[arm] = motion_boundary.initial_derivatives(session, arm)
            inherited[arm] = (v, a)
        minimum_steps = max(minimum_steps, len(targets))
        report["effective_minimum_steps"] = minimum_steps
        if maximum < minimum_steps:
            raise motion.PlanningError('Insufficient remaining short-motion step budget')

        def plan(arm, low, high):
            q, bounds, velocity = initial[arm]
            return motion.plan_path(
                kin.fk[arm], kin.root(arm), q, bounds, velocity, points[arm], dt,
                replace(limits, max_steps=high), start_velocity=inherited[arm][0],
                start_acceleration=inherited[arm][1], minimum_steps=low)

        # Both arms are regenerated at the same physical duration. Never resample
        # one arm's joint targets linearly to match the other arm's sample count.
        for arm in ARMS:
            plans[arm] = plan(arm, minimum_steps, maximum)
        common = max(minimum_steps, *(p['execution_steps'] for p in plans.values()))
        for common in range(common, maximum + 1):
            try:
                aligned = {arm: plan(arm, common, common) for arm in ARMS}
            except motion.PlanningError:
                continue
            plans = aligned
            break
        else:
            raise motion.PlanningError('No common dual-arm duration within short-motion budget')
        np.savez_compressed(candidate_path, control_dt=dt, execution_steps=common,
            boundary_reference_json=np.asarray(json.dumps({
                arm: dict(source=sources[arm], **plans[arm]['boundary_recovery']) for arm in ARMS})),
            candidate_collision_checked=np.asarray(False),
            **{f'{arm}_{key}': (initial[arm][0] if key == 'start_joints' else
                np.tile(initial[arm][0], (common, 1)) if key == 'joints' and not plans[arm]['execution_steps']
                else plans[arm][key])
               for arm in ARMS for key in ('joints', 'dense_joints', 'start_joints',
                   'start_velocity', 'start_acceleration', 'end_velocity', 'end_acceleration')})
        report['candidate_plan_npz'] = str(candidate_path)
        actions = np.tile(np.asarray(session.obs['states'], np.float32), (common, 1))
        for arm, offset in (('left', 0), ('right', 7)):
            segment = plans[arm]
            if not segment['execution_steps']:
                segment['joints'] = np.tile(initial[arm][0], (common, 1))
            if len(segment['joints']) != common:
                raise motion.PlanningError('Dual-arm duration alignment failed')
            openings, gripper_receipt = gripper_schedule(targets, arm, common)
            actions[:, offset:offset + 6] = segment['joints']
            actions[:, offset + 6] = openings
            lower, upper = initial[arm][1].T
            if (np.any(actions[:, offset:offset + 6] < lower)
                    or np.any(actions[:, offset:offset + 6] > upper)):
                raise motion.PlanningError('Float32 native targets exceed joint position bounds')
            dp, dr = motion.pose_error(
                kin.root(arm) @ kin.fk[arm].matrix(actions[-1, offset:offset + 6]),
                transform(targets[-1][arm]['position'], targets[-1][arm]['quaternion_wxyz']))
            if dp > limits.ik_position_m or dr > limits.ik_angle_rad:
                raise motion.PlanningError('Float32 native endpoint does not reach the complete original goal')
            gripper_receipt["planned_openings_float32"] = actions[:, offset + 6].tolist()
            report["gripper_schedule"][arm] = gripper_receipt
            check_with_diagnostics(kin, arm, segment['dense_joints'], report,
                'planned_dense_joint_chords', candidate_path)
            check_with_diagnostics(kin, arm, np.vstack([initial[arm][0], segment['joints']]),
                report, 'planned_control_joint_chords', candidate_path)
        report.update(status='running', planned_steps=common, boundary_reference={
            arm: dict(source=sources[arm], **plans[arm]['boundary_recovery']) for arm in ARMS},
            final_reference_error={arm: dict(zip(('position_m', 'angle_rad'), motion.pose_error(
                kin.root(arm) @ kin.fk[arm].matrix(actions[-1, offset:offset + 6]),
                transform(targets[-1][arm]['position'], targets[-1][arm]['quaternion_wxyz']))))
                for arm, offset in (('left', 0), ('right', 7))})
        np.savez_compressed(directory / f'{decision:03d}_plan.npz', actions=actions,
                            control_dt=dt, **{f'{arm}_dense_joints': plans[arm]['dense_joints'] for arm in ARMS})
        save()
        for index, action in enumerate(actions):
            diagnostics = {}
            for arm, offset in (('left', 0), ('right', 7)):
                actual = motion.robot_state(kin, arm)[0]
                q = action[offset:offset + 6]
                pre_send_error = float(np.max(np.abs(q - actual)))
                report["pre_send_tracking"].append(dict(step_id=session.step_id + 1, arm=arm,
                    joint_error_rad=pre_send_error, original_tracking_would_stop=bool(pre_send_error > .12)))
                if pre_send_error > limits.tracking_joint_rad:
                    raise motion.PlanningError('Short-motion joint tracking error before send')
                check_with_diagnostics(kin, arm, np.vstack([actual, q]), report,
                    'actual_start_command_chord', candidate_path,
                    next_step_id=session.step_id + 1, execution_index=index)
                start = kin.root(arm) @ kin.fk[arm].matrix(actual)
                end = kin.root(arm) @ kin.fk[arm].matrix(q)
                for alpha in np.arange(1, 10) / 9:
                    native = kin.root(arm) @ kin.fk[arm].matrix((1 - alpha) * actual + alpha * q)
                    dp, dr = motion.pose_error(native, motion.reference(start, end, alpha))
                    if dp > limits.path_position_m or dr > limits.path_angle_rad:
                        raise motion.PlanningError('Short-motion actual-start chord violates Cartesian corridor')
                diagnostics[arm] = motion_boundary.command_differences(session, arm, q)
            result = session.chunk_step([action.copy()], teacher_request_id=teacher_request_id)
            for row in result['steps']:
                row['edited_target'] = {arm: dict(pose(kin.root(arm) @ kin.fk[arm].matrix(
                    action[offset:offset + 6])), gripper_closed=bool(row['executed_action'][offset + 6] < .5),
                    gripper_opening=float(row['executed_action'][offset + 6])) for arm, offset in (('left', 0), ('right', 7))}
                row['ik_diagnostics'] = dict(method=motion.VERSION,
                    requested_targets=targets[-1], original_goal_replaced=False,
                    command_differences=diagnostics, physical_tracking_verified=False)
            rows.extend(result['steps'])
            for arm, offset in (('left', 0), ('right', 7)):
                report['gripper_schedule'][arm]['acknowledged_openings'].extend(
                    float(row['executed_action'][offset + 6]) for row in result['steps'])
            for item in diagnostics.values():
                item['acknowledged'] = bool(result['steps'])
                item['proposed_not_acknowledged'] = not bool(result['steps'])
            report['command_boundary_differences'].append(diagnostics)
            errors = {}
            for arm, offset in (('left', 0), ('right', 7)):
                q = action[offset:offset + 6]
                dp, dr = motion.pose_error(observed(arm), kin.root(arm) @ kin.fk[arm].matrix(q))
                dq = float(np.max(np.abs(motion.robot_state(kin, arm)[0] - q)))
                errors[arm] = dict(position_error_m=dp, rotation_error_rad=dr, joint_error_rad=dq,
                    original_tracking_would_stop=bool(dp > .025 or dr > .20 or dq > .12))
                if dp > limits.tracking_m or dr > limits.tracking_rad or dq > limits.tracking_joint_rad:
                    report['status'] = 'tracking_error'
                stationary = all(max(motion.pose_error(
                    transform(point['position'], point['quaternion_wxyz']),
                    kin.root(arm) @ kin.fk[arm].matrix(initial[arm][0]))) < 1e-10
                    for point in points[arm])
                if stationary:
                    hd, hr = motion.pose_error(observed(arm), kin.root(arm) @ kin.fk[arm].matrix(initial[arm][0]))
                    if hd > limits.held_arm_m or hr > limits.held_arm_rad:
                        report['status'] = 'held_arm_tracking_error'
            report['tracking'].append(dict(step_id=session.step_id, arms=errors))
            if session.terminated or session.truncated:
                report['status'] = 'native_terminal'
            save()
            if report['status'] != 'running':
                break
        report['final_errors'] = {arm: dict(zip(('position_m', 'angle_rad'), motion.pose_error(
            observed(arm), transform(targets[-1][arm]['position'], targets[-1][arm]['quaternion_wxyz']))))
            for arm in ARMS}
        report['final_arrival_verified'] = all(
            item['position_m'] <= limits.arrival_m and item['angle_rad'] <= limits.arrival_rad
            for item in report['final_errors'].values())
        if report['status'] == 'running':
            report['status'] = ('targets_reached_not_grasp_verified' if report['final_arrival_verified']
                                else 'arrival_not_reached')
    except (motion.PlanningError, SelfCollisionError) as exc:
        report.update(status='self_collision' if isinstance(exc, SelfCollisionError) else
                      ('planning_failed' if not rows else 'execution_refused'), error=str(exc))
    finally:
        if report['status'] in ('planning', 'running'):
            report['status'] = 'interrupted'
        save()
    return dict(episode_id=session.episode_id, step_id=session.step_id, steps=rows,
                correction_result=report)


def execute(session, targets, minimum_steps, decision, teacher_request_id,
            limits=motion.DEFAULTS, *, gripper_intent=None):
    """Preserve keep channels and synchronize explicit GPT closing pose events."""
    if (type(minimum_steps) is not int or not 1 <= minimum_steps <= 15
            or type(decision) is not int or decision < 0
            or not isinstance(targets, list) or not targets
            or any(set(target) != set(ARMS) for target in targets)):
        raise ValueError('Invalid short-motion request')
    if session.finished or session.terminated or session.truncated:
        raise ValueError('Episode already ended')
    if gripper_intent is not None and (set(gripper_intent) != set(ARMS)
            or any(value not in ('keep', 'open', 'closed') for value in gripper_intent.values())):
        raise ValueError('Invalid explicit short gripper intent')
    for target in targets:
        for item in target.values():
            if type(item['gripper_closed']) is not bool:
                raise ValueError('Explicit boolean gripper required')
            opening = item.get('gripper_opening', float(not item['gripper_closed']))
            if not np.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError('Invalid gripper opening')
    if ((gripper_intent is not None and 'closed' not in gripper_intent.values())
            or (gripper_intent is None and not any(
                target[arm]['gripper_closed'] for target in targets for arm in ARMS))):
        return _execute_segment(session, targets, minimum_steps, decision, teacher_request_id, limits)
    from .gripper_hold import execute_events
    return execute_events(session, 'short_motion', targets, minimum_steps, decision,
                          teacher_request_id, limits, _execute_segment,
                          gripper_intent=gripper_intent)
