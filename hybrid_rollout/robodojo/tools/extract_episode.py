"""Read-only rollout extraction. RGB is validated and indexed, never re-encoded.

Output is NOT yet a LeRobot dataset. Both possible action labels are retained.
Only manifest.json marks a completely validated export; failures leave report.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..force_hold_archive import MAPPING_KEYS, observation_diagnostics, validate_hold_mapping, validate_proposal_mapping, resolve_base_model_targets

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
VERSION = 'robodojo_training_extract.v3'


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def vector(x):
    x = np.asarray(x, dtype=np.float32)
    require(x.shape == (14,) and np.isfinite(x).all(), 'Invalid 14D vector')
    return x


def decision_index(history, n):
    owners = np.full(n, -1, dtype=np.int64)
    rows, cursor, seen = [], 0, set()
    for index, row in enumerate(history):
        student_only = 'end_step' in row and 'response' not in row
        a = row['end_step']-row['executed_steps'] if student_only else row['start_tick']
        b = row['end_step'] if student_only else row['end_tick']
        d = index if student_only else row['decision']
        require(d not in seen, 'Duplicate decision')
        seen.add(d)
        require(a == cursor and a <= b <= n and b-a == row['executed_steps'],
                f'Decision {d}: discontinuous/invalid execution range')
        owners[a:b] = d
        cursor = b
        response = dict(mode='student') if student_only else row['response']
        mode = response['mode']
        kind = 'student' if mode == 'student' else 'long' if 'motion' in response else 'short'
        rows.append(dict(decision_id=d, start_step=a, end_step_exclusive=b,
                         label_available_at_step=a, mode=mode, motion_kind=kind,
                         response=response, source=row.get('source'), prediction_id=row.get('prediction_id'),
                         correction_result=row.get('correction_result')))
    require(cursor == n and np.all(owners >= 0), 'History does not cover executed steps')
    return owners, rows


def extract(archive, output, *, allow_wall_time_limit=False):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    require(not output.is_relative_to(archive), 'Export must be outside original archive')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rollout_extraction_report.v1', status='validating', archive=str(archive))
    try:
        summary = read(archive/'sim/summary.json')
        if summary.get('reason') not in ('terminal', 'teacher_abandon'):
            sample = read(archive.parent/'sample.json') if allow_wall_time_limit else {}
            attempt = next((a for a in sample.get('attempts', [])
                            if Path(a.get('archive') or '').name == archive.name), {})
            cleanup = read(archive/'component_reset.json') if allow_wall_time_limit else {}
            require(allow_wall_time_limit and summary.get('reason') == 'controller_error'
                    and attempt.get('state') == 'wall_time_limit' and attempt.get('finished_at') is not None
                    and cleanup.get('all_owned_processes_exited') is True and cleanup.get('ports_released') is True,
                    'Requires terminal/teacher_abandon or explicitly allowed finalized wall-time interruption')
            report['finalized_interruption'] = 'wall_time_limit'
        n, dt = summary['step_id'], summary['control_dt']
        require(n > 0 and np.isfinite(dt) and dt > 0, 'Invalid episode length/control_dt')
        sim = archive/'sim'/summary['episode_id']
        history = read(archive/'controller/history.json')
        student_only = bool(history and all('end_step' in row and 'response' not in row for row in history))
        owners, decisions = decision_index(history, n)
        require(len(list(sim.glob('action_*.json'))) == n, 'Action count differs from summary')
        require(len(list((sim/'observations').glob('*.npz'))) == n+1, 'Observation count differs from T+1')
        states, commands, poses, quats, image_refs, flags = [], [], [], [], [], []
        diagnostics, receipts = [], []
        instruction, shapes = None, {}
        for t in range(n+1):
            path = sim/'observations'/f'{t:06d}.npz'
            with np.load(path, allow_pickle=False) as obs:
                states.append(vector(obs['states']))
                diagnostics.append(observation_diagnostics(obs, t))
                text = str(obs['instruction'].item())
                if instruction is None:
                    instruction = text
                require(text == instruction, 'Task instruction changed within episode')
                for camera in CAMERAS:
                    im = obs[camera]
                    require(im.dtype == np.uint8 and im.ndim == 3 and im.shape[-1] == 3,
                            f'Invalid RGB at {t}: {camera}')
                    shapes.setdefault(camera, list(im.shape))
                    require(list(im.shape) == shapes[camera], 'Image dimensions changed')
                p, q = obs['eef_positions'], obs['eef_quaternions_wxyz']
                require(p.shape == (2, 3) and q.shape == (2, 4)
                        and np.isfinite(p).all() and np.isfinite(q).all(), 'Invalid EEF state')
                require(np.allclose(np.linalg.norm(q, axis=1), 1, atol=.01), 'Non-unit EEF quaternion')
                poses.append(p.copy()); quats.append(q.copy())
            image_refs.append(dict(step=t, path=str(path.relative_to(archive)),
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if t < n:
                action = read(sim/f'action_{t:06d}.json')
                receipts.append(action)
                require(action['valid'] and action['step_id'] == t+1, f'Invalid action receipt {t}')
                cmd = vector(action['executed_action'])
                require(np.all((cmd[[6, 13]] >= 0) & (cmd[[6, 13]] <= 1)), 'Invalid gripper command')
                commands.append(cmd)
                flags.append([bool(action[k]) for k in ('terminated', 'truncated', 'success')])
                # Receipt state is POST-action, not the input for this action.
                with np.load(sim/'observations'/f'{t+1:06d}.npz', allow_pickle=False) as nxt:
                    np.testing.assert_allclose(vector(action['obs']['states']), nxt['states'], atol=1e-6, rtol=0)
            if t % 100 == 0:
                print(json.dumps(dict(event='extract_progress', step=t, total=n)), flush=True)
        event_rows, events = validate_hold_mapping(sim, receipts)
        states, commands = np.asarray(states), np.asarray(commands)
        diagnostic_arrays = {'observation_'+key: np.asarray([row[key] for row in diagnostics])
                             for key in diagnostics[0]}
        mapping_arrays = {key: np.asarray([row.get(key) if row.get(key) is not None else -1 for row in receipts], np.int64)
                          for key in MAPPING_KEYS[:2]}
        mapping_arrays['hold_event_id'] = np.asarray([row.get('hold_event_id') or '' for row in receipts], dtype=str)
        require(not np.any(np.asarray(flags[:-1])[:, :2]) if n > 1 else True,
                'Execution continues after a terminal receipt')
        for i, key in enumerate(('terminated', 'truncated', 'success')):
            if key in summary:
                require(flags[-1][i] == bool(summary[key]), f'Final {key} disagrees with summary')
        for row in decisions:
            a, b, d = row['start_step'], row['end_step_exclusive'], row['decision_id']
            has_mapping = any(any(key in receipt for key in MAPPING_KEYS) for receipt in receipts[a:b])
            has_events = any(event.get('student_prediction_id') == row['prediction_id'] for event in events.values())
            if has_mapping or has_events:
                with np.load(archive/'controller'/f'proposal_{d:03d}.npz', allow_pickle=False) as proposal:
                    np.testing.assert_array_equal(proposal['states'], states[a])
                    validate_proposal_mapping(receipts[a:b], proposal['actions'], events, row['prediction_id'])
            # Zero-step planning failures and holds have no training actions.
            if a == b:
                continue
            execution_path = archive/'controller'/f'execution_{d:03d}.npz'
            if student_only and not execution_path.exists():
                continue  # Historical pure-student archives only persisted per-ACK commands.
            with np.load(execution_path, allow_pickle=False) as execution:
                np.testing.assert_allclose(execution['actions'], commands[a:b], atol=1e-6, rtol=0)
                np.testing.assert_allclose(execution['states'], states[a+1:b+1], atol=1e-6, rtol=0)
        proposal_paths = {row['prediction_id']: archive/'controller'/f"proposal_{row['decision_id']:03d}.npz"
                          for row in decisions if row['prediction_id'] is not None}
        resolved_bases = resolve_base_model_targets(events, proposal_paths, receipts)
        write(output/'event_base_targets.json', resolved_bases)
        np.savez_compressed(output/'steps.npz', state=states[:-1], command=commands,
                            next_state=states[1:], observation_state=states,
                            observation_eef_positions=np.asarray(poses),
                            observation_eef_quaternions_wxyz=np.asarray(quats),
                            step=np.arange(n), simulation_time=np.arange(n)*dt,
                            decision_id=owners, outcome_flags=np.asarray(flags, dtype=bool),
                            **diagnostic_arrays, **mapping_arrays)
        write(output/'force_hold_events.json', event_rows)
        write(output/'action_mapping.json', [dict(step=t, **{key: row.get(key) for key in MAPPING_KEYS})
                                           for t, row in enumerate(receipts)])
        write(output/'decisions.json', decisions)
        write(output/'observations.json', image_refs)
        metadata = dict(summary=summary, task_instruction=instruction, source_archive=str(archive),
                        provenance={})
        for name in ('evaluation_case', 'collection_job', 'policy_identity', 'resume_identity'):
            p = archive/f'{name}.json'
            metadata['provenance'][name] = read(p) if p.exists() else None
        metadata['evaluation_outcome'] = read(archive/'sim/evaluation_outcome.json')
        for name in ('result', 'scene_query_usage', 'controller_protocol'):
            p = archive/'controller'/f'{name}.json'
            metadata['provenance'][name] = read(p) if p.exists() else None
        write(output/'episode.json', metadata)
        delta = np.abs(commands-states[1:])
        stats = {}
        for kind in ('student', 'short', 'long'):
            ids = [r['decision_id'] for r in decisions if r['motion_kind'] == kind]
            subset = delta[np.isin(owners, ids)]
            stats[kind] = dict(steps=len(subset),
                joint_abs_error_rad_mean=float(subset[:, JOINTS].mean()) if len(subset) else None,
                joint_abs_error_rad_max=float(subset[:, JOINTS].max()) if len(subset) else None,
                gripper_abs_error_mean=float(subset[:, [6, 13]].mean()) if len(subset) else None)
        report.update(status='passed', steps=n, observations=n+1, decisions=len(decisions),
                      zero_step_decisions=sum(r['start_step'] == r['end_step_exclusive'] for r in decisions),
                      command_vs_next_state=stats, image_shapes=shapes,
                      warnings=['Quality not reviewed; success does not make every step a good training target.',
                                'Simulation timestamps derived from control_dt, not sensor wall-clock timestamps.',
                                'Images are references into immutable source archive; keep source available.',
                                'Training action label is deliberately not selected.'])
        write(output/'report.json', report)
        files = ['steps.npz', 'decisions.json', 'observations.json', 'episode.json', 'report.json',
                 'force_hold_events.json', 'action_mapping.json', 'event_base_targets.json']
        write(output/'manifest.json', dict(schema=VERSION, episode_id=summary['episode_id'],
            steps=n, control_dt=dt, source_archive=str(archive), image_storage='source_npz_references',
            cameras=CAMERAS, training_action_label=None, available_action_labels=['command', 'next_state'],
            state_order='left joints 1..6, left gripper, right joints 1..6, right gripper',
            state_gripper_semantics='normalized internal target after feedback-based limiting; not measured finger position',
            gripper_semantics='0 closed, 1 open', outcome_flags=['terminated', 'truncated', 'success'],
            time_basis='nominal simulation time from archived control_dt; not wall-clock',
            eef_reference='environment_origin / link6; quaternion wxyz',
            force_hold_schema='pi_force_hold_event_v1', force_hold_event_count=len(events),
            diagnostic_input_policy='measured fingers and force are diagnostics, not automatic policy inputs',
            diagnostic_alignment='observation[t] force describes action[t-1] 40ms window; action[t] force is observation[t+1]; reset unknown',
            unknown_diagnostics='NaN numeric values with false validity masks; JSON mapping null is NPZ index -1 or empty event id',
            files=files, file_sha256={f: hashlib.sha256((output/f).read_bytes()).hexdigest() for f in files}))
        return report
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if not (output/'report.json').exists():
            write(output/'report.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
