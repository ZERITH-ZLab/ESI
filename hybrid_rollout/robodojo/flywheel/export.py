"""Export immutable native rollout evidence into the rollout.v1 contract.

This is an offline adapter. It never relabels unexecuted proposals as actions,
changes simulator state, calls a teacher, or modifies the original archive.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ..force_hold_archive import (MAPPING_KEYS, json_diagnostics,
                                  observation_diagnostics, validate_hold_mapping, validate_proposal_mapping, resolve_base_model_targets)

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
JOINTS = [*range(6), *range(7, 13)]
FILTER_VERSION = 'native_complete_receipt_tracking012_previous_failure_v1'


def read(path):
    """Read a required JSON evidence file."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Create output exclusively; existing exports are never overwritten."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path):
    """Return a content identity without loading large files into memory."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ranges(indices):
    """Encode unique integer indices as nonoverlapping half-open intervals."""
    result = []
    for index in sorted(set(indices)):
        if result and result[-1][1] == index:
            result[-1][1] += 1
        else:
            result.append([int(index), int(index) + 1])
    return result


def full_windows(mask, horizon):
    """Only admit chunks with every command valid; never cross excluded gaps."""
    if horizon < 1:
        raise ValueError('horizon must be positive')
    if len(mask) < horizon:
        return []
    sums = np.convolve(np.asarray(mask, int), np.ones(horizon, int), mode='valid')
    return np.flatnonzero(sums == horizon).tolist()


def completed_spans(path):
    """Pair timings within one writer; missing/failed sends cannot prove delivery."""
    pending, result = {}, []
    for line in Path(path).open():
        row = json.loads(line)
        key = (row.get('writer'), row.get('span_id'))
        if row.get('event') == 'start':
            pending[key] = row
        elif row.get('event') == 'end' and key in pending:
            start = pending.pop(key)
            result.append(dict(start, **row))
    return result


def privileged_queries(archive, spans, episode_id):
    """Require a matching successful tool reply AND completed send span."""
    workspace = archive / 'controller/codex_workspace'
    calls = {}
    for line in (workspace / 'rpc_out.jsonl').open():
        row = json.loads(line)
        if row.get('method') == 'item/tool/call':
            calls[row['id']] = row['params']
    requests = {}
    for path in workspace.glob('call_*_request.json'):
        row = read(path)
        requests[row['call_id']] = int(path.name.split('_')[1])
    sends = {row['call_id']: row for row in spans
             if row['stage'] == 'tool_result_send' and row['status'] == 'ok'}
    result = []
    for line_number, line in enumerate((workspace / 'rpc_in.jsonl').open(), 1):
        reply = json.loads(line)
        payload = reply.get('result', {})
        if reply.get('id') not in calls or not payload.get('success'):
            continue
        call = calls[reply['id']]
        index = requests.get(call['callId'])
        for item in payload.get('contentItems', []):
            if item.get('type') != 'inputText':
                continue
            packet = json.loads(item['text'])
            identity = 'scene_identities' in packet
            geometry = 'roles' in packet and 'objects' in packet
            if not (identity or geometry):
                continue
            result.append(dict(query_id=f'{episode_id}:query:{len(result)}',
                episode_id=episode_id, decision_id=packet.get('decision_id'),
                step_id=packet['step_id'], trigger='existing_gated_recovery',
                requested_fields=['identity_image'] if identity else ['object_pose'],
                response_ref=f'{workspace}/rpc_in.jsonl#L{line_number}',
                delivered_to_teacher=index in sends,
                coordinate_schema=None if identity else dict(frame=packet['frame'],
                    position_unit=packet['position_unit'], quaternion_order=packet['quaternion_order']),
                wall_s=None, send_wall_s=sends[index]['duration_s'] if index in sends else None,
                call_id=call['callId'], kind='identity_image' if identity else 'object_poses'))
    return result


def export_archive(archive, output, *, collection_commit, training_case_id, horizon=50,
                   required_motion='link6_relaxed_quintic_ack_soft_boundary_v1', validation_only=False,
                   training_case_ids=None, heldout_case_ids=None):
    """Build a new auditable bundle, with a student-input allowlist and filters."""
    archive, output = Path(archive).resolve(), Path(output).resolve()
    reset, summary = read(archive/'sim/reset.json'), read(archive/'sim/summary.json')
    outcome = read(archive/'sim/evaluation_outcome.json')
    artifact = read(archive/'artifact_manifest.json')
    run, history = read(archive/'controller/run.json'), read(archive/'controller/history.json')
    identity = read(archive/'resume_identity.json')
    case = reset['metadata']['evaluation_case']
    if case['case_id'] != training_case_id:
        raise ValueError('Collection case differs from the explicitly assigned training case')
    if reset['metadata'].get('motion_executor') != required_motion:
        raise ValueError('Collection used an unexpected IK executor')
    if not summary['complete'] or not outcome['complete'] or artifact['status'] != 'verified':
        raise ValueError('Require a complete native episode and verified archive')
    if not outcome['valid_for_success_rate']:
        raise ValueError('Invalid or uncertain native episode')
    if run['control_dt'] != .04 or run['action_dim'] != 14:
        raise ValueError('Unexpected native control contract')
    episode_id, count = reset['episode_id'], summary['step_id']
    raw = archive/'sim'/episode_id
    if len(list(raw.glob('action_*.json'))) != count:
        raise ValueError('Action inventory mismatch')
    if len(list((raw/'observations').glob('*.npz'))) != count + 1:
        raise ValueError('Observation inventory mismatch')
    states, observations = [], []
    for step in range(count + 1):
        path = raw/'observations'/f'{step:06d}.npz'
        with np.load(path, allow_pickle=False) as obs:
            state = obs['states']
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError(f'Invalid proprioception at {step}')
            for camera in CAMERAS:
                image = obs[camera]
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f'Invalid RGB at {step}/{camera}')
            states.append(state)
            diagnostics = json_diagnostics(observation_diagnostics(obs, step))
        observations.append(dict(observation_id=f'{episode_id}:{step}', episode_id=episode_id,
            step_id=step, sim_time_s=step*.04, images={c: f'{path}#{c}' for c in CAMERAS},
            state=state.tolist(), eef_pose=None, valid=True, arrays_ref=str(path), sha256=digest(path),
            measured_diagnostics=diagnostics))
    states = np.stack(states)
    action_receipts = [read(raw/f'action_{step:06d}.json') for step in range(count)]
    event_rows, hold_events = validate_hold_mapping(raw, action_receipts)
    force_hold_intervened = bool(hold_events) or any(r.get('gripper_override') for r in action_receipts)
    force_hold_steps = sum(event['actual_count'] for event in hold_events.values())
    resolved_bases = resolve_base_model_targets(hold_events,
        {row['prediction_id']: archive/'controller'/f"proposal_{row['decision']:03d}.npz" for row in history},
        [read(raw/f'action_{step:06d}.json') for step in range(count)])
    spans = completed_spans(archive/'timing/controller.jsonl')
    queries = privileged_queries(archive, spans, episode_id)
    workspace = archive/'controller/codex_workspace'
    rejected = Counter()
    for path in workspace.glob('call_*_result.json'):
        packet = read(path)
        if packet.get('error_type') == 'recoverable_tool_input':
            request_id = packet.get('request_id')
            if request_id:
                rejected[request_id] += 1
    compact_to_contract = {}
    for record in history:
        request = read(archive/'controller'/f"request_{record['decision']:03d}.json")
        compact = hashlib.sha256(request['request_id'].encode()).hexdigest()[:16]
        compact_to_contract[compact] = f"{episode_id}:decision:{record['decision']}"
    for query in queries:
        query['decision_id'] = compact_to_contract.get(query['decision_id'])
    delivered = [q for q in queries if q['delivered_to_teacher']]
    first_exposure = min((q['step_id'] for q in delivered), default=count + 1)
    decisions, segments, rows = [], [], [None] * count
    quality = np.ones(count, bool)
    excluded = []
    for record in history:
        index, start, end = record['decision'], record['start_tick'], record['end_tick']
        response = record['response']
        decision_id = f'{episode_id}:decision:{index}'
        if end-start != record['executed_steps'] or start < 0 or end > count:
            raise ValueError('Invalid decision interval')
        request = read(archive/'controller'/f'request_{index:03d}.json')
        if request['step_id'] != start:
            raise ValueError('Decision does not use the pre-action observation')
        proposal_path = archive/'controller'/f'proposal_{index:03d}.npz'
        with np.load(proposal_path, allow_pickle=False) as proposal:
            if not np.array_equal(proposal['states'], states[start]):
                raise ValueError('Student proposal input differs from pre-action state')
            decision_receipts = [read(raw/f'action_{step:06d}.json') for step in range(start,end)]
            if (any(any(key in receipt for key in MAPPING_KEYS) for receipt in decision_receipts)
                    or any(event.get('student_prediction_id') == record['prediction_id'] for event in hold_events.values())):
                validate_proposal_mapping(decision_receipts,proposal['actions'],hold_events,record['prediction_id'])
        with np.load(archive/'controller'/f'execution_{index:03d}.npz', allow_pickle=False) as execution:
            commands = execution['actions'].copy()
        if commands.shape != (end-start, 14):
            raise ValueError('Execution command shape mismatch')
        source = {'student':'student', 'edit':'gpt_edit', 'eef':'gpt_eef'}[response['mode']]
        for offset, step in enumerate(range(start, end)):
            receipt_path = raw/f'action_{step:06d}.json'
            receipt = read(receipt_path)
            action = np.asarray(receipt['executed_action'], np.float32)
            if rows[step] is not None or receipt['step_id'] != step + 1:
                raise ValueError('Overlapping decisions or misaligned execution receipt')
            if not np.array_equal(action, commands[offset]):
                raise ValueError('Saved execution differs from the actual dispatched action')
            if not np.array_equal(np.asarray(receipt['obs']['states'],np.float32),states[step+1]):
                raise ValueError('Receipt post-state differs from next observation')
            valid = bool(receipt['valid'] and np.isfinite(action).all()
                         and np.all((action[[6,13]] >= 0) & (action[[6,13]] <= 1)))
            tracking = float(np.max(np.abs(action[JOINTS] - states[step+1, JOINTS])))
            rows[step] = dict(episode_id=episode_id, step_id=step,
                observation_id=f'{episode_id}:{step}', next_observation_id=f'{episode_id}:{step+1}',
                action=action.tolist(), action_source=source, decision_id=decision_id,
                prediction_id=record['prediction_id'], privileged_context_active=step>=first_exposure,
                valid=valid, terminated=receipt['terminated'], truncated=receipt['truncated'],
                diagnostics_ref=str(receipt_path), post_action_tracking_error_rad=tracking,
                **{key:receipt.get(key) for key in MAPPING_KEYS})
            if not valid or tracking > .12:
                quality[step] = False
                excluded.append(dict(step_ranges=[[step,step+1]], reason='invalid_receipt' if not valid else 'joint_tracking_error_above_0.12rad'))
        motion = record.get('motion_result')
        if motion and motion['status'] not in ('waypoints_reached_not_grasp_verified','native_terminal_or_decision_limit'):
            quality[start:end] = False
            excluded.append(dict(step_ranges=[[start,end]], reason='motion_'+motion['status']))
        assessment = response['assessment']
        decision = dict(decision_id=decision_id, episode_id=episode_id,
            observation_id=f'{episode_id}:{start}', prediction_id=record['prediction_id'],
            start_step=start, end_step_exclusive=end, mode=response['mode'],
            requested_steps=response['steps'], executed_steps=end-start,
            student_proposal_steps=record.get('student_proposal_steps'),
            force_hold_steps=record.get('force_hold_steps',0),tighten_steps=record.get('tighten_steps',0),
            force_hold_event_ids=[e['event_id'] for e in record.get('force_hold_events',[])],
            reason=response['reason'], raw_response_ref=str(archive/'controller'/f'response_{index:03d}.json'),
            proposal_ref=str(proposal_path), takeover_trigger=record.get('takeover_trigger'),
            privileged_context_active=start>=first_exposure,
            privileged_query_refs=[q['query_id'] for q in queries if q['step_id']==start],
            pi05_infer_wall_s=request.get('inference_seconds'), teacher_round_wall_s=None,
            execute_wall_s=None, validation_reject_count=rejected[request['request_id']],
            **{k:assessment.get(k) for k in ('current_subgoal','execution_status','execution_evidence','intent_status','intent_evidence','expected_next_intent','predicted_next_intent')})
        decisions.append(decision)
        segments.append(dict(segment_id=decision_id+':segment', episode_id=episode_id,
            start_step=start, end_step_exclusive=end, decision_ids=[decision_id],
            subtask=assessment.get('current_subgoal'), failure_reason=None,
            label_source='gpt_existing', label_available_at_step=start, label_version='existing_assessment_v1'))
    if any(row is None for row in rows):
        raise ValueError('Executed controls missing a decision')
    # Existing assessments concern the preceding decision. Their later labels
    # may filter training data, but must never become earlier student inputs.
    for previous,current in zip(history,history[1:]):
        assessment=current['assessment']
        if assessment['execution_status']=='failed':
            start,end=previous['start_tick'],previous['end_tick']
            quality[start:end]=False
            excluded.append(dict(step_ranges=[[start,end]],
                reason='teacher_reported_previous_execution_failure',
                label_source='gpt_existing_not_independent_ground_truth',
                label_available_at_step=current['start_tick'],
                evidence=assessment['execution_evidence']))
            segment=segments[previous['decision']]
            segment['failure_reason']=assessment['execution_evidence']
            segment['label_available_at_step']=current['start_tick']
    native_stages = None
    if case['task'] == 'deposit_coin':
        native_stages = [json.loads(line) for line in (archive/'sim/native_stages.jsonl').open()]
        if [r['step_id'] for r in native_stages] != list(range(count+1)):
            raise ValueError('Native stage evidence is incomplete')
        best = 0; last_progress = 0
        for stage in native_stages:
            level = (3 if stage['inserted'] and stage['arms_home'] else
                     2 if stage['inserted'] else 1 if stage['lifted'] else 0)
            if level > best:
                best = level; last_progress = stage['step_id']
        # A failed episode is retained in full. Its unverified tail is excluded
        # from behavior cloning, not relabelled as a successful correction.
        if not outcome['native_success']:
            quality[last_progress:] = False
            excluded.append(dict(step_ranges=[[last_progress,count]],
                reason='failed_episode_tail_after_last_native_stage_progress',
                label_source='native_offline_predicates',label_available_at_step=count))
    starts = full_windows(quality, horizon)
    used = np.zeros(count, bool)
    for start in starts:
        used[start:start+horizon] = True
    if np.any(quality & ~used):
        excluded.append(dict(step_ranges=ranges(np.flatnonzero(quality & ~used)),
                             reason='not_in_any_complete_valid_action_window'))
    quality &= used
    counts = Counter(row['action_source'] for row in rows)
    eligible_counts = Counter(row['action_source'] for row, ok in zip(rows,quality) if ok)
    def total(stage):
        values = [s['duration_s'] for s in spans if s['stage']==stage and s['status']=='ok']
        return sum(values) if values else None
    reset_spans = [s for s in spans if s['stage']=='rpc.reset' and s['status']=='ok']
    movement_spans = [s for s in spans if s['stage'] in ('rpc.chunk_step','rpc.execute_reference_motion','rpc.execute_joint_motion','rpc.eef_short_motion') and s['status']=='ok']
    loop_wall = None
    if len(reset_spans)==1 and movement_spans:
        last = max(movement_spans,key=lambda s:s['mono'])
        if last['writer']==reset_spans[0]['writer']:
            loop_wall = last['mono']-reset_spans[0]['mono']
    retries = [s for s in spans if s['stage'].startswith('retry_wait.')]
    costs = dict(started_at=None, finished_at=None, wall_time_limit_s=identity['max_seconds'],
        episode_wall_s=None, startup_wall_s=None, control_loop_wall_s=loop_wall,
        finalization_wall_s=None, pi05_infer_wall_s_total=total('pi05_infer_and_save'),
        teacher_round_wall_s_total=sum(s['duration_s'] for s in spans if s['stage'].startswith('teacher_to_')),
        execute_wall_s_total=total('service.robodojo_execute'), teacher_retry_wait_s_total=sum(s['duration_s'] for s in retries),
        pi05_infer_count=len(list((archive/'controller').glob('proposal_*.npz'))),
        teacher_round_count=sum(s['stage'].startswith('teacher_to_') for s in spans),
        teacher_retry_count=len(retries), privileged_query_count=len(delivered), token_usage=None,
        source_ref=str(archive/'timing/controller.jsonl'), missing_values_are_unmeasured=True)
    experiment_root = next(p for p in archive.parents if p.parent.name=='results')
    location_path = archive/'archive_location.json'
    if location_path.exists():
        batch_path = Path(read(location_path)['batch_path'])
    else:
        batch_path = experiment_root/'_replicas'/archive.parent.name/archive.name/'batch.json'
    batch = read(batch_path)
    lifecycle = next(r for r in batch['episodes'] if Path(r['archive']).resolve()==archive)
    costs.update(started_at=lifecycle['started_utc'],finished_at=lifecycle.get('finished_utc'),
                 token_usage=lifecycle.get('token_usage'))
    snapshot = read(archive/'source_snapshot/source_manifest.json')
    teacher_config = dict(model=run['teacher_model'],effort=run['teacher_reasoning_effort'],
                          prompt_sha256=run['teacher_prompt_sha256'])
    episode = dict(schema_version='rollout.v1', episode_id=episode_id,
        experiment_id=archive.parts[archive.parts.index('results')+1],
        attempt_id=archive.name, repeat_id=(read(location_path)['repeat_id'] if location_path.exists() else None), task_id=case['task'],
        **{k:case[k] for k in ('runtime_task','variant','case_id','eval_seed','layout_id','reset_seed')},
        task_instruction=reset['instruction'], student_checkpoint_id=run['student_identity_sha256'],
        teacher_config_id=hashlib.sha256(json.dumps(teacher_config,sort_keys=True).encode()).hexdigest(),
        code_version=dict(git_commit=collection_commit, dirty=False, snapshot_sha256=snapshot['sha256']),
        config_id=digest(archive/'resume_identity.json'), robot_id='dual_arx_x5',
        action_schema_id='dual_arx_x5_absolute_joint_rad_opening01_14_v1',
        state_schema_id='dual_arx_x5_measured_joint_rad_commanded_opening01_14_v1',control_dt_s=.04,
        teacher_invoked=bool(history), privileged_exposed=bool(delivered),
        control_class='privileged_assisted' if delivered else 'gpt_assisted' if counts['gpt_edit']+counts['gpt_eef'] else 'force_hold_assisted' if force_hold_intervened else 'student_only' if count else 'no_execution',
        force_hold_assisted=force_hold_intervened,force_hold_intervened=force_hold_intervened,
        force_hold_enabled=bool(run.get('pi_force_hold',{}).get('enabled',False)),
        pi_force_hold=run.get('pi_force_hold'),force_hold_steps=force_hold_steps,
        student_proposal_steps=counts['student']-force_hold_steps,
        student_steps=counts['student'],gpt_edit_steps=counts['gpt_edit'],gpt_eef_steps=counts['gpt_eef'],
        termination_reason='native_success' if outcome['native_success'] else 'native_failure',
        native_terminated=summary['terminated'],native_truncated=summary['truncated'],
        native_success=outcome['native_success'],native_score=outcome.get('native_score'),
        error_detail=None,artifact_status=artifact['status'],cost=costs)
    manifest = dict(dataset_version='flywheel.v1',filter_version=FILTER_VERSION+('_deposit_native_progress_v1' if native_stages is not None else ''),split='train',
        episode_id=episode_id,episode_ref='episode.json',archive=str(archive),
        task_instruction=reset['instruction'], step_ranges=ranges(np.flatnonzero(quality)),
        eligible_steps=int(quality.sum()),exclusion_reasons=excluded,label_version=None,sampling_weight=.5,
        horizon=horizon,chunk_start_steps=starts,chunk_count=len(starts),
        tail_policy='drop incomplete or excluded-command windows; never invent padded labels',
        action_label_semantics='actual_dispatched_native_commands',
        student_input_allowlist=[*CAMERAS,'states','task_instruction'],
        force_hold_schema='pi_force_hold_event_v1',force_hold_events_ref='force_hold_events.json',
        event_base_targets_ref='event_base_targets.json',
        diagnostics_policy='force and measured fingers remain diagnostics; action labels are actual commands',
        diagnostics_alignment='observation t contains previous action force window; action t uses post-observation t+1 force',
        training_ready=bool(starts) and not validation_only,validation_only=validation_only,
        training_case_ids=training_case_ids or [training_case_id],heldout_case_ids=(heldout_case_ids if heldout_case_ids is not None else [c['case_id'] for c in read(archive/'evaluation_manifest.json')['cases'] if c['case_id']!=training_case_id]),
        source_counts=dict(counts),eligible_source_counts=dict(eligible_counts))
    task_summary = dict(task_id=case['task'],attempted_episodes=1,executed_steps=count,
        eligible_steps=int(quality.sum()),eligible_student_steps=eligible_counts['student'],
        eligible_gpt_steps=eligible_counts['gpt_edit']+eligible_counts['gpt_eef'],
        timeout_episodes=int(summary['truncated']),collection_wall_s=None,
        eligible_steps_per_hour=None,target_steps=None,remaining_steps=None,
        target_complete_episodes=1,quota_note='One complete episode export; cumulative inventory is stored separately by the campaign.')
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('episode',episode),('observations',observations),('steps',rows),('decisions',decisions),('segments',segments),('privileged_queries',queries),('manifest',manifest),('task_summary',task_summary)]:
        write(output/f'{name}.json',value)
    write(output/'force_hold_events.json',event_rows)
    write(output/'event_base_targets.json',resolved_bases)
    if native_stages is not None:
        write(output/'native_stages.json',native_stages)
    write(output/'teacher_config.json',teacher_config)
    write(output/'schemas.json',dict(action=dict(id=episode['action_schema_id'],
        dimension=14,order=['left_joint_1..6','left_gripper','right_joint_1..6','right_gripper'],
        joint_units='rad',joint_semantics='absolute_native_target',gripper='continuous 0 closed, 1 open',normalized=False),
        state=dict(id=episode['state_schema_id'],dimension=14,joints='measured radians',
            gripper='normalized internal target after feedback-based limiting; not measured finger position'),
        label_provenance=dict(online='actual dispatched commands',demo='accepted official next-frame measured state')))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--collection-commit',required=True)
    parser.add_argument('--training-case-id',required=True)
    args=parser.parse_args()
    print(json.dumps(export_archive(**vars(args)),ensure_ascii=False))


if __name__=='__main__':
    main()
