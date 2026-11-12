"""Read-only force-hold provenance and post-observation diagnostic helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('left', 'right')
FINGERS = ('link7', 'link8')
MAPPING_KEYS = ('proposal_action_index', 'trigger_proposal_action_index', 'hold_event_id')


def requested_action(receipt):
    """Validate a zero override; recover the immutable pre-override command.

    Training still consumes executed_action, never this diagnostic command.
    """
    actual = np.asarray(receipt['executed_action'], np.float32)
    record = receipt.get('gripper_override')
    if record is None:
        return actual
    original = np.asarray(record.get('original_action'), np.float32)
    if (record.get('version') != 'contact_zero_v4' or original.shape != (14,)
            or actual.shape != (14,) or not np.isfinite(original).all()
            or np.any(original[[6, 13]] < 0) or np.any(original[[6, 13]] > 1)
            or record.get('observation_step_id') != receipt['step_id']-1
            or not record.get('arms') or set(record['arms']) - set(ARMS)):
        raise ValueError('Invalid zero gripper override provenance')
    expected = original.copy()
    for arm, details in record['arms'].items():
        index = ARMS.index(arm)*7+6
        contact = details.get('contact', {})
        if (details.get('original_target') != float(original[index])
                or details.get('issued_target') != 0.
                or not isinstance(details.get('actor'), str)
                or not details['actor'].startswith('/')
                or type(details.get('activation_step_id')) is not int
                or not 1 <= details['activation_step_id'] < receipt['step_id']
                or contact.get('valid') is not True or contact.get('actor') != details['actor']
                or contact.get('step_id') != receipt['step_id']-1):
            raise ValueError('Invalid zero gripper contact/target provenance')
        expected[index] = 0.
    if not np.array_equal(actual, expected):
        raise ValueError('Zero gripper override changes an arm or an unlisted gripper')
    return original


def observation_diagnostics(obs, step):
    """Return diagnostic arrays; unknown numeric entries remain NaN, never zero."""
    finger = np.full((2, 2), np.nan)
    finger_valid = False
    if bool(obs.get('finger_positions_available', False)):
        value = np.asarray(obs.get('finger_positions_m', []), float)
        if int(obs.get('finger_positions_step_id', -1)) != step or value.shape != (2, 2) or not np.isfinite(value).all():
            raise ValueError(f'Misaligned or invalid measured fingers at observation {step}')
        finger, finger_valid = value, True
    mean, peak, ratio = np.full((2, 2, 3), np.nan), np.full((2, 2), np.nan), np.full((2, 2), np.nan)
    valid = np.asarray(obs.get('gripper_contact_valid_mask', np.zeros((2, 2), bool)), bool)
    if valid.shape != (2, 2):
        raise ValueError('Invalid force validity mask')
    if (bool(obs.get('gripper_contact_valid', False)) or valid.any()) and int(obs.get('gripper_contact_step_id', -1)) != step:
        raise ValueError(f'Misaligned force window at observation {step}')
    if step == 0 and valid.any():
        raise ValueError('Reset observation cannot contain an executed force window')
    for a, arm in enumerate(ARMS):
        for f, name in enumerate(FINGERS):
            if not valid[a, f]:
                continue
            prefix = f'gripper_contact_{arm}_{name}_'
            vector = np.asarray(obs[prefix+'force_world_mean_N'], float)
            p, r = float(obs[prefix+'force_magnitude_peak_N']), float(obs[prefix+'contact_ratio'])
            if vector.shape != (3,) or not np.isfinite(vector).all() or not np.isfinite([p, r]).all() or p < 0 or not 0 <= r <= 1:
                raise ValueError('Invalid valid force measurement')
            mean[a, f], peak[a, f], ratio[a, f] = vector, p, r
    return dict(finger_positions_m=finger, finger_positions_valid=np.asarray(finger_valid),
                force_world_mean_N=mean, force_magnitude_peak_N=peak, contact_ratio=ratio,
                force_valid_mask=valid.copy())


def json_diagnostics(values):
    """Convert explicit unknown array entries to JSON null with their validity mask."""
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    return {key: convert(np.asarray(value).tolist()) for key, value in values.items()}


def read_hold_events(episode):
    """Load append-only lifecycle records; a later record seals an earlier event."""
    path = Path(episode)/'pi_force_hold_events.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    events = {}
    for row in rows:
        if row.get('schema') != 'pi_force_hold_event_v1' or not row.get('event_id'):
            raise ValueError('Invalid force-hold event schema or identity')
        previous = events.get(row['event_id'])
        if previous is not None:
            immutable = ('schema', 'event_id', 'student_prediction_id', 'trigger_proposal_action_index',
                         'insert_before_proposal_action_index', 'insert_after_step_id', 'held_action',
                         'original_action_count', 'discarded_proposal_range', 'version',
                         'base_held_action', 'base_model_action', 'base_prediction_id',
                         'base_proposal_action_index', 'base_model_step_id', 'raw_target_ref',
                         'base_clipped_model_target', 'margin_plans', 'settings', 'reused_model_hold_samples')
            if any(previous.get(key) != row.get(key) for key in immutable):
                raise ValueError('Force-hold lifecycle rewrites immutable event evidence')
            if (previous.get('status') == 'ended' or row.get('actual_count', -1) < previous.get('actual_count', 0)
                    or row.get('steps', [])[:len(previous.get('steps', []))] != previous.get('steps', [])):
                raise ValueError('Force-hold lifecycle loses ACKs or continues after sealing')
        events[row['event_id']] = row
    return rows, events


def validate_hold_mapping(episode, receipts):
    """Validate ACK identities, immutable original indices and sealed hold counts."""
    rows, events = read_hold_events(episode)
    event_steps = {}
    previous_index, prediction = -1, None
    for step, receipt in enumerate(receipts, 1):
        if receipt.get('step_id') != step:
            raise ValueError('Non-contiguous action ACK index')
        requested_action(receipt)
        if not any(key in receipt for key in MAPPING_KEYS):
            continue  # Historical archives predate optional mapping.
        event_id = receipt.get('hold_event_id')
        original = receipt.get('proposal_action_index')
        if event_id is not None:
            if original is not None or event_id not in events:
                raise ValueError('Hold ACK has an original proposal index or missing event')
            event = events[event_id]
            if receipt.get('student_prediction_id') != event.get('student_prediction_id'):
                raise ValueError('Hold ACK prediction identity mismatch')
            if receipt.get('trigger_proposal_action_index') != event['trigger_proposal_action_index']:
                raise ValueError('Hold trigger index mismatch')
            if event.get('version') in ('contact_tighten_v2', 'contact_margin_v3', 'contact_zero_v4'):
                samples = [sample for sample in event.get('samples', []) if sample.get('step_id') == step]
                if len(samples) != 1 or not np.array_equal(
                        np.asarray(receipt['executed_action'], np.float32),
                        np.asarray(samples[0]['actual_command'], np.float32)):
                    raise ValueError('Tighten ACK differs from its per-step actual command')
                if receipt.get('hold_compensation') != samples[0].get('compensation'):
                    raise ValueError('Tighten ACK compensation differs from event sample')
                if not np.array_equal(np.asarray(receipt.get('hold_base_action'), np.float32),
                                      np.asarray(event['base_held_action'], np.float32)):
                    raise ValueError('Tighten ACK rewrites the model base command')
            elif not np.array_equal(np.asarray(receipt['executed_action'], np.float32), np.asarray(event['held_action'], np.float32)):
                raise ValueError('Hold action differs from frozen event command')
            event_steps.setdefault(event_id, []).append(step)
        elif original is not None:
            current = receipt.get('student_prediction_id')
            if any(event.get('student_prediction_id') == current and step > event['insert_after_step_id']
                   for event in events.values()):
                raise ValueError('Original chunk resumed after a force-hold reinference boundary')
            if current != prediction:
                previous_index, prediction = -1, current
            if type(original) is not int or original != previous_index + 1:
                raise ValueError('Original proposal index skipped or consumed by hold')
            previous_index = original
    for event_id, event in events.items():
        actual = event_steps.get(event_id, [])
        if event.get('steps') != actual or event.get('actual_count') != len(actual):
            raise ValueError('Force-hold event ACK inventory mismatch')
        insert = event['insert_after_step_id']
        if type(insert) is not int or not 0 <= insert <= len(receipts):
            raise ValueError('Invalid hold insertion ACK')
        before = event.get('trigger_before')
        if before is not None:
            if before.get('step_id') != insert:
                raise ValueError('Hold trigger observation index mismatch')
            force_step = before.get('force_window_step_id')
            if force_step is not None and (type(force_step) is not int or not 0 <= force_step <= insert):
                raise ValueError('Hold trigger uses invalid or future execution force')
            if event.get('trigger_before_observation') != f'observations/{insert:06d}.npz':
                raise ValueError('Hold trigger observation reference mismatch')
        index = event['trigger_proposal_action_index']
        if type(index) is not int or not 0 <= index < event['original_action_count']:
            raise ValueError('Invalid preempted proposal index')
        if event.get('insert_before_proposal_action_index') != index:
            raise ValueError('Ambiguous force-hold insertion index')
        if index > 0:
            if insert == 0:
                raise ValueError('Missing ACK before force-hold insertion')
            previous = receipts[insert-1]
            if previous.get('proposal_action_index') != index-1 or previous.get('student_prediction_id') != event.get('student_prediction_id'):
                raise ValueError('Force-hold insertion does not follow its proposal prefix')
        if actual != list(range(insert+1, insert+len(actual)+1)):
            raise ValueError('Non-contiguous inserted hold ACKs')
        discard = event['discarded_proposal_range']
        if discard != [index, event['original_action_count']]:
            raise ValueError('Invalid original proposal discard range')
        if event.get('status') in ('started', 'pending', 'running') or not event.get('end_reason'):
            raise ValueError('Unsealed force-hold event')
        if event.get('version') in ('contact_tighten_v2', 'contact_margin_v3', 'contact_zero_v4'):
            validate_tighten_samples(event, receipts)
    return rows, events


def validate_proposal_mapping(receipts, actions, events=None, prediction_id=None):
    """Check every consumed original action against its unchanged clipped proposal."""
    actions = np.asarray(actions, np.float32)
    for event in (events or {}).values():
        if event.get('student_prediction_id') == prediction_id:
            if event['original_action_count'] != len(actions):
                raise ValueError('Hold original length differs from immutable proposal')
            index = event['trigger_proposal_action_index']
            if 'trigger_action' in event and not np.array_equal(
                    np.asarray(event['trigger_action'], np.float32), actions[index]):
                raise ValueError('Hold trigger differs from immutable original proposal')
    for receipt in receipts:
        index = receipt.get('proposal_action_index')
        if index is None:
            continue
        if type(index) is not int or not 0 <= index < len(actions):
            raise ValueError('Original proposal index outside immutable prediction')
        if not np.array_equal(requested_action(receipt), actions[index]):
            raise ValueError('Original executed action differs from its immutable proposal')


def validate_tighten_samples(event, receipts):
    """Check variable tighten commands, physical-unit budgets and force timing."""
    base = np.asarray(event['base_held_action'], np.float32)
    if base.shape != (14,) or not np.isfinite(base).all():
        raise ValueError('Invalid frozen contact-tighten base')
    settings = event['settings']
    scale = float(event['single_finger_scale_m'])
    modern = event.get('version') in ('contact_margin_v3', 'contact_zero_v4')
    if modern:
        from .robodojo_server.pi_force_hold import configured, margin_plan
        config = configured(settings)
        if scale != .054:
            raise ValueError('Invalid margin mapping scale')
        for arm in event['closing_arms']:
            plan = event['margin_plans'][arm]
            ref_step = plan['reference_step_id']
            if not 1 <= ref_step <= event['insert_after_step_id']:
                raise ValueError('Margin reference is not an executed observation')
            ref = receipts[ref_step-1]
            i = ARMS.index(arm)
            if ref.get('hold_event_id') is not None:
                raise ValueError('Margin reference cannot be an already boosted ACK')
            expected_plan = margin_plan(config, ref['obs']['finger_positions_m'][i],
                                       requested_action(ref)[i*7+6], enabled=plan['tightening_enabled'])
            for key, value in expected_plan.items():
                if isinstance(value, bool):
                    valid = plan.get(key) is value
                else:
                    valid = np.allclose(plan.get(key), value, atol=1e-9, rtol=0)
                if not valid:
                    raise ValueError('Frozen margin plan differs from original action/physical feedback: '+key)
            remaining = max(0., 2*(-.010+float(base[i*7+6])*scale)
                            - plan['requested_final_target_width_m'])
            if abs(plan['remaining_extra_gap_m']-remaining) > 1e-9:
                raise ValueError('Remaining margin compensation was rebased')
        ramp_active = any(p['remaining_extra_gap_m'] > 1e-9 for p in event['margin_plans'].values())
    else:
        gap_step, cap = settings['gap_step_m'], settings['max_gap_reduction_m']
        if scale != .054 or not 0 < gap_step <= .001 or not 0 < cap <= .005:
            raise ValueError('Invalid single-finger scale or total-gap budget')
    samples = event.get('samples', [])
    if len(samples) != event['actual_count'] or len(samples) > min(8, settings['max_extra_steps']):
        raise ValueError('Tighten sample/extra-ACK budget mismatch')
    if [sample['step_id'] for sample in samples] != event['steps']:
        raise ValueError('Tighten sample indices differ from actual ACKs')
    previous_counts = {arm: 0 for arm in event['closing_arms']}
    previous_actual = {arm: 0. for arm in event['closing_arms']}
    increases = 0
    for sample_number, sample in enumerate(samples, 1):
        step = sample['step_id'];command = requested_action(receipts[step-1])
        if command.shape != (14,) or not np.isfinite(command).all():
            raise ValueError('Invalid per-ACK tighten command')
        if sample.get('force_window_step_id') != step:
            raise ValueError('Tighten post-force window is misaligned')
        for prefix, expected_step in (('pre', step-1), ('post', step)):
            if (sample.get(prefix+'_observation_step_id') != expected_step
                    or sample.get(prefix+'_observation_ref') != f'observations/{expected_step:06d}.npz'):
                raise ValueError('Tighten pre/post observation reference is misaligned')
            for contact in sample[prefix+'_contact'].values():
                if contact.get('step_id') != expected_step:
                    raise ValueError('Tighten pre/post force window is misaligned')
        post = receipts[step-1]['obs']
        if not np.array_equal(np.asarray(sample['post_internal_target_state'], np.float32),
                              np.asarray(post['states'], np.float32)):
            raise ValueError('Tighten post internal state differs from actual ACK')
        if sample['post_finger_positions_m'] is not None and not np.array_equal(
                np.asarray(sample['post_finger_positions_m'], float),
                np.asarray(post.get('finger_positions_m'), float)):
            raise ValueError('Tighten post measured fingers differ from actual ACK')
        if step > 1:
            pre = receipts[step-2]['obs']
            if not np.array_equal(np.asarray(sample['pre_internal_target_state'], np.float32),
                                  np.asarray(pre['states'], np.float32)):
                raise ValueError('Tighten pre sensor state contains future or stale values')
            if not np.array_equal(np.asarray(sample['pre_finger_positions_m'], float),
                                  np.asarray(pre.get('finger_positions_m'), float)):
                raise ValueError('Tighten pre measured fingers are misaligned')
        if not np.array_equal(command[[*range(6), *range(7, 13)]], base[[*range(6), *range(7, 13)]]):
            raise ValueError('Tightening changes frozen arm joints')
        compensation = sample['compensation'];increased = False
        if set(compensation) != set(previous_counts):
            raise ValueError('Tighten compensation arms differ from triggered arms')
        for a, arm in enumerate(ARMS):
            index = a * 7 + 6
            if arm not in compensation:
                if command[index] != base[index]:
                    raise ValueError('Inactive gripper changed during tightening')
                continue
            value = compensation[arm];count = value['increase_count']
            if modern:
                expected_count = min(sample_number, 1 if config.tighten_mode == 'zero' else config.ramp_steps) if ramp_active else 0
                if type(count) is not int or count != expected_count:
                    raise ValueError('Invalid margin ramp stage')
                requested = event['margin_plans'][arm]['remaining_extra_gap_m'] * count / (1 if config.tighten_mode == 'zero' else config.ramp_steps)
            else:
                if type(count) is not int or not previous_counts[arm] <= count <= previous_counts[arm]+1 or count > 5:
                    raise ValueError('Invalid tighten increment count')
                requested = min(count * gap_step, cap)
            previous_counts[arm] = count
            expected = np.clip(float(base[index])-requested/(2*scale), 0, 1)
            actual = float(base[index])-float(command[index])
            increased |= actual > previous_actual[arm] + 1e-10
            previous_actual[arm] = actual
            expected_values = dict(requested_cumulative_gap_reduction_m=requested,
                actual_cumulative_gap_reduction_m=actual*2*scale,
                requested_normalized_reduction=requested/(2*scale),
                actual_normalized_reduction=actual,base_model_target=float(base[index]),
                issued_target=float(command[index]))
            if abs(float(command[index])-expected) > 1e-7 or any(
                    not np.isfinite(value[key]) or abs(value[key]-target) > 1e-7
                    for key, target in expected_values.items()):
                raise ValueError('Tighten requested/clipped compensation is inconsistent')
        increases += bool(increased)
        if bool(receipts[step-1].get('hold_tightening_increased')) != bool(increased):
            raise ValueError('Tighten increase flag differs from actual increment history')
    if event.get('tighten_steps') != increases:
        raise ValueError('Tighten-step count includes unchanged hold commands')


def resolve_base_model_targets(events, proposal_paths, receipts):
    """Resolve v2 raw/base references without modifying original prediction files."""
    resolved = []
    for event in events.values():
        if event.get('version') not in ('contact_tighten_v2', 'contact_margin_v3', 'contact_zero_v4'):
            continue
        prediction = event['base_prediction_id'];index = event['base_proposal_action_index']
        if prediction not in proposal_paths:
            raise ValueError('Tighten base prediction is absent from controller history')
        path = Path(proposal_paths[prediction])
        with np.load(path, allow_pickle=False) as data:
            raw, actions = data['raw_actions'], data['actions']
        if (raw.shape != actions.shape or actions.ndim != 2 or actions.shape[1] != 14
                or type(index) is not int or not 0 <= index < len(actions)):
            raise ValueError('Tighten base index is outside its original proposal')
        reference = dict(prediction_id=prediction, proposal_action_index=index,
                         field='raw_actions', gripper_indices=[6, 13])
        if event['raw_target_ref'] != reference:
            raise ValueError('Tighten raw-target reference is ambiguous or incorrect')
        step = event['base_model_step_id'];insert = event['insert_after_step_id']
        originals = [row for row in receipts[:insert]
                     if type(row.get('proposal_action_index')) is int and row.get('source') == 'student']
        if not originals or originals[-1]['step_id'] != step:
            raise ValueError('Tighten base is not the most recent original model ACK')
        receipt = originals[-1]
        if receipt.get('student_prediction_id') != prediction or receipt['proposal_action_index'] != index:
            raise ValueError('Tighten base provenance points at a different original command')
        if not np.array_equal(requested_action(receipt), actions[index]):
            raise ValueError('Tighten base ACK differs from its immutable prediction')
        if not np.array_equal(np.asarray(event['base_model_action'], np.float32), actions[index]):
            raise ValueError('Tighten event rewrites the last original model action')
        for a, arm in enumerate(ARMS):
            if not np.isclose(event['base_clipped_model_target'][arm], actions[index,a*7+6],atol=1e-7,rtol=0):
                raise ValueError('Tighten clipped base target differs from model proposal')
        for arm, evidence in event['trigger_evidence'].items():
            steps, history = evidence['executed_model_step_ids'], evidence['executed_model_history']
            if len(steps) != len(history):
                raise ValueError('Closing-intent model history lengths differ')
            for model_step, model_value in zip(steps, history):
                if (type(model_step) is not int or not 1 <= model_step <= insert
                        or type(receipts[model_step-1].get('proposal_action_index')) is not int
                        or receipts[model_step-1].get('hold_event_id') is not None):
                    raise ValueError('Auxiliary ACK contaminated closing-intent model history')
                if not np.isclose(model_value, requested_action(receipts[model_step-1])[ARMS.index(arm)*7+6],atol=1e-7,rtol=0):
                    raise ValueError('Closing-intent history rewrites an original model command')
        resolved.append(dict(event_id=event['event_id'],base_prediction_id=prediction,
            base_proposal_action_index=index,base_model_step_id=step,proposal_ref=str(path),
            proposal_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            raw_model_target={arm:float(raw[index,a*7+6]) for a,arm in enumerate(ARMS)},
            clipped_model_target={arm:float(actions[index,a*7+6]) for a,arm in enumerate(ARMS)}))
    return resolved
