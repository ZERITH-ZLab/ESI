"""Compact teacher calls; expand only at the host execution boundary."""
from copy import deepcopy
import hashlib

from ..io import InputError
from ..skill.schema import _obj, response_schema


OUTPUT_CONTRACT_VERSION = 'mode_specific_v1'


def decision_id(request_id):
    return hashlib.sha256(request_id.encode()).hexdigest()[:16]


def compact_response_schema():
    original = response_schema()['properties']
    variants = []
    for mode in ('student', 'edit', 'eef', 'stop'):
        fields = dict(decision_id={'type': 'string'}, mode={'type': 'string', 'enum': [mode]},
                      steps=original['steps'], assessment=original['assessment'], reason=original['reason'])
        if mode in ('edit', 'eef'):
            fields['steps'] = {'type': 'integer', 'minimum': 1, 'maximum': 5}
            key = 'edit' if mode == 'edit' else 'target'
            fields[key] = original[key]
        variants.append(_obj(fields))
    return {'anyOf': variants}


def visible_next_call(next_call, request=None):
    if not next_call:
        return None
    if next_call['tool'] == 'robodojo_start':
        return dict(tool='robodojo_start', task=next_call['task'])
    if next_call['tool'] == 'robodojo_execute':
        return dict(tool='robodojo_execute', decision_id=decision_id(request['request_id']),
                    response='Supply assessment, reason, mode and steps; edit only for edit, target only for eef.')
    raise RuntimeError('Internal inference phase must not be exposed to teacher')


def next_call(rollout):
    if getattr(rollout, 'feedforward', False) and rollout.phase == 'infer':
        return rollout.planning_call()
    return visible_next_call(rollout.next_call(), getattr(rollout, 'request', None))


def expand_arguments(rollout, name, arguments):
    """Reject stale/extra fields before invoking the unchanged native validator."""
    expected = rollout.next_call()
    if not expected or expected['tool'] != name:
        raise InputError('Tool does not match the current phase; use latest next_call')
    if name == 'robodojo_start':
        if set(arguments) != {'task'} or arguments['task'] != expected['task']:
            raise InputError('Start requires only the configured task')
        return {k: expected[k] for k in ('task', 'output_dir')}
    if name != 'robodojo_execute' or set(arguments) != {'response'}:
        raise InputError('Execute accepts only response; paths are host-managed')
    response = arguments['response']
    if not isinstance(response, dict):
        raise InputError('response must be an object')
    mode = response.get('mode')
    if mode not in ('student', 'edit', 'eef', 'stop'):
        raise InputError('Unknown decision mode')
    keys = {'decision_id', 'mode', 'steps', 'assessment', 'reason'}
    if mode in ('edit', 'eef'):
        keys.add('edit' if mode == 'edit' else 'target')
    if set(response) != keys:
        raise InputError('Supply exactly the fields for the selected mode')
    request = rollout.request
    if response['decision_id'] != decision_id(request['request_id']):
        raise InputError('Stale decision_id; use the latest proposal, never replay')
    expanded = deepcopy(response)
    expanded.pop('decision_id')
    expanded['request_id'] = request['request_id']
    expanded.setdefault('edit', {arm: dict(delta_position=[0, 0, 0],
        delta_rotation_vector=[0, 0, 0], gripper='keep') for arm in ('left', 'right')})
    if 'target' not in expanded:
        expanded['target'] = {}
        for arm in ('left', 'right'):
            pose = request['current_eef'][arm]
            target = {key: deepcopy(pose[key]) for key in ('position', 'quaternion_wxyz')}
            # Measured EEF uses continuous opening; this inactive legacy field
            # is never executed in student/edit/stop mode.
            target['gripper_closed'] = (pose['gripper_closed'] if 'gripper_closed' in pose
                                        else bool(pose['gripper_opening_command'] < .5))
            expanded['target'][arm] = target
    return dict(proposal_path=expected['proposal_path'], output_dir=expected['output_dir'],
                response=expanded)
