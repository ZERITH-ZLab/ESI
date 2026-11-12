from copy import deepcopy

import pytest

from ..io import InputError
from .output_contract import decision_id, expand_arguments, compact_response_schema
from .test_cycle import Rollout


def arguments(mode='student'):
    r = Rollout(); r.phase = 'execute'
    response = dict(decision_id=decision_id('fresh'), mode=mode, steps=3,
                    assessment={'evidence': 'unchanged'}, reason='same reason')
    if mode == 'eef':
        response['target'] = deepcopy(r.request['current_eef'])
    if mode == 'edit':
        response['edit'] = {arm: dict(delta_position=[.01, 0, 0],
            delta_rotation_vector=[0, 0, 0], gripper='keep') for arm in ('left', 'right')}
    return r, dict(response=response)


@pytest.mark.parametrize('mode', ['student', 'edit', 'eef', 'stop'])
def test_expansion_preserves_active_fields_and_original(mode):
    r, args = arguments(mode); before = deepcopy(args)
    expanded = expand_arguments(r, 'robodojo_execute', args)
    assert args == before
    assert expanded['proposal_path'] == str(r.proposal_path)
    out = expanded['response']
    for key, value in args['response'].items():
        if key != 'decision_id':
            assert out[key] == value
    assert out['request_id'] == 'fresh'
    assert {'edit', 'target'} <= out.keys()


@pytest.mark.parametrize('change', ['stale', 'path', 'extra_target', 'missing', 'phase'])
def test_invalid_calls_rejected_before_execution(change):
    r, args = arguments()
    if change == 'stale': args['response']['decision_id'] = 'old'
    if change == 'path': args['output_dir'] = '/override'
    if change == 'extra_target': args['response']['target'] = {}
    if change == 'missing': args['response'].pop('assessment')
    if change == 'phase': r.phase = 'done'
    with pytest.raises(InputError): expand_arguments(r, 'robodojo_execute', args)
    assert not r.calls


def test_schema_has_only_active_action_fields():
    for branch in compact_response_schema()['anyOf']:
        p = branch['properties']; mode = p['mode']['enum'][0]
        assert ('edit' in p) == (mode == 'edit')
        assert ('target' in p) == (mode == 'eef')
        assert 'request_id' not in p


def test_native_continuous_gripper_state_and_validator():
    from ..robodojo_server.client import validate_dual_response
    r, args = arguments()
    for pose in r.request['current_eef'].values():
        pose.pop('gripper_closed')
        pose['gripper_opening_command'] = .8
    response = expand_arguments(r, 'robodojo_execute', args)['response']
    assert response['target']['left']['gripper_closed'] is False
    validate_dual_response(response, r.request)
    response['steps'] = 16
    with pytest.raises(ValueError, match='execution steps'):
        validate_dual_response(response, r.request)


def test_eef_native_bounds_and_gate_still_apply():
    from ..robodojo_server.client import validate_dual_response
    r, args = arguments('eef')
    r.request['student_eef_trajectory'] = [deepcopy(r.request['current_eef'])]*50
    response = expand_arguments(r, 'robodojo_execute', args)['response']
    validate_dual_response(response, r.request)
    response['target']['left']['position'][0] = .1
    with pytest.raises(ValueError, match='5 cm'):
        validate_dual_response(response, r.request)
    response['target']['left']['position'][0] = 0
    r.request.update(gate_policy='failure-or-intent', step_id=1)
    with pytest.raises(ValueError, match='task_progress'):
        validate_dual_response(response, r.request)
