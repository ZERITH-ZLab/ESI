from copy import deepcopy
import json

import pytest

from .input_view import teacher_view
from . import run
from .output_contract import decision_id


def packet():
    pose = dict(position=[.123456789, -.234567891, .987654321],
                quaternion_wxyz=[.70710678, 0., 0., .70710678],
                gripper_opening=.1234567, gripper_closed=True, rpy_deg=[0., 0., 90.])
    assessment = dict(current_subgoal='grasp', execution_evidence='not held')
    prior = dict(assessment=assessment, response=dict(assessment=assessment,
        reason='retry grasp', mode='eef', target=dict(left=pose, right=pose)),
        executed_steps=5, executed_gripper_closed=[dict(left=True, right=False)]*5,
        source_detail='eef', native_success=False, terminal=False,
        simulator_decision_receipt={'status': 'received'}, prediction_id='old')
    return dict(student_eef_trajectory=[dict(index=i, left=deepcopy(pose),
        right=deepcopy(pose)) for i in range(50)], current_eef=dict(left=pose, right=pose),
        previous_result=prior, previous_observation=dict(step_id=10, images=[],
            current_state=list(range(14)), executed_steps=5,
            executed_gripper_closed=deepcopy(prior['executed_gripper_closed']), source_detail='eef'),
        images=[dict(camera=c, path=c+'.png') for c in ('head','left','right')],
        next_call={'tool': 'robodojo_execute', 'proposal_path': '/raw/actions.npz'},
        request_id='fresh', rollout_finished=False)


def test_shapes_precision_and_nonmutation():
    original = packet()
    saved = deepcopy(original)
    view = teacher_view(original)
    assert original == saved
    rows = view['student_eef_trajectory']['rows']
    assert len(rows) == 50
    assert [r[0] for r in rows] == list(range(50))
    for r in rows:
        assert len(r) == 3 and len(r[1]) == len(r[2]) == 8
    assert rows[0][1] == [.1235, -.2346, .9877, .70711, 0., 0., .70711, .123]
    assert view['current_eef'] == original['current_eef']
    assert view['decision_id'] == decision_id('fresh') and 'request_id' not in view
    assert 'proposal_path' not in view['next_call']
    assert view['previous_result']['assessment'] == original['previous_result']['assessment']
    assert 'assessment' not in view['previous_result']['response']
    assert 'executed_steps' not in view['previous_observation']
    assert view['previous_result']['response']['target'] == original['previous_result']['response']['target']


def test_nonduplicate_history_is_preserved():
    p = packet()
    p['previous_result']['response']['assessment'] = {'different': True}
    p['previous_observation']['executed_steps'] = 4
    v = teacher_view(p)
    assert v['previous_result']['response']['assessment'] == {'different': True}
    assert v['previous_observation']['executed_steps'] == 4


@pytest.mark.parametrize('p', [{}, {'error':'bad input','no_execution':True},
                             {'rollout_finished':True,'result':{'success':False}}])
def test_packets_without_proposals(p):
    assert teacher_view(p) == dict(p, output_contract_version='mode_specific_v1')


def test_actual_content_items_use_view_and_same_images(monkeypatch):
    p = packet(); saved = deepcopy(p)
    monkeypatch.setattr(run, 'prepare_image', lambda item, edge: (dict(item), b'png'))
    items = run.content_items(p, images=True)
    assert p == saved
    assert len(items) == 4
    v = json.loads(items[0]['text'])
    assert len(v['student_eef_trajectory']['rows']) == 50
    assert v['images'] == p['images']
    assert all(i['type'] == 'inputImage' for i in items[1:])


def test_incomplete_or_reordered_fk_rejected():
    p = packet(); p['student_eef_trajectory'].pop()
    with pytest.raises(ValueError): teacher_view(p)
    p = packet(); p['student_eef_trajectory'][1]['index'] = 9
    with pytest.raises(ValueError): teacher_view(p)
