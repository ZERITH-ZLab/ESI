"""Teacher-only projection. Never mutate execution data or archival packets."""
from copy import deepcopy
from .output_contract import decision_id, visible_next_call, OUTPUT_CONTRACT_VERSION

INPUT_VIEW_VERSION = 'compact_fk_v1'


def teacher_view(packet):
    visible = deepcopy(packet)
    visible['output_contract_version'] = OUTPUT_CONTRACT_VERSION
    if 'request_id' in visible:
        visible['decision_id'] = decision_id(visible.pop('request_id'))
    if visible.get('next_call') and visible['next_call']['tool'] in ('robodojo_start', 'robodojo_execute'):
        # Rejection packets already carry the compact next_call.
        if 'decision_id' not in visible['next_call']:
            visible['next_call'] = visible_next_call(visible['next_call'], packet)
    trajectory = visible.get('student_eef_trajectory')
    if trajectory is not None:
        if len(trajectory) != 50:
            raise ValueError('Compact FK requires all 50 proposal rows')
        rows = []
        for index, row in enumerate(trajectory):
            if row['index'] != index:
                raise ValueError('FK indices must be contiguous 0..49')
            arms = []
            for arm in ('left', 'right'):
                pose = row[arm]
                if len(pose['position']) != 3 or len(pose['quaternion_wxyz']) != 4:
                    raise ValueError('Invalid FK pose shape')
                arms.append([*[round(v, 4) for v in pose['position']],
                             *[round(v, 5) for v in pose['quaternion_wxyz']],
                             round(pose['gripper_opening'], 3)])
            rows.append([index, *arms])
        visible['student_eef_trajectory'] = dict(
            frame='environment_origin', eef_link='link6', position_unit='m',
            columns=['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'opening'],
            row_format=['step_index', 'left', 'right'], rows=rows)
        visible['input_view_version'] = INPUT_VIEW_VERSION

    previous = visible.get('previous_result')
    if isinstance(previous, dict):
        response = previous.get('response')
        if isinstance(response, dict):
            response.pop('request_id', None)
            if response.get('mode') != 'edit':
                response.pop('edit', None)
            if response.get('mode') != 'eef':
                response.pop('target', None)
        if (isinstance(response, dict) and 'assessment' in previous
                and response.get('assessment') == previous['assessment']):
            response.pop('assessment', None)
        # Receipt is audit evidence of decision receipt, not an execution ACK.
        # Preserve actual execution fields, native outcome and all reasoning.
        previous.pop('simulator_decision_receipt', None)
        previous.pop('prediction_id', None)
        observation = visible.get('previous_observation')
        if isinstance(observation, dict):
            for key in ('executed_steps', 'executed_gripper_closed', 'source_detail'):
                if key in previous and observation.get(key) == previous[key]:
                    observation.pop(key, None)
    return visible
