"""Read-only scene RPC. Invoked on the simulator owner thread."""
from ..io import InputError

INFORMATION_ACCESS = 'gated_scene_query_v1'
FIELDS = ('pose', 'description', 'functional_points')


def simulator_query(session, arguments):
    if not session.scene_query_enabled:
        raise RuntimeError('Scene query disabled')
    if session.finished or session.terminated or session.truncated:
        raise RuntimeError('Cannot query a finished episode')
    from .visual_scene_query import VisualSceneQuery
    query = VisualSceneQuery(session.env)
    try:
        if set(arguments) != {'camera', 'identifiers', 'fields'}:
            raise InputError('Expected camera, identifiers, fields')
        if arguments['camera'] not in session.env.camera_manager.camera_names[0]:
            raise InputError('Unknown native camera')
        ids, fields = arguments['identifiers'], arguments['fields']
        if ids is not None and (not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids)):
            raise InputError('identifiers must be null or exact object IDs')
        if not isinstance(fields, list) or any(x not in FIELDS for x in fields) or len(fields) != len(set(fields)):
            raise InputError('Unsupported or duplicate fields')
        query._select(query._records(0), ids)
    except (InputError, KeyError, ValueError) as error:
        return dict(query_error=str(error), no_execution=True)
    packet = query.query(**arguments)
    for key in ('segment_to_object', 'segment_paths', 'unmapped_segments'):
        packet['visual'][key] = {str(k): v for k, v in packet['visual'][key].items()}
    return packet
