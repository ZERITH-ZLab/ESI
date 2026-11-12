"""Role-limited privileged geometry. Identity-only labels precede any pose query."""
from copy import deepcopy
from PIL import Image, ImageDraw
import numpy as np

from ..io import InputError, write_json
from ..skill.schema import _obj
from .scene_query import SceneQueryTools, allow_query
from .output_contract import decision_id, next_call

VERSION = 'role_query_v2'
EPISODE_BUDGET = 8
RECOVERY_BUDGET = 2
LABEL_BUDGET = 6


def tool_spec():
    return dict(type='function', name='robodojo_query',
        description='Read ONLY current recovery grasp-object and optional placement-object poses. '
                    'Requires gated identity image. Binding cannot change until student executes.',
        inputSchema=_obj(dict(decision_id={'type': 'string'},
            grasp_object_id={'type': 'string'},
            placement_object_id={'type': ['string', 'null']})))


class RoleQueryTools(SceneQueryTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery = None
        self.binding = None
        self.known_ids = set()
        self.recovery_queries = 0
        self.label_count = 0
        self.first_exposure_step = None
        self.exposures = []

    def recovery_key(self):
        return sum(1 for row in self.history if row.get('executed_steps', 0) > 0
                   and row['response']['mode'] == 'student')

    def start(self, **kwargs):
        packet = super().start(**kwargs)
        self.run['information_access'] = VERSION
        write_json(self.output/'run.json', self.run)
        return packet

    def finish(self, *args, **kwargs):
        result = super().finish(*args, **kwargs)
        result['information_access'] = VERSION
        write_json(self.output/'result.json', result)
        return result

    def access(self):
        active = self.phase == 'execute' and self.streak() >= 2
        return dict(version=VERSION, available=bool(active and self.known_ids
                    and self.query_count < EPISODE_BUDGET and self.recovery_queries < RECOVERY_BUDGET),
            consecutive_takeovers=self.streak(), threshold=2, binding=self.binding,
            queries=self.query_count, remaining_episode_queries=max(0, EPISODE_BUDGET-self.query_count),
            remaining_recovery_queries=max(0, RECOVERY_BUDGET-self.recovery_queries),
            label_snapshots=self.label_count, ever_queried=self.query_count > 0,
            ever_exposed=self.first_exposure_step is not None,
            first_exposure_step=self.first_exposure_step, first_query_step=self.first_query_step)

    def scene(self, ids, fields):
        cameras = self.meta['scene_query']['cameras']
        camera = next((c for c in ('cam_head', 'cam_high') if c in cameras), cameras[0])
        scene = self._rpc('scene_query', camera=camera, identifiers=ids, fields=fields)
        if 'query_error' in scene:
            raise RuntimeError('Scene identity/query failed: '+str(scene['query_error']))
        return camera, scene

    def record_exposure(self, kind, payload):
        if self.first_exposure_step is None:
            self.first_exposure_step = self.tick
        entry = dict(payload, kind=kind, episode_id=self.episode, step_id=self.tick,
            decision_id=decision_id(self.request['request_id']), recovery_id=self.recovery)
        self.exposures.append(entry)
        write_json(self.output/'privileged_exposures.json', self.exposures)
        self.audit_access()

    def infer(self, **kwargs):
        packet = super().infer(**kwargs)
        key = self.recovery_key()
        if key != self.recovery:
            self.recovery = key
            self.binding, self.known_ids, self.recovery_queries = None, set(), 0
        if self.streak() >= 2 and not self.known_ids and self.label_count < LABEL_BUDGET:
            camera, scene = self.scene(None, [])  # identities only; NEVER request all poses
            visual = scene['visual']
            directory = self.output/'scene_labels'/f'{self.label_count:03d}'
            directory.mkdir(parents=True, exist_ok=False)
            marked = Image.fromarray(visual['rgb']); draw = ImageDraw.Draw(marked)
            mapping = []
            for i, item in enumerate(visual['visible_objects'], 1):
                label = f'Q{i}'
                box = item['bbox_xyxy']
                draw.rectangle(tuple(box), outline='yellow', width=2)
                draw.text(tuple(box[:2]), label, fill='yellow', stroke_width=1, stroke_fill='black')
                mapping.append(dict(display_id=label, object_id=item['object_id']))
            marked.save(directory/'numbered.png')
            np.savez_compressed(directory/'evidence.npz', rgb=visual['rgb'], instance_ids=visual['instance_ids'])
            self.known_ids = {x['object_id'] for x in mapping}
            self.label_count += 1
            labels = dict(step_id=self.tick, objects=mapping, geometry_included=False)
            write_json(directory/'identities.json', labels)
            packet['scene_identities'] = labels
            packet['images'] = [*packet['images'], dict(camera=camera+'_identities', path=str(directory/'numbered.png'))]
            self.record_exposure('identity_image', labels)
        packet['scene_query'] = self.access()
        packet['privileged_history_exposed'] = self.first_exposure_step is not None
        return packet

    def query_scene(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) != {'decision_id','grasp_object_id','placement_object_id'}:
            raise InputError('Query requires decision_id, grasp_object_id, placement_object_id only')
        if not self.access()['available']:
            raise InputError('Query locked or budget exhausted')
        if arguments['decision_id'] != decision_id(self.request['request_id']):
            raise InputError('Stale decision_id')
        grasp, place = arguments['grasp_object_id'], arguments['placement_object_id']
        if not isinstance(grasp, str) or grasp not in self.known_ids:
            raise InputError('Grasp object must be an ID from the gated label image')
        if place is not None and (not isinstance(place, str) or place not in self.known_ids or place == grasp):
            raise InputError('Placement must be a different labeled object or null')
        binding = dict(grasp_object_id=grasp, placement_object_id=place)
        if self.binding is not None and binding != self.binding:
            raise InputError('Object roles are bound for this recovery; cannot enumerate other objects')
        ids = [grasp] + ([place] if place else [])
        _, scene = self.scene(ids, ['pose'])
        objects = [dict(id=o['id'], pose=o['fields']['pose']) for o in scene['objects'] if o['id'] in ids]
        self.binding = binding
        self.query_count += 1; self.recovery_queries += 1
        if self.first_query_step is None: self.first_query_step = self.tick
        packet = dict(episode_id=self.episode, step_id=self.tick,
            decision_id=arguments['decision_id'], roles=binding, objects=objects,
            frame='environment_origin', position_unit='m', quaternion_order='wxyz',
            physical_steps=0, scene_query=self.access(), privileged_history_exposed=True,
            next_call=next_call(self), rollout_finished=False)
        directory = self.output/'scene_queries'/f'{self.query_count-1:04d}'
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory/'query.json', dict(arguments=arguments, result=packet))
        self.record_exposure('object_poses', dict(roles=binding, objects=objects))
        return packet

    def execute(self, **kwargs):
        exposed = self.first_exposure_step is not None
        packet = super().execute(**kwargs)
        if self.history:
            self.history[-1]['privileged_history_exposed'] = exposed
            self.history[-1]['privileged_binding'] = deepcopy(self.binding)
            write_json(self.output/'history.json', self.history)
        return packet


PROMPT = '''
# Role-limited privileged recovery (role_query_v2)
Only after two consecutive executed corrections may the host append an extra
numbered head image and ID mapping. This is identity information, not geometry.
Use normal RGB to choose your current subgoal BEFORE querying. robodojo_query
accepts one grasp_object_id and at most one placement_object_id (null for a free
space target); only pose is returned. No all-object queries or collision-object
enumeration. IDs must come from the label image. Roles bind until a student
execution; do not execute student just to reset the query restriction.
Budgets: 2 pose queries per recovery, 8 per episode, 6 identity images per episode.
Refresh only if necessary. Q labels are snapshot-local, object IDs persist.
All positions are environment-origin meters, quaternion wxyz. Object origin is
NOT a grasp pose or link6. No success, grasp stability or collision oracle exists.
Verify grasp and progress with normal RGB/state, not truth-based success labels.
Old truth remains in history after access closes; never treat it as a new measurement.
All three normal image attachments are already provided. Do not routinely reopen
the identical preview files. Read an original/crop only for a specific unresolved
detail that could change the decision. Continue using the fresh student proposal.
'''
