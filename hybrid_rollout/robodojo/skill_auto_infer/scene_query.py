"""Gated privileged access, isolated from the baseline controller."""
from copy import deepcopy
import hashlib
import numpy as np
from PIL import Image, ImageDraw

from ..io import InputError, write_json
from ..manipulation.simulator_query import INFORMATION_ACCESS, FIELDS
from ..robodojo_server.client import RoboDojoTools
from ..robodojo_server.gate_assessment import GATE_INSTRUCTION
from ..skill.schema import _obj
from .output_contract import decision_id, next_call

THRESHOLD = 2
MAX_QUERIES_PER_DECISION = 2


def tool_spec():
    return dict(type='function', name='robodojo_query',
        description='Optional current scene truth ONLY when scene_query.available. No motion or inference. Use current decision_id; object IDs, not display Q labels. null selects all registered objects.',
        inputSchema=_obj(dict(decision_id={'type': 'string'},
            identifiers={'type': ['array', 'null'], 'items': {'type': 'string'}},
            fields={'type': 'array', 'items': {'type': 'string', 'enum': list(FIELDS)}, 'uniqueItems': True})))


class SceneQueryTools(RoboDojoTools):
    """Same physical execution path; additional read-only controller service."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_count = 0
        self.first_query_step = None
        self.queried_decisions = {}

    def streak(self):
        count = 0
        for row in reversed(self.history):
            if row.get('executed_steps', 0) <= 0:
                continue
            if row['response']['mode'] == 'student':
                break
            if row['response']['mode'] in ('edit', 'eef'):
                count += 1
        return count

    def access(self):
        active = self.phase == 'execute'
        key = decision_id(self.request['request_id']) if active else None
        return dict(version=INFORMATION_ACCESS, available=bool(active and self.streak() >= THRESHOLD
                    and self.queried_decisions.get(key, 0) < MAX_QUERIES_PER_DECISION),
                    consecutive_takeovers=self.streak(), threshold=THRESHOLD,
                    queries=self.query_count, ever_queried=self.query_count > 0,
                    first_query_step=self.first_query_step)

    def audit_access(self):
        value = dict(self.access(), episode_id=self.episode, step_id=self.tick,
                     usage_stage='prepared_for_teacher; consult rpc_in for send evidence')
        write_json(self.output/'scene_query_usage.json', value)
        return value

    def start(self, **kwargs):
        # Base start owns connection creation; validate capability before it resets.
        factory = self.rpc_factory
        def checked_factory(*args, **options):
            client = factory(*args, **options)
            try:
                if not client.request('metadata').get('scene_query', {}).get('enabled'):
                    raise RuntimeError('Simulator must enable scene query')
            except BaseException:
                client.close()
                raise
            return client
        self.rpc_factory = checked_factory
        try:
            packet = super().start(**kwargs)
        finally:
            self.rpc_factory = factory
        self.meta['scene_query'] = self.sim.request('metadata')['scene_query']
        self.run['information_access'] = INFORMATION_ACCESS
        self.run['gate_instruction_sha256'] = hashlib.sha256(allow_query(GATE_INSTRUCTION).encode()).hexdigest()
        write_json(self.output/'run.json', self.run)
        self.audit_access()
        return packet

    def infer(self, **kwargs):
        packet = super().infer(**kwargs)
        # Avoid a repeated baseline prohibition contradicting the opt-in prompt.
        self.request['gate_instruction'] = allow_query(self.request['gate_instruction'])
        write_json(self.output/f"request_{self.request['decision']:03d}.json", self.request)
        packet['scene_query'] = self.access()
        return packet

    def execute(self, **kwargs):
        packet = super().execute(**kwargs)
        self.audit_access()
        return packet

    def finish(self, *args, **kwargs):
        result = super().finish(*args, **kwargs)
        result.update(information_access=INFORMATION_ACCESS, scene_query_usage=self.audit_access())
        write_json(self.output/'result.json', result)
        return result

    def query_scene(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) != {'decision_id', 'identifiers', 'fields'}:
            raise InputError('Query requires decision_id, identifiers, fields; paths are host-managed')
        if not self.access()['available']:
            raise InputError('Scene query locked: requires two executed consecutive takeovers; max two queries per decision')
        key = decision_id(self.request['request_id'])
        if arguments['decision_id'] != key:
            raise InputError('Stale decision_id')
        ids, fields = arguments['identifiers'], arguments['fields']
        if ids is not None and (not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids)):
            raise InputError('Use exact object IDs or null')
        if not isinstance(fields, list) or any(x not in FIELDS for x in fields) or len(fields) != len(set(fields)):
            raise InputError('Invalid fields')
        cameras = self.meta['scene_query']['cameras']
        camera = next((x for x in ('cam_head', 'cam_high') if x in cameras), cameras[0])
        scene = self._rpc('scene_query', camera=camera, identifiers=ids, fields=fields)
        if 'query_error' in scene:
            raise InputError(scene['query_error'])
        directory = self.output/'scene_queries'/f'{self.query_count:04d}'
        directory.mkdir(parents=True, exist_ok=False)
        scene = deepcopy(scene)
        visual = scene['visual']
        rgb, instances = visual.pop('rgb'), visual.pop('instance_ids')
        np.savez_compressed(directory/'arrays.npz', rgb=rgb, instance_ids=instances)
        Image.fromarray(rgb).save(directory/'rgb.png')
        marked = Image.fromarray(rgb)
        draw = ImageDraw.Draw(marked)
        mapping = []
        for i, item in enumerate(visual['visible_objects'], 1):
            x0, y0, x1, y1 = item['bbox_xyxy']
            label = f'Q{i}'
            draw.rectangle((x0, y0, x1-1, y1-1), outline='yellow', width=2)
            draw.text((x0, y0), label, fill='yellow', stroke_width=1, stroke_fill='black')
            mapping.append(dict(display_id=label, object_id=item['object_id'], bbox_xyxy=item['bbox_xyxy']))
        marked.save(directory/'numbered.png')
        write_json(directory/'query.json', dict(episode_id=self.episode, step_id=self.tick,
                   decision_id=key, arguments=arguments, scene=scene))
        self.query_count += 1
        self.queried_decisions[key] = self.queried_decisions.get(key, 0) + 1
        if self.first_query_step is None:
            self.first_query_step = self.tick
        self.audit_access()
        objects = [dict(id=o['id'], label=o['label'], physics_type=o['physics_type'],
                       fields={k: (v['value'] if v['available'] else {'unavailable': v['reason']})
                               for k, v in o['fields'].items()}) for o in scene['objects']]
        packet = dict(episode_id=self.episode, step_id=self.tick, decision_id=key,
            information_access=INFORMATION_ACCESS, scene_query=self.access(),
            coordinate_frame=scene['coordinate_frame'], position_unit='m', quaternion_order='wxyz',
            objects=objects, visible_objects=mapping,
            images=[dict(camera=camera, path=str(directory/'numbered.png'))],
            physical_steps=0, next_call=next_call(self), rollout_finished=False)
        write_json(directory/'tool_result.json', packet)
        return packet


def allow_query(text):
    return text.replace('Two additional tools expose', 'Three additional tools expose').replace(
        'Do not use object truth, reward or future simulated object states.',
        'Current object truth is allowed only through gated robodojo_query. No reward or future simulated object states.').replace(
        'object truth/reward', 'unrecorded object truth/reward').replace(
        'snapshots/object truth are not planning inputs.',
        'snapshots are not planning inputs; only gated robodojo_query truth is permitted.')


PROMPT = '''
# Gated rigid-object recovery: gated_scene_query_v1
The host automatically infers after every nonterminal execution. Keep the existing
takeover assessment, action bounds and native termination rules unchanged.
After two consecutively executed edit/eef decisions, scene_query.available permits
an optional robodojo_query BEFORE constructing another correction. Student execution
resets the streak. Queries/rejections do not count; at most two queries per decision.
Do not query for routine student execution or cloth recovery. Use decision_id from
the latest proposal; identifiers=null discovers objects, then use exact object IDs.
The host selects the head camera and manages paths. Query does not step physics,
consume the proposal, run inference or change decision_id. Continue with execute.
Returned numbered RGB maps Q labels to object IDs. Q labels are snapshot-local;
only object IDs identify objects across queries. Occlusion means missing boxes do
not imply missing objects. Use pose/description first; functional_points if needed.
Positions are meters relative to the sub-environment origin, quaternions wxyz.
Object origins and functional points are NOT executable grasp poses; current robot
targets still refer to link6, not a fingertip/TCP. No planner or grasp-success oracle
is supplied. Use visual evidence and measured motion to verify grasp, not closure alone.
Only this recorded query permits truth. No reward, future-state queries, alternate
RPC or filesystem truth exploration. Previously disclosed truth remains in history
even after the gate closes; stale object positions must not be treated as current.
'''
