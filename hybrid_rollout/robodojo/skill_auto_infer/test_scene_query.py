import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from ..io import InputError
from ..robodojo_server.client import RoboDojoTools
from ..manipulation.simulator_query import simulator_query
from ..manipulation.visual_scene_query import _map_segments
from ..dual_campaign import configure
from ..test_dual_deployment import configuration
from .output_contract import decision_id
from .scene_query import SceneQueryTools, allow_query, PROMPT, tool_spec
from .run import content_items, SKILL_ROOT, build_prompt, tool_specs


def row(mode, count=1):
    return dict(response=dict(mode=mode), executed_steps=count)


def fake(tmp_path):
    obj = object.__new__(SceneQueryTools)
    obj.output = tmp_path
    obj.task = 'arrange_largest_number'
    obj.history = [row('edit'), row('eef')]
    obj.phase, obj.episode, obj.tick = 'execute', 'episode', 10
    obj.query_count, obj.first_query_step, obj.queried_decisions = 0, None, {}
    obj.request = dict(request_id='request', decision=2)
    obj.meta = dict(scene_query=dict(cameras=['cam_head']))
    obj.next_call = lambda: dict(tool='robodojo_execute')
    obj.calls = []
    def rpc(op, **args):
        obj.calls.append((op, args))
        return dict(coordinate_frame='sub_environment', objects=[dict(id='block', label='block',
            physics_type='rigid', fields={'pose':dict(available=True, value={'position':[1,2,3]})})],
            visual=dict(rgb=np.zeros((16,16,3),np.uint8), instance_ids=np.zeros((16,16),np.uint32),
                visible_objects=[dict(object_id='block',bbox_xyxy=[0,0,12,12])]))
    obj._rpc = rpc
    return obj


def args():
    return dict(decision_id=decision_id('request'), identifiers=None, fields=['pose'])


def test_gate_counts_only_acknowledged_takeovers(tmp_path):
    obj = fake(tmp_path)
    assert obj.access()['available']
    obj.history += [row('student', 0)]
    assert obj.access()['available']
    obj.history += [row('student')]
    assert not obj.access()['available']
    obj.history += [row('eef')]
    assert not obj.access()['available']
    obj.history += [row('edit')]
    assert obj.access()['available']
    obj.phase = 'done'
    assert not obj.access()['available']


@pytest.mark.parametrize('change', [dict(decision_id='old'), dict(fields=['mesh']), dict(identifiers=[2]), dict(output_dir='bad')])
def test_reject_before_rpc(tmp_path, change):
    obj = fake(tmp_path)
    with pytest.raises(InputError):
        obj.query_scene(dict(args(), **change))
    assert not obj.calls


def test_query_preserves_proposal_and_archives(tmp_path):
    obj = fake(tmp_path)
    before = deepcopy(obj.request)
    packet = obj.query_scene(args())
    assert obj.request == before and obj.phase == 'execute' and obj.tick == 10
    assert packet['physical_steps'] == 0 and len(packet['images']) == 1
    assert packet['visible_objects'][0]['object_id'] == 'block'
    assert (tmp_path/'scene_queries/0000/arrays.npz').exists()
    assert json.loads((tmp_path/'scene_query_usage.json').read_text())['first_query_step'] == 10
    items = content_items(packet, images=True)
    assert len(items) == 2 and items[1]['type'] == 'inputImage'
    assert 'instance_ids' not in items[0]['text']
    obj.query_scene(args())
    with pytest.raises(InputError): obj.query_scene(args())
    assert len(obj.calls) == 2
    obj.history += [row('student')]
    assert obj.access()['ever_queried'] and not obj.access()['available']


def test_infer_packet_has_no_gate_but_archive_is_consistent(tmp_path, monkeypatch):
    obj = fake(tmp_path)
    obj.request['gate_instruction'] = 'Do not use object truth, reward or future simulated object states.'
    monkeypatch.setattr(RoboDojoTools, 'infer', lambda self, **kw: {'request_id': 'request'})
    packet = obj.infer()
    assert packet['scene_query']['available']
    assert 'gated robodojo_query' in json.loads((tmp_path/'request_002.json').read_text())['gate_instruction']


def test_disabled_simulator_does_not_import_or_render():
    with pytest.raises(RuntimeError, match='disabled'):
        simulator_query(SimpleNamespace(scene_query_enabled=False), {})


@pytest.mark.parametrize('enabled', [False, True])
def test_start_checks_capability_before_reset(tmp_path, monkeypatch, enabled):
    obj = fake(tmp_path)
    obj.phase, obj.sim = 'start', None
    events = []
    class Client:
        def request(self, op):
            events.append(op)
            return dict(scene_query=dict(enabled=enabled, cameras=['cam_head']), instruction='short')
        def close(self): events.append('close')
    factory = lambda *a, **kw: Client()
    obj.rpc_factory = factory
    def start(self, **kw):
        self.sim = self.rpc_factory('localhost', 0)
        events.append('reset')
        self.meta, self.run = dict(instruction='full task instruction'), {}
        self.phase = 'infer'
        return {'started': True}
    monkeypatch.setattr(RoboDojoTools, 'start', start)
    if not enabled:
        with pytest.raises(RuntimeError, match='enable scene query'): obj.start()
        assert events == ['metadata', 'close']
    else:
        assert obj.start()['started']
        assert events == ['metadata', 'reset', 'metadata']
        assert obj.meta['instruction'] == 'full task instruction'
        assert obj.run['information_access'] == 'gated_scene_query_v1'
    assert obj.rpc_factory is factory


def test_segment_mapping_uses_registered_ancestor():
    records = [('rigid',dict(prim_path='/World/block',inst_name='block'))]
    packet = _map_segments(np.array([[0,1],[2,1]], dtype=np.uint32),
        {0:'BACKGROUND',1:'/World/block/mesh',2:'/World/block2/mesh'}, records)
    assert packet['segment_to_object'] == {1:'block'}
    assert packet['visible_objects'][0]['object_id'] == 'block'


def test_baseline_specs_and_prompt_unchanged(tmp_path):
    original = build_prompt(SKILL_ROOT, tmp_path)
    assert 'Do not use object truth' in original
    assert 'Do not use object truth' not in allow_query(original) + PROMPT
    assert len(tool_specs()) == 2
    assert tool_spec()['name'] == 'robodojo_query'


def test_only_new_route_can_enable(tmp_path):
    env, _ = configuration(tmp_path)
    env['ROLLOUT_ENABLE_SCENE_QUERY'] = '1'
    with pytest.raises(ValueError, match='requires'): configure(env)
    env.update(ROLLOUT_CONTROLLER_ROUTE='auto_infer_v1', ROLLOUT_EVALUATION_METHOD='pi05_plus_gpt')
    assert configure(env)[0]['ROLLOUT_ENABLE_SCENE_QUERY'] == '1'


def test_worker_query_does_not_auto_infer_or_execute(tmp_path):
    from . import run
    from .test_cycle import Rollout
    from ..test_network_continue import Transport, event, completed
    def call(name, index, arguments):
        value = event('item/tool/call', callId=str(index), tool=name, arguments=arguments)
        value['id'] = index
        return value
    worker = object.__new__(run.CodexPolicy)
    worker.workspace = worker.agent_workspace = tmp_path
    worker.thread_id = 'thread'
    worker.method, worker.timeout, worker.image_max_edge = 'pi05_plus_gpt', 1, 480
    worker.scene_query_enabled = True
    worker.transport = Transport([
        call('robodojo_start', 1, dict(task='fold')),
        call('robodojo_query', 2, dict(decision_id=decision_id('fresh'), identifiers=None, fields=['pose'])),
        call('robodojo_execute', 3, dict(response=dict(decision_id=decision_id('fresh'),
            mode='student', steps=3, assessment={}, reason='aligned'))),
        completed('t0', 'completed', None)])
    rollout = Rollout(terminal=True)
    def query(arguments):
        rollout.calls.append('query')
        assert rollout.phase == 'execute'
        return dict(physical_steps=0, images=[], next_call=dict(tool='robodojo_execute',
                    decision_id=decision_id('fresh')), rollout_finished=False)
    rollout.query_scene = query
    worker.run(rollout)
    assert rollout.calls == ['start', 'infer', 'query', 'execute']
    assert len(worker.transport.replies) == 3
