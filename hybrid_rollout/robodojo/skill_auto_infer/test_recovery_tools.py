from copy import deepcopy
import json
import numpy as np
import pytest

from ..io import InputError
from ..robodojo_server.client import RoboDojoTools
from .motion import MotionMixin, error, pose, VERSION
from .role_query import RoleQueryTools
from .scene_query import SceneQueryTools
from .test_scene_query import fake as old_fake
from .test_cycle import Rollout
from .output_contract import decision_id


def query_fake(tmp_path):
    obj=old_fake(tmp_path)
    obj.__class__=RoleQueryTools
    obj.recovery=0;obj.binding=None;obj.known_ids={'block','pad','other'}
    obj.recovery_queries=0;obj.label_count=1;obj.first_exposure_step=10;obj.exposures=[]
    return obj


def query_args():
    return dict(decision_id=decision_id('request'),grasp_object_id='block',placement_object_id='pad')


def test_role_binding_and_budget(tmp_path):
    obj=query_fake(tmp_path)
    packet=obj.query_scene(query_args())
    assert obj.calls[0][1]['identifiers']==['block','pad']
    assert obj.calls[0][1]['fields']==['pose']
    assert 'images' not in packet and 'visible_objects' not in packet
    with pytest.raises(InputError,match='bound'):
        obj.query_scene(dict(query_args(),placement_object_id='other'))
    obj.query_scene(query_args())
    with pytest.raises(InputError,match='budget'):obj.query_scene(query_args())
    assert len(obj.calls)==2
    assert json.loads((tmp_path/'privileged_exposures.json').read_text())[0]['roles']['grasp_object_id']=='block'


@pytest.mark.parametrize('change',[{'grasp_object_id':None},{'grasp_object_id':['block']},
    {'placement_object_id':'unknown'},{'identifiers':None},{'decision_id':'stale'}])
def test_invalid_role_no_rpc(tmp_path,change):
    obj=query_fake(tmp_path)
    with pytest.raises(InputError):obj.query_scene(dict(query_args(),**change))
    assert not obj.calls


def test_identity_only_on_gate_and_history_persists(tmp_path,monkeypatch):
    obj=query_fake(tmp_path);obj.known_ids=set();obj.label_count=0;obj.first_exposure_step=None
    monkeypatch.setattr(SceneQueryTools,'infer',lambda self,**kw:dict(images=[]))
    packet=obj.infer()
    assert obj.calls[0][1]['fields']==[] and len(packet['images'])==1
    assert 'pose' not in json.dumps(packet['scene_identities'])
    assert packet['privileged_history_exposed']
    obj.history.append(dict(response=dict(mode='student'),executed_steps=1))
    packet=obj.infer()
    assert not packet['scene_query']['available'] and packet['privileged_history_exposed']
    assert not obj.known_ids and obj.binding is None


class Motion(MotionMixin, RoboDojoTools): pass


def motion_fake(tmp_path):
    obj=object.__new__(Motion);r=Rollout();r.phase='execute'
    obj.phase=r.phase;obj.request=deepcopy(r.request);obj.request['decision']=0
    for v in obj.request['current_eef'].values():v['gripper_opening_command']=1.
    obj.request['student_eef_trajectory']=[deepcopy(obj.request['current_eef'])]*50
    obj.output=tmp_path;obj.next_call=r.next_call;obj.tick=0
    obj.history=[dict(response=dict(mode='eef'),executed_steps=1)]
    obj.meta=dict(motion_executor=VERSION)
    obj.measured=deepcopy(obj.request['current_eef'])
    def rpc(op,**kw):
        assert op=='teacher_observation'
        return dict(eef_positions=np.array([obj.measured[a]['position'] for a in ('left','right')]),
            eef_quaternions_wxyz=np.array([obj.measured[a]['quaternion_wxyz'] for a in ('left','right')]))
    obj._rpc=rpc
    return obj


def motion_args(grasp=False):
    def point(position,gripper='keep'):
        return dict(position=position,quaternion_wxyz=[1,0,0,0],gripper=gripper)
    points=[point([.02,0,0])]
    if grasp:
        points=[point([.01,0,0],'open'),
            point([.02,0,0],'close'),point([.02,0,.03])]
    return dict(decision_id=decision_id('fresh'),arm='left',
        waypoints=points,assessment={},reason='Visible correction')














@pytest.mark.parametrize('change',[{'decision_id':'old'},{'arm':'bad'},
    {'waypoints':[dict(position=[1,0,0],quaternion_wxyz=[1,0,0,0])]},
    {'gripper':'bad'}])
def test_motion_rejected_before_plan(tmp_path,change):
    obj=motion_fake(tmp_path)
    with pytest.raises(InputError):obj.prepare_motion('robodojo_move',dict(motion_args(),**change))
    assert obj._motion_plan is None


@pytest.mark.parametrize('count',[0,5,6])
def test_waypoint_count_boundary(tmp_path,count):
    from .motion import tool_specs
    assert all(s['inputSchema']['properties']['waypoints']['maxItems']==5 for s in tool_specs())
    obj=motion_fake(tmp_path);args=motion_args()
    args['waypoints']=[deepcopy(args['waypoints'][0]) for _ in range(count)]
    if count==5:
        obj.prepare_motion('robodojo_move',args)
        assert len(obj._motion_plan['waypoints'])==5
    else:
        with pytest.raises(InputError,match='1..5'):obj.prepare_motion('robodojo_move',args)
        assert obj._motion_plan is None


@pytest.mark.parametrize('change',[
    {'gripper':'bad'}, {'gripper':'close','gripper_timing':None},
    {'hold_steps':3}, {'gripper':'open','gripper_timing':'before_move','hold_steps':True},
    {'gripper':'open','gripper_timing':'after_arrival','hold_steps':11},
])
def test_bad_event_rejected_before_execution(tmp_path,change):
    obj=motion_fake(tmp_path);args=motion_args()
    args['waypoints'][0].update(change)
    with pytest.raises(InputError):obj.prepare_motion('robodojo_move',args)
    assert obj._motion_plan is None










def test_local_ik_environment_frame_and_joint_bound(tmp_path):
    from ..robodojo_server.kinematics import ArmFK
    # One revolute arm with a fixed one-metre link and translated base.
    path=tmp_path/'arm.urdf'
    path.write_text('''<robot name="test"><joint name="q" type="revolute">
      <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/></joint>
      <joint name="tip" type="fixed"><parent link="arm"/><child link="link6"/>
      <origin xyz="1 0 0" rpy="0 0 0"/></joint></robot>''')
    fk=ArmFK(path,['q'],base='base');root=np.eye(4);root[:3,3]=[2,3,1]
    q=np.array([0.]);goal=.1
    target=dict(position=[2+np.cos(goal),3+np.sin(goal),1],
        quaternion_wxyz=[np.cos(goal/2),0,0,np.sin(goal/2)])
    for _ in range(20):
        nxt,diag=fk.bounded_target(q,[[-1,1]],root,target)
        assert np.max(np.abs(nxt-q))<=.050001 and np.isfinite(nxt).all()
        q=nxt
    assert np.linalg.norm((root@fk.matrix(q))[:3,3]-target['position'])<.003




def test_first_takeover_gate(tmp_path):
    obj=motion_fake(tmp_path);obj.history=[]
    with pytest.raises(InputError,match='First takeover'):
        obj.prepare_motion('robodojo_move',motion_args())
    assert obj._motion_plan is None and obj.tick==0
    obj.history.append(dict(response=dict(mode='eef'),executed_steps=0))
    assert not obj.motion_access()['available']
    obj.history.append(dict(response=dict(mode='edit'),executed_steps=2))
    assert obj.motion_access()['available']
    obj.history.append(dict(response=dict(mode='student'),executed_steps=1))
    assert not obj.motion_access()['available']



@pytest.mark.parametrize('terminal', [False, True])
def test_macro_real_client_archives_and_auto_infer(tmp_path, terminal):
    from ..test_contract import Sim, Student, invoke, response, POSE
    from ..timing import Timing
    class MacroSim(Sim):
        def request(self, op, **args):
            if op == 'execute_joint_motion':
                result = super().request('chunk_step', actions=np.zeros((2,14)))
                for row in result['steps']:
                    row.update(truncated=terminal, edited_target={a:deepcopy(POSE) for a in ('left','right')},
                               ik_diagnostics=dict(method=VERSION))
                result['motion_result'] = dict(version=VERSION, executed_steps=2,
                    status='native_terminal' if terminal else 'waypoints_reached_not_grasp_verified',
                    tracking=[], segments=[])
                return result
            result = super().request(op, **args)
            if op == 'metadata':
                result.update(motion_executor=VERSION, max_episode_steps=1000)
            if op == 'teacher_observation' and not terminal:
                result['remaining_steps']=1000-self.tick
            return result
    obj=Motion(tmp_path,'arrange_largest_number',Student(),rpc_factory=MacroSim,max_decisions=0)
    invoke(obj);invoke(obj)
    obj.history.append(dict(response=dict(mode='eef'),executed_steps=1))
    args=motion_args(True);args['decision_id']=decision_id(obj.request['request_id'])
    args['assessment']=response(obj.request,'eef')['assessment']
    packet=obj.execute_motion('robodojo_grasp',args,Timing(tmp_path/'timing.jsonl'),0)
    assert obj.tick==2 and obj._motion_plan is None
    assert packet['motion_result']['executed_steps']==2
    assert np.load(tmp_path/'execution_001.npz')['actions'].shape==(2,14)
    assert json.loads((tmp_path/'history.json').read_text())[-1]['motion_result']['version']==VERSION
    assert obj.phase==('done' if terminal else 'execute')
    assert obj.counters['predictions']==(1 if terminal else 2)


def test_old_simulator_rejected_before_action(tmp_path):
    obj=motion_fake(tmp_path);obj.meta={}
    with pytest.raises(InputError,match='Simulator lacks'):
        obj.prepare_motion('robodojo_move',motion_args())
    assert obj._motion_plan is None


def test_parallel_gripper_schema_and_host_payload(tmp_path):
    from .motion import tool_specs
    for spec in tool_specs():
        item = spec['inputSchema']['properties']['waypoints']['items']
        assert set(item['properties']) == {'position', 'quaternion_wxyz', 'gripper'}
        assert set(item['required']) == set(item['properties'])
    obj = motion_fake(tmp_path)
    obj.prepare_motion('robodojo_grasp', motion_args(True))
    assert [p['gripper'] for p in obj._motion_plan['waypoints']] == ['open', 'close', 'keep']
    assert all(set(p) == {'position', 'quaternion_wxyz', 'gripper'}
               for p in obj._motion_plan['waypoints'])
