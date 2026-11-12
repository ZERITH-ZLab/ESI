"""Optional sparse motion tools; simulator-local robot-only planned execution."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
from scipy.spatial.transform import Rotation

from ..io import InputError, write_json
from ..skill.schema import _obj, response_schema
from .output_contract import expand_arguments, decision_id
from .cycle import dispatch

from ..robodojo_server.joint_motion import VERSION, DEFAULTS

MAX_STEPS = DEFAULTS.max_steps


def allow_motion(text):
    return text.replace('3. Call robodojo_execute with only response.',
        '3. For ordinary student/edit/eef control call robodojo_execute with only response. '
        'For a bounded multi-step correction you may instead call robodojo_move or '
        'robodojo_grasp under the same assessment gate (see motion contract below).')


def rotation(p):
    return Rotation.from_quat(np.asarray(p['quaternion_wxyz'])[[1,2,3,0]])


def error(a, b):
    return (float(np.linalg.norm(np.asarray(a['position'])-b['position'])),
            float((rotation(a)*rotation(b).inv()).magnitude()))


def pose(value):
    if not isinstance(value, dict) or set(value) != {'position','quaternion_wxyz'}:
        raise InputError('Waypoint requires position and quaternion_wxyz only')
    try:
        p, q = np.asarray(value['position'],float), np.asarray(value['quaternion_wxyz'],float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p,q]).all() or abs(np.linalg.norm(q)-1)>1e-4:
            raise ValueError()
    except (ValueError, TypeError):
        raise InputError('Finite position and unit wxyz required') from None
    return dict(position=p.tolist(), quaternion_wxyz=q.tolist())


def tool_specs():
    vector=lambda n:dict(type='array',items={'type':'number'},minItems=n,maxItems=n)
    waypoint=_obj(dict(position=vector(3),quaternion_wxyz=vector(4),
        gripper={'type':'string','enum':['keep','open','close']}))
    shared=dict(decision_id={'type':'string'},arm={'type':'string','enum':['left','right']},
        waypoints=dict(type='array',items=waypoint,minItems=1,maxItems=5),
        assessment=response_schema()['properties']['assessment'],reason={'type':'string'})
    return [dict(type='function',name='robodojo_'+name,
        description=desc,inputSchema=_obj(shared)) for name,desc in (
        ('move','Bounded link6 sparse-waypoint motion. Same executor in collection/evaluation; self-collision checked; no scene planner. Holds other arm. Then auto infer.',
         ),
        ('grasp','Explicit sparse grasp attempt. Close waits for pose arrival and measured-finger hold. Same executor as move; NO implicit opening, closing or lift. Then auto infer.')) ]


class MotionMixin:
    _motion_plan = None

    def motion_access(self):
        # A completed nonempty short correction is required since the last student
        # action. Failed/rejected calls and empty executions never unlock tools.
        available=False
        for row in reversed(getattr(self,'history',[])):
            if row.get('executed_steps',0)<=0:continue
            response=row.get('response',{})
            if response.get('mode')=='student':break
            if response.get('mode') in ('eef','edit') and not response.get('motion') and not row.get('motion_result'):
                available=True;break
        return dict(available=available, required_first_action='short_edit_or_eef',
            note='Eligibility is not a recommendation; use long motion only in clear space.')

    def infer(self, **kwargs):
        packet=super().infer(**kwargs)
        packet['motion_tools']=self.motion_access()
        return packet

    def validate(self, response, request):
        if self._motion_plan is None:
            return super().validate(response,request)
        # Preserve the existing assessment/gate/language validation. Extended
        # movement bounds are checked separately BEFORE any physical action.
        surrogate=deepcopy(response);surrogate['steps']=1
        surrogate['target']=self._motion_plan['initial']
        return super().validate(surrogate,request)

    def prepare_motion(self, name, arguments):
        if name not in ('robodojo_move','robodojo_grasp'):raise InputError('Unknown motion tool')
        keys={'decision_id','arm','waypoints','assessment','reason'}
        if not isinstance(arguments,dict) or set(arguments)!=keys:
            raise InputError('Invalid motion fields')
        if self.phase!='execute' or arguments['decision_id']!=decision_id(self.request['request_id']):
            raise InputError('Motion requires the latest decision_id and fresh proposal')
        if not self.motion_access()['available']:
            raise InputError('First takeover requires a short edit/eef correction and its feedback; long motion is locked. Do not split actions solely to unlock it.')
        if getattr(self, 'meta', {}).get('motion_executor') != VERSION:
            raise InputError('Simulator lacks '+VERSION+'; restart with --enable-motion-tools and install requirements-motion.txt')
        arm=arguments['arm']
        if arm not in ('left','right'):raise InputError('Unknown arm')
        points=arguments['waypoints']
        if not isinstance(points,list) or not 1<=len(points)<=5:raise InputError('Provide 1..5 waypoints')
        validated=[]
        for point in points:
            if not isinstance(point,dict) or set(point)!={'position','quaternion_wxyz','gripper'}:
                raise InputError('Waypoint requires position, quaternion_wxyz and gripper only')
            value=pose({k:point[k] for k in ('position','quaternion_wxyz')})
            grip=point['gripper']
            if grip not in ('keep','open','close'):raise InputError('Invalid gripper target')
            validated.append(dict(value,gripper=grip))
        points=validated
        initial={a:dict(pose({k:self.request['current_eef'][a][k] for k in ('position','quaternion_wxyz')}),
            gripper_opening=float(self.request['current_eef'][a]['gripper_opening_command']),
            gripper_closed=bool(self.request['current_eef'][a]['gripper_opening_command']<.5)) for a in ('left','right')}
        previous=initial[arm];distance=angle=0.
        for point in points:
            d,a=error(previous,point);distance+=d;angle+=a;previous=point
        if distance>.20 or angle>1.:raise InputError('Motion path exceeds 0.20 m or 1.0 rad; use a shorter segment')
        # Validate ordinary decision BEFORE installing extended execution state.
        compact=dict(decision_id=arguments['decision_id'],mode='eef',steps=1,
            assessment=arguments['assessment'],reason=arguments['reason'],target=initial)
        expanded=expand_arguments(self,'robodojo_execute',dict(response=compact))
        try:super().validate(expanded['response'],self.request)
        except (ValueError,KeyError,TypeError,AttributeError) as e:raise InputError(str(e)) from e
        self._motion_plan=dict(tool=name,arm=arm,waypoints=points,initial=initial,
            status='running',executed_steps=0,tracking=[],version=VERSION,
            completed_waypoints=0,active_waypoint=None,gripper_commands=[],
            parameters=asdict(DEFAULTS),
            privileged_planning=False)
        (self.output/'motion').mkdir(exist_ok=True)
        expanded['response']['steps']=MAX_STEPS  # explicit maximum, not promised duration
        expanded['response']['motion']=deepcopy(self._motion_plan)
        return expanded

    def execute_correction(self, response):
        if self._motion_plan is None:
            return super().execute_correction(response)
        plan = self._motion_plan
        result = self._rpc('execute_joint_motion', arm=plan['arm'],
                           waypoints=plan['waypoints'], decision=self.request['decision'],
                           teacher_request_id=self.request['request_id'])
        self.tick = result['step_id']
        plan.update(result['motion_result'])
        write_json(self.output/'motion'/f"{self.request['decision']:03d}.json", plan)
        return result['steps']

    def execute_motion(self,name,arguments,timing,call_id):
        expanded=self.prepare_motion(name,arguments)
        decision=self.request['decision']
        try:
            packet=dispatch(self,'robodojo_execute',expanded,timing,call_id)
            if self._motion_plan['status']=='running':
                self._motion_plan['status']='native_terminal_or_decision_limit'
            self._motion_plan['executed_steps']=self.history[-1]['executed_steps']
            write_json(self.output/'motion'/f'{decision:03d}.json',self._motion_plan)
            packet['motion_result']={k:v for k,v in self._motion_plan.items() if k not in ('tracking','initial')}
            self.history[-1]['motion_result']=deepcopy(packet['motion_result'])
            write_json(self.output/'history.json',self.history)
            return packet
        except InputError as e:
            # Never advertise no_execution after a partially executed macro.
            raise RuntimeError('Motion failed; do not replay') from e
        finally:
            plan=self._motion_plan
            if plan is not None and plan['status']=='running':
                plan['status']='terminal_or_interrupted'
                write_json(self.output/'motion'/f'{decision:03d}.json',plan)
            self._motion_plan=None


PROMPT='''
# Continuous link6 motion tools (link6_relaxed_quintic_ack_soft_boundary_v1)
GPT short EEF/edit and long motion use complete-target quintic planning. Ordinary
student actions and grippers pass through literally with their selected tick count;
there is no QP prefix modification, transition bridge or hidden inference.
Terminal reference velocity/acceleration independently default to zero, so bounded
chunk-end deceleration remains. Strict command or physical C2 continuity is not claimed.
robodojo_move and robodojo_grasp are OPTIONAL corrections under the same
failure-or-intent gate; student remains preferred when its proposal is aligned.
Both work identically with query enabled or disabled. Robot self-collision
spheres are checked; no object truth, scene collision planning, teleportation
or automatic grasp-success detection is used by them.
The FIRST takeover at episode start or after any nonempty student execution MUST
use ordinary short edit/eef. Long tools unlock only after an executed short correction
and its returned feedback. Check motion_tools.available. Do not split actions or
perform token corrections merely to unlock tools. Eligibility does not imply safety.
Supply latest decision_id, assessment/reason, left/right arm and 1..5 sparse poses.
All poses target link6 in environment-origin meters, unit wxyz, NOT fingertips/TCP.
Measured joints anchor the initial pose. Genuine ACK command history supplies
entrance reference velocity/acceleration. The planner first tries those exact
vectors, then explicitly recorded scales 1, .25, .05, .01, 0 within the same hard
position, path, collision and tick constraints. Recovery never edits ACK history
or claims that its effective boundary is the real ACK derivative.
The continuous via-path uses complete iterative robot-only IK and shared quintic
progress. Geometry retains the 2mm/.02rad corridor, continuous joint-position
bounds, dense dynamic checks and native joint-linear chord checks. Ordinary
vias do not force stops; explicit close events add an arrival-and-hold boundary.
The other arm holds its measured initial joint target.
Joint reference speed is capped by min(URDF*6, configured 5, native 5)rad/s,
acceleration 200rad/s^2, jerk 20000rad/s^3; EE caps are 2m/s and 12rad/s.
Consecutive targets are capped at .20rad/joint, .08m and.48rad. These are finite
simulation planning settings; the native PD/interpolation is unchanged.
Tracking limits are .08m/.6rad/.6radjoint; held-arm limits stay .03m/.2rad.
Receipts independently record old tracking thresholds and genuine cross-ACK
command velocity/acceleration/jerk, so soft boundaries do not hide a jump.
Motion alone adds no initial/final stop-and-wait phase; explicit closure adds the
hold described below. The endpoint pose is checked after the final issued action;
nonzero measured speed is allowed and recorded. Planning
knots cost no physical actions. Timing is conservative, not time-optimal.
There is NO online progress normalization, pullback, collision avoidance or grasp
oracle. GPT short and long motion each fit at most min(60, remaining episode
steps). Short steps are a requested minimum: complete original targets may use
more ticks within that budget, never a clipped substitute. For short edit, keep
preserves the original continuous pi gripper index/time sequence, linearly
resampled only when arm planning expands the tick count; it does not freeze the
opening. Explicit short EEF closure or edit gripper_intent=closed instead uses
arrival-and-hold events, as do long close waypoints. Opening is not deferred.
Total path <=20cm and rotation <=1rad. Endpoint tolerance 5mm /
0.05rad. Tracking failure or budget exhaustion stops and returns observations;
Waypoint arrival requires measured evidence; command receipts do not verify a grasp.
Unreachable/invalid paths return planning_failed. Paths failing the robot
self-collision check return self_collision before movement along that segment.
Choose a different waypoint/pose after inspecting feedback; do not blindly repeat
the rejected target. The guard covers configured within-arm spheres, not
inter-arm, object, table or carried-object collision geometry.
These are not collision guarantees: account for the carried object's full extent.
Both tools use the SAME executor. Each waypoint has exactly position,
quaternion_wxyz and gripper=keep/open/close. Arm joints and gripper opening are
sent in the SAME 25Hz action, including holds. Long keep retains the preceding
issued opening target (initial observation only before any action). Long open=1
applies during approach. For an explicit close event, movement retains the
preceding issued opening target until the specified pose is measurably reached;
then close=0 is issued while both arm joint targets are held. Retaining a target
does not freeze fingers that are still moving toward it.
The first explicit close per arm in a request triggers this hold even if the
preceding command was already closed; repeated close commands within that request
do not retrigger it unless the opening target increased in between.
Default hold: at least 3 and at most 8 physical actions; both measured fingers of
each closing arm must change by less than 0.1mm on 2 consecutive post-close
transitions. Runtime configuration may override these settings; inspect receipts.
This detects stopped fingers, NOT object contact or a successful grasp. A stationary
close also uses this hold. Holds count toward the same 60-step/episode budget;
timeout, failed arrival or tracking/collision failure stops the remaining request.
Earlier actions may already have executed: inspect actual executed steps and feedback.
There is no caller-supplied timing or hold_steps field. There is NO implicit open,
close or lift, even in grasp.
Include any test lift as an explicit waypoint; its distance counts in the budget.
Example transport: clear approach/open -> further approach/keep. A contact/close
point waits for measured pose arrival before closing and holding; reaching that
pose does not prove object contact. A support/open point releases WHILE approaching
support. To release only after support, first reach support with keep, inspect
feedback, then explicitly open at the supported pose in a subsequent action.
Do not reopen an already-held object for transport. Merge only already-decided
motions; stop the segment wherever fresh visual evidence is needed. If contact
location is uncertain, approach OPEN to a useful viewing/pregrasp pose and inspect
the returned observation before closing. Clear current evidence may justify a
combined approach-and-close request only when the contact pose is already clear.
Query object origins are not grasp points
and do not remove contact uncertainty. After a miss, use visible error to change
contact position or orientation; do not blindly repeat close/lift attempts.
Use ordinary short eef control for uncertain contact adjustments. Tools are optional,
not a requirement to perform long open-loop actions; prefer aligned student proposals.
Use long segments only for visibly clear approaches, retreats or stable transport.
Stop conservatively BEFORE uncertain contact, then use ordinary eef (5cm / 0.35rad,
1..5 steps) with fresh visual feedback for grasping, release and placement alignment.
Query coordinates do not establish a collision-free path. Do not bundle an uncertain
approach, closure and long transport. Prefer keeping the gripper unchanged during
long movement; explicit gripper commands remain available but are not a reason
to skip feedback.
Tool completion proves neither grasp nor stability: check returned RGB/state for
object following, slip and orientation. Query coordinates do not provide a grasp pose.
Apply the rigid-object pick-and-place preferences in the robot contract; these
also apply to ordinary EEF corrections, not only to motion tools.
Nonterminal completion returns observations AND a fresh pi05 proposal automatically;
review it and hand back to student as soon as appropriate. Never manually call infer.
The original short EEF 5cm/.35rad and edit-offset limits remain; short minimum
steps can expand within the 60-tick/episode budget. Current short planning also
retains a cumulative.205m/1.01rad gate, which can refuse some otherwise legal
edit prefixes. Long tools are separate bounded multi-step segments. There is no mid-segment GPT/student switching or automatic retry.
'''
