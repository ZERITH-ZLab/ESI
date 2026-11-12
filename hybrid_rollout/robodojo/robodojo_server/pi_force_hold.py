"""RoboDojo/X5 contact-triggered holding/tightening with immutable model commands.

Release presets live in configs/environments/robodojo.json. These depend on X5
finger mapping (zero means closed), PhysX contacts and native interpolation;
other environments must opt into their own calibrated execution adaptation.
"""
from dataclasses import asdict, dataclass
import json
import os

import numpy as np

from .action_edit_kinematics import transform
from .self_collision import SelfCollisionError

ARMS = ('left', 'right')
SCHEMA = 'pi_force_hold_event_v1'
VERSION = 'contact_margin_v3'
ZERO_VERSION = 'contact_zero_v4'
BOOTSTRAP_CHECKPOINT_SHA256 = 'd15fb8bd1d29cb30b69f01b71c66596cb0293c1a8a94111b343c1580dd3e3e5b'
# Native X5 per-finger affine target mapping has a 54 mm span (-10 to 44 mm).
SINGLE_FINGER_SCALE_M = .054


@dataclass(frozen=True)
class ForceHoldConfig:
    """Uncalibrated experiment values; no fixed open/close or departure gates."""
    action_trend_delta: float = .01
    opening_trend_delta: float = .01
    internal_trend_delta: float = .001
    physical_closing_delta_m: float = .0001
    action_history_size: int = 5
    force_mean_N: float = .1
    opposition_cos_max: float = -.2
    contact_windows: int = 2
    stable_intervals: int = 2
    stable_delta_m: float = .0001
    max_extra_steps: int = 8
    tighten_enabled: bool = False
    tighten_mode: str = 'margin'
    flywheel_round: int | None = None
    controller_route: str | None = None
    checkpoint_sha256: str | None = None
    margin_threshold_m: float = .010
    minimum_margin_m: float = .015
    margin_multiplier: float = 1.5
    ramp_steps: int = 3
    model_hold_joint_delta_rad: float = .003
    allowed_object_paths: tuple = ()

    def __post_init__(self):
        positive = (self.action_trend_delta, self.opening_trend_delta,
            self.internal_trend_delta, self.physical_closing_delta_m,
            self.force_mean_N, self.stable_delta_m, self.margin_threshold_m,
            self.minimum_margin_m, self.margin_multiplier, self.model_hold_joint_delta_rad)
        if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError('Positive finite contact-tighten settings required')
        if (isinstance(self.opposition_cos_max, bool) or not np.isfinite(self.opposition_cos_max)
                or not -1 <= self.opposition_cos_max < 0):
            raise ValueError('Opposing contact vectors require -1 <= opposition_cos_max < 0')
        if any(type(v) is not int or v < 1 for v in
                (self.action_history_size, self.contact_windows, self.stable_intervals, self.max_extra_steps)):
            raise ValueError('Positive integer history/window/ACK settings required')
        if self.action_history_size < 2 or self.max_extra_steps > 8:
            raise ValueError('History needs 2 samples; maximum extra ACKs is 8')
        if type(self.tighten_enabled) is not bool:
            raise ValueError('tighten_enabled must be boolean')
        if self.tighten_mode not in ('margin', 'zero'):
            raise ValueError('tighten_mode must be margin or zero')
        # A selected tightening mode is inactive when tightening is disabled;
        # this supports measured holding alone without changing the saved mode.
        if self.ramp_steps != 3 or type(self.ramp_steps) is not int:
            raise ValueError('Margin tightening uses exactly three ramp steps')
        if self.minimum_margin_m < self.margin_threshold_m or self.margin_multiplier < 1:
            raise ValueError('Margin policy must not loosen the original command')
        if self.tighten_enabled and self.tighten_mode == 'zero':
            if (type(self.flywheel_round) is not int or self.flywheel_round < 1
                    or self.controller_route not in ('auto_infer_v1', 'subtask_feedforward_v1')
                    or not isinstance(self.checkpoint_sha256, str)
                    or len(self.checkpoint_sha256) != 64
                    or any(c not in '0123456789abcdef' for c in self.checkpoint_sha256)):
                raise ValueError('Zero tightening requires a positive flywheel_round, supported route and pinned checkpoint SHA256')
        if self.tighten_enabled and self.tighten_mode == 'margin' and (type(self.flywheel_round) is not int or self.flywheel_round != 1
                or self.controller_route != 'auto_infer_v1'
                or self.checkpoint_sha256 != BOOTSTRAP_CHECKPOINT_SHA256):
            raise ValueError('Tightening requires flywheel_round=1, auto_infer_v1 and the approved official checkpoint')
        if not isinstance(self.allowed_object_paths, (list, tuple)) or any(
                not isinstance(p, str) or not p.startswith('/') for p in self.allowed_object_paths):
            raise ValueError('allowed_object_paths must be exact absolute native actor paths')
        object.__setattr__(self, 'allowed_object_paths', tuple(self.allowed_object_paths))


def configured(value=None):
    """Reject obsolete near-close/departure options instead of silently ignoring."""
    if value is None:
        value = json.loads(os.environ.get('ROBODOJO_PI_FORCE_HOLD_CONFIG', '{}'))
    return value if isinstance(value, ForceHoldConfig) else ForceHoldConfig(**value)


def execution_version(config):
    config = configured(config)
    return ZERO_VERSION if config.tighten_enabled and config.tighten_mode == 'zero' else VERSION


def validate_tightening_context(config, *, controller_route, checkpoint_sha256=None):
    """Both modes pin runtime identity; only margin is bootstrap-only."""
    config = configured(config)
    if config.tighten_enabled:
        if controller_route != config.controller_route:
            raise ValueError('Tightening controller route differs from the declared route')
        if checkpoint_sha256 is not None and checkpoint_sha256 != config.checkpoint_sha256:
            raise ValueError('Tightening checkpoint differs from the pinned checkpoint')


def validate_tightening_policy(config, policy_version):
    """The simulator checks the reset identity against the pinned checkpoint."""
    config = configured(config)
    if config.tighten_enabled:
        expected = config.checkpoint_sha256
        if not (policy_version == expected or policy_version.endswith('/'+expected[:16])):
            raise ValueError('Tightening reset requires the pinned policy identity')


def margin_plan(config, positions, base_target, *, enabled=True):
    """Freeze a symmetric X5 total-gap plan; all distances are metres."""
    width = float(np.sum(positions))
    target = 2 * (-.010 + SINGLE_FINGER_SCALE_M * float(base_target))
    margin = width - target
    boost = config.tighten_enabled and enabled
    requested_margin = ((config.minimum_margin_m if margin <= config.margin_threshold_m
                         else margin * config.margin_multiplier) if boost else margin)
    if boost and config.tighten_mode == 'zero':
        requested_margin = width + .020  # normalized 0 maps to -10 mm per finger
    reduction = max(0., requested_margin - margin)
    issued = float(np.clip(float(base_target) - reduction / (2 * SINGLE_FINGER_SCALE_M), 0, 1))
    return dict(reference_finger_positions_m=np.asarray(positions).tolist(),
        reference_width_m=width, base_target_width_m=target, original_margin_m=margin,
        requested_margin_m=requested_margin, requested_final_target_width_m=width-requested_margin,
        requested_extra_gap_m=reduction, final_issued_target=issued,
        clipped_final_margin_m=width-2*(-.010+SINGLE_FINGER_SCALE_M*issued),
        tightening_enabled=boost)


def registered_dynamic_actors(env):
    """Resolve native task rigid actors; fixtures/robots have separate registries."""
    from pxr import Usd, UsdPhysics
    manager = env.scene_manager
    objects = manager._rigid_and_dynamic_objects[0]
    actors = set()
    for obj in objects.values():
        path = getattr(obj, 'usd_prim_path', None) or getattr(obj, 'prim_path', None)
        if not path:
            raise ValueError('Registered task object has no native prim path')
        prim = manager.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f'Registered task object prim missing: {path}')
        for body in Usd.PrimRange(prim):
            if not body.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            api = UsdPhysics.RigidBodyAPI(body)
            if api.GetRigidBodyEnabledAttr().Get() is not False and not api.GetKinematicEnabledAttr().Get():
                actors.add(str(body.GetPath()))
    return actors


def _pose(session, arm):
    i = ARMS.index(arm)
    return transform(session.obs['eef_positions'][i], session.obs['eef_quaternions_wxyz'][i])


def _fingers(session):
    obs = session.obs
    value = np.asarray(obs.get('finger_positions_m', []), float)
    if (obs.get('finger_positions_available') is not True
            or obs.get('finger_positions_step_id') != session.step_id
            or value.shape != (2, 2) or not np.isfinite(value).all()):
        return None
    return value


def measured_joints(session):
    """Read physical arm feedback directly, never commanded state echoes."""
    values = []
    for arm in ARMS:
        q = np.asarray(session.kinematics.manager.get_joint(session.kinematics.robots[arm])[0], float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Invalid measured pi force hold arm joints')
        values.append(q)
    return values


class PiForceHold:
    """Use executed original-command trends; compensation never becomes intent."""

    def __init__(self, config):
        self.config = config
        self.version = execution_version(config)
        from .gripper_zero import ZeroLatch
        self.zero_latch = ZeroLatch(self) if config.tighten_enabled and config.tighten_mode == 'zero' else None
        self.allowed_actors = set()
        self.arms = {arm: self._empty() for arm in ARMS}
        self.last_action = None
        self.last_model_action = None
        self.last_model_reference = None
        self.finger_closing_directions = {}
        self.event_count = 0
        self.transitions = []
        self.pending_transitions = []
        self.trigger_evidence = {}
        self.plans = {}
        self.tightened_actors = {}
        self.native_holds = {}
        self.pending_native = {}

    @staticmethod
    def _empty():
        return dict(closing=False, used=False, action_high=None, action_low=None,
                    model_history=[], model_step_ids=[], model_finger_history=[],
                    model_internal_history=[], action_window_net_change=None, fingers=None,
                    closing_finger_baseline_m=None, closing_internal_baseline=None,
                    closing_evidence_start_step_id=None, closing_evidence_detected_step_id=None,
                    physical_baseline_step_id=None,
                    physical_net_closing_m=None, physical_closing_seen=False,
                    internal_net_closing=None, internal_closing_seen=False,
                    physical_finger_delta_m=None, internal_target=None,
                    internal_target_delta=None, contact=None)

    def initialize(self, session):
        """Resolve actual eligible bodies and require measured joint/finger inputs."""
        self.__init__(self.config)  # Every native reset starts a new contact lifecycle.
        registered = registered_dynamic_actors(session.env)
        requested = set(self.config.allowed_object_paths)
        if requested - registered:
            raise ValueError(f'Unregistered/non-dynamic allowlist actors: {sorted(requested - registered)}')
        self.allowed_actors = requested or registered
        if not self.allowed_actors:
            raise ValueError('Contact tightening requires registered dynamic rigid task actors')
        if session.gripper_contact is None or _fingers(session) is None:
            raise ValueError('Contact tightening requires CPU PhysX contacts and real finger feedback')
        measured_joints(session)
        for arm in ARMS:
            robot = session.kinematics.robots[arm]
            span = float(robot.gripper_scale[1] - robot.gripper_scale[0])
            sign = float(robot.gripper_move['sign'])
            mimic = float(robot.gripper_move['mimic'][1])
            if (not np.isclose(span, SINGLE_FINGER_SCALE_M)
                    or not np.isclose(robot.gripper_scale[0], -.010) or sign != 1 or mimic != 1):
                raise ValueError('Contact tightening requires the verified symmetric X5 54mm mapping span')
            self.finger_closing_directions[arm] = -np.sign([sign, sign * mimic])
        self.last_action = np.asarray(session.obs['states'], np.float32).copy()
        session.metadata['pi_force_hold'].update(version=self.version,
            allowed_object_paths_resolved=sorted(self.allowed_actors),
            object_resolution='native_scene_registered_dynamic_rigid_actors',
            single_finger_scale_m=SINGLE_FINGER_SCALE_M,
            single_finger_scale_semantics='affine_target_mapping_span_not_absolute_opening',
            native_gripper_scales={arm: list(session.kinematics.robots[arm].gripper_scale) for arm in ARMS},
            finger_closing_directions={arm: v.tolist() for arm, v in self.finger_closing_directions.items()})

    def control_switch(self, session, source):
        """Reset model-intent history at genuine source handoffs."""
        self.arms = {arm: self._empty() for arm in ARMS}
        self.plans.clear()
        self.native_holds.clear()
        self.pending_native.clear()
        self.last_model_action = None
        self.last_model_reference = None
        transition = dict(schema='pi_force_hold_state_v1', version=self.version,
            episode_id=session.episode_id, step_id=session.step_id,
            reason='control_source_model_history_reset', source=source,
            last_ack_action=self.last_action.tolist())
        session._write_force_hold_transition(transition)
        self.transitions.append(transition)

    def opening(self, arm, target):
        """Detect a cumulative opening reversal with hysteresis, never a scalar gate."""
        low = self.arms[arm]['action_low']
        return low is not None and float(target) - low + 1e-7 >= self.config.opening_trend_delta

    def contact(self, session, arm):
        """Return exact-window bilateral opposing normal contact or an explicit reason."""
        receipt = session.gripper_contact_cache or {}
        result = dict(valid=False, reason='invalid_or_stale_force_window', actor=None,
            step_id=receipt.get('step_id'), episode_id=receipt.get('episode_id'))
        if (receipt.get('episode_id') != session.episode_id or receipt.get('step_id') != session.step_id
                or receipt.get('reports_valid') is not True or receipt.get('archive_valid', True) is not True
                or _fingers(session) is None):
            return result
        fingers = receipt.get('arms', {}).get(arm, {})
        if not all(fingers.get(name, {}).get('valid') is True for name in ('link7', 'link8')):
            result['reason'] = 'invalid_finger_force'
            return result
        common = (set(fingers['link7'].get('objects', {})) &
                  set(fingers['link8'].get('objects', {})) & self.allowed_actors)
        result.update(reason='no_same_eligible_bilateral_actor', observed_objects={
            name: sorted(fingers[name].get('objects', {})) for name in ('link7', 'link8')})
        eligible = []
        for actor in sorted(common):
            means, vectors = [], []
            for name in ('link7', 'link8'):
                item = fingers[name]['objects'][actor]
                means.append(item.get('force_magnitude_mean_N', np.nan))
                vectors.append(np.asarray(item.get('force_world_mean_N', []), float))
            if (not np.isfinite(means).all() or any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors)):
                result['reason'] = 'invalid_object_normal_force'
                continue
            norms = [float(np.linalg.norm(v)) for v in vectors]
            if min(means) < self.config.force_mean_N or min(norms) < self.config.force_mean_N:
                result['reason'] = 'bilateral_force_below_contact_threshold'
                continue
            cosine = float(np.dot(vectors[0], vectors[1]) / (norms[0] * norms[1]))
            if cosine > self.config.opposition_cos_max:
                result['reason'] = 'normal_forces_not_opposing'
                continue
            eligible.append(dict(actor=actor, forces_mean_N=list(map(float, means)),
                force_world_mean_N=[v.tolist() for v in vectors], opposition_cos=cosine))
        if len(eligible) != 1:
            if len(eligible) > 1:
                result['reason'] = 'ambiguous_multiple_bilateral_actors'
            return result
        result.update(eligible[0], valid=True, reason='same_actor_opposing_normal_contact')
        return result

    def consider(self, session, action):
        """Inspect current contact and the next candidate without committing model intent."""
        self.pending_transitions = []
        self.trigger_evidence = {}
        self.pending_native = {}
        opening_arms = [arm for i, arm in enumerate(ARMS) if self.opening(arm, action[i * 7 + 6])]
        if opening_arms:
            # Either arm's real opening candidate takes priority over all holds.
            # The original two-arm action executes; only its ACK commits the trend.
            return []
        arms = []
        for arm in ARMS:
            state = self.arms[arm]
            if (not state['closing'] or state['used'] or self.last_model_action is None
                    or not state['physical_closing_seen'] or not state['internal_closing_seen']):
                continue
            contact = self.contact(session, arm)
            if contact['valid']:
                arms.append(arm)
                self.trigger_evidence[arm] = dict(contact=contact,
                    executed_model_history=list(state['model_history']),
                    executed_model_step_ids=list(state['model_step_ids']),
                    action_high=state['action_high'], action_low=state['action_low'],
                    action_window_net_change=state['action_window_net_change'],
                    internal_target=state['internal_target'],
                    internal_target_delta=state['internal_target_delta'],
                    physical_finger_delta_m=state['physical_finger_delta_m'],
                    physical_net_closing_m=state['physical_net_closing_m'],
                    physical_closing_seen=state['physical_closing_seen'],
                    internal_net_closing=state['internal_net_closing'],
                    internal_closing_seen=state['internal_closing_seen'],
                    closing_finger_baseline_m=state['closing_finger_baseline_m'],
                    closing_internal_baseline=state['closing_internal_baseline'],
                    closing_evidence_start_step_id=state['closing_evidence_start_step_id'],
                    closing_evidence_detected_step_id=state['closing_evidence_detected_step_id'],
                    physical_baseline_step_id=state['physical_baseline_step_id'],
                    finger_closing_directions=self.finger_closing_directions[arm].tolist(),
                    closing_evidence='executed_original_action_history')
        return arms

    def ensure_plans(self, session, arms):
        positions = _fingers(session)
        for arm in arms:
            actor = self.trigger_evidence[arm]['contact']['actor']
            if arm not in self.plans or self.plans[arm]['actor'] != actor:
                i = ARMS.index(arm)
                self.plans[arm] = dict(margin_plan(self.config, positions[i],
                    self.last_model_action[i*7+6], enabled=self.zero_latch is not None or self.tightened_actors.get(arm) != actor),
                    actor=actor, reference_step_id=session.step_id)

    def prefer_model_hold(self, session, action, arms):
        """Execute a stationary model command first; count only its later real ACK."""
        self.ensure_plans(session, arms)
        joints = measured_joints(session)
        for arm in arms:
            if self.native_holds.get(arm, {}).get('step_id') != session.step_id:
                self.native_holds.pop(arm, None)
            i = ARMS.index(arm)
            # A flat gripper command during transport is not a holding posture.
            reference = self.native_holds.get(arm, {}).get('joints', joints[i])
            if (np.max(np.abs(action[i*7:i*7+6] - joints[i])) > self.config.model_hold_joint_delta_rad
                    or np.max(np.abs(action[i*7:i*7+6] - reference)) > self.config.model_hold_joint_delta_rad
                    or action[i*7+6] > min(self.last_model_action[i*7+6],
                                         self.plans[arm]['final_issued_target']) + 1e-7):
                return False
        self.pending_native = {arm: dict(actor=self.plans[arm]['actor'],
            joints=np.asarray(self.native_holds.get(arm, {}).get('joints', joints[ARMS.index(arm)])).copy())
            for arm in arms}
        return True

    def observe_model_hold(self, session, positions):
        for arm, pending in self.pending_native.items():
            contact = self.contact(session, arm)
            old = self.native_holds.get(arm)
            if positions is None or not contact['valid'] or contact['actor'] != pending['actor']:
                self.native_holds.pop(arm, None)
                continue
            current = positions[ARMS.index(arm)].copy()
            stable = (old['stable'] + 1 if old and old['step_id'] == session.step_id-1
                and old['actor'] == pending['actor']
                and np.max(np.abs(current-old['fingers'])) < self.config.stable_delta_m else 0)
            count = old['count']+1 if old and old['step_id'] == session.step_id-1 else 1
            self.native_holds[arm] = dict(pending, fingers=current, stable=stable, count=count,
                                         step_id=session.step_id)
            if self.plans[arm]['tightening_enabled']:
                self.tightened_actors[arm] = pending['actor']
                if self.zero_latch is not None:
                    self.zero_latch.activate(session, arm, pending['actor'],
                        self.last_model_action[ARMS.index(arm)*7+6])
            if stable >= self.config.stable_intervals and count >= self.config.contact_windows:
                self.arms[arm]['used'] = True
                self.pending_transitions.append(dict(schema='pi_force_hold_state_v1', version=self.version,
                    episode_id=session.episode_id, step_id=session.step_id, arm=arm,
                    reason='model_hold_stability_confirmed', original_samples=count,
                    stable_intervals=stable, actor=pending['actor']))
        self.pending_native = {}

    def commit_consider(self, session, committed_arms):
        """Flush only transitions derived from acknowledged original actions."""
        del committed_arms
        for transition in self.pending_transitions:
            session._write_force_hold_transition(transition)
            self.transitions.append(transition)
        self.pending_transitions = []

    def observe_ack(self, session, action, *, original=True,
                    student_prediction_id=None, proposal_action_index=None, original_action=None):
        """Track physical feedback always, but model intent only on original ACKs."""
        self.last_action = np.asarray(action, np.float32).copy()
        if original_action is not None:
            action = np.asarray(original_action, np.float32)
        if self.zero_latch is not None:
            self.zero_latch.refresh(session)
            self.zero_latch.observe_gpt_ack(session, action)
        positions = _fingers(session)
        for arm in ARMS:
            contact = self.contact(session, arm)
            if not contact['valid']:
                self.tightened_actors.pop(arm, None)
                self.plans.pop(arm, None)
                self.native_holds.pop(arm, None)
            elif arm in self.tightened_actors and self.tightened_actors[arm] != contact['actor']:
                self.tightened_actors.pop(arm, None)
                self.plans.pop(arm, None)
                self.native_holds.pop(arm, None)
        if original:
            self.observe_model_hold(session, positions)
            self.last_model_action = np.asarray(action, np.float32).copy()
            self.last_model_reference = dict(prediction_id=student_prediction_id,
                proposal_action_index=proposal_action_index, step_id=session.step_id)
        for i, arm in enumerate(ARMS):
            state = self.arms[arm]
            previous_fingers = state['fingers']
            state['physical_finger_delta_m'] = (None if positions is None or previous_fingers is None
                else float(np.max(np.abs(positions[i] - previous_fingers))))
            state['fingers'] = None if positions is None else positions[i].copy()
            internal = float(session.obs['states'][i * 7 + 6])
            state['internal_target_delta'] = None if state['internal_target'] is None else internal - state['internal_target']
            state['internal_target'] = internal
            state['contact'] = self.contact(session, arm)
            if not original:
                continue
            value = float(action[i * 7 + 6])
            if state['action_high'] is None:
                state['action_high'] = state['action_low'] = value
            elif self.opening(arm, value):
                transition = dict(schema='pi_force_hold_state_v1', version=self.version,
                    episode_id=session.episode_id, step_id=session.step_id, arm=arm,
                    reason='opening_trend_cancelled_and_rearmed', target=value,
                    prior_min_target=state['action_low'], used_before=state['used'])
                self.pending_transitions.append(transition)
                state.update(closing=False, used=False, action_high=value, action_low=value,
                             model_history=[], model_step_ids=[], model_finger_history=[],
                             model_internal_history=[], closing_finger_baseline_m=None,
                             closing_internal_baseline=None, physical_net_closing_m=None,
                             closing_evidence_start_step_id=None, closing_evidence_detected_step_id=None,
                             physical_baseline_step_id=None,
                             physical_closing_seen=False, internal_net_closing=None,
                             internal_closing_seen=False)
            else:
                state['action_high'] = max(state['action_high'], value)
                state['action_low'] = min(state['action_low'], value)
                window = (state['model_history'] + [value])[-self.config.action_history_size:]
                state['action_window_net_change'] = window[-1] - window[0]
                if len(window) >= 2 and -state['action_window_net_change'] + 1e-7 >= self.config.action_trend_delta:
                    if not state['closing']:
                        start = max(0, len(state['model_history']) + 1 - self.config.action_history_size)
                        state['closing_finger_baseline_m'] = state['model_finger_history'][start]
                        state['closing_internal_baseline'] = state['model_internal_history'][start]
                        state['closing_evidence_start_step_id'] = state['model_step_ids'][start]
                        state['closing_evidence_detected_step_id'] = session.step_id
                        state['physical_baseline_step_id'] = state['model_step_ids'][start]
                    state['closing'] = True
                if state['closing']:
                    baseline = state['closing_finger_baseline_m']
                    if baseline is None and positions is not None:
                        state['closing_finger_baseline_m'] = baseline = positions[i].tolist()
                        state['physical_baseline_step_id'] = session.step_id
                    if positions is not None and baseline is not None:
                        net = (positions[i] - np.asarray(baseline)) * self.finger_closing_directions[arm]
                        state['physical_net_closing_m'] = net.tolist()
                        if np.all(net >= self.config.physical_closing_delta_m):
                            state['physical_closing_seen'] = True
                    internal_base = state['closing_internal_baseline']
                    if internal_base is not None:
                        state['internal_net_closing'] = float(internal_base - internal)
                        if state['internal_net_closing'] >= self.config.internal_trend_delta:
                            state['internal_closing_seen'] = True
            state['model_finger_history'].append(None if positions is None else positions[i].tolist())
            state['model_internal_history'].append(internal)
            state['model_finger_history'] = state['model_finger_history'][-self.config.action_history_size:]
            state['model_internal_history'] = state['model_internal_history'][-self.config.action_history_size:]
            state['model_history'].append(value)
            state['model_step_ids'].append(session.step_id)
            state['model_history'] = state['model_history'][-self.config.action_history_size:]
            state['model_step_ids'] = state['model_step_ids'][-self.config.action_history_size:]

    def run_hold(self, session, action, arms, index, original_count, kwargs):
        """After verified contact only, freeze measured arms and tighten a bounded gap."""
        self.ensure_plans(session, arms)
        self.event_count += 1
        event_id = f'{session.episode_id}-pi-hold-{self.event_count:04d}'
        base = self.last_action.copy()
        for i, q in enumerate(measured_joints(session)):
            base[i * 7:i * 7 + 6] = q
        actors = {}
        for arm in arms:
            self.arms[arm]['used'] = True
            base[ARMS.index(arm) * 7 + 6] = self.last_model_action[ARMS.index(arm) * 7 + 6]
            actors[arm] = self.trigger_evidence[arm]['contact']['actor']
        plans = {arm: dict(self.plans[arm]) for arm in arms}
        ramp_steps = 1 if self.zero_latch is not None else self.config.ramp_steps
        if self.zero_latch is not None:
            for arm in arms:
                self.zero_latch.activate(session, arm, actors[arm],
                    self.last_model_action[ARMS.index(arm)*7+6])
        # Native holding may already have supplied some or all tightening.
        for arm in arms:
            offset = ARMS.index(arm)*7+6
            plans[arm]['remaining_extra_gap_m'] = max(0.,
                2*(-.010+float(base[offset])*SINGLE_FINGER_SCALE_M)
                - plans[arm]['requested_final_target_width_m'])
        ramp_active = any(plan['remaining_extra_gap_m'] > 1e-9 for plan in plans.values())
        native = {arm: self.native_holds.get(arm) for arm in arms}
        reuse_native = not ramp_active and all(n and n['step_id'] == session.step_id
            and n['actor'] == actors[arm] for arm, n in native.items())
        event = dict(schema=SCHEMA, version=self.version, event_id=event_id, status='running',
            end_reason=None, episode_id=session.episode_id,
            student_prediction_id=kwargs.get('student_prediction_id'),
            trigger_proposal_action_index=index, insert_before_proposal_action_index=index,
            insert_after_step_id=session.step_id, held_action=base.tolist(),
            margin_plans=plans, reused_model_hold_samples={a: n['count'] for a, n in native.items()} if reuse_native else {},
            base_held_action=base.tolist(), base_model_action=self.last_model_action.tolist(),
            base_prediction_id=self.last_model_reference['prediction_id'],
            base_proposal_action_index=self.last_model_reference['proposal_action_index'],
            base_model_step_id=self.last_model_reference['step_id'],
            raw_target_ref=dict(prediction_id=self.last_model_reference['prediction_id'],
                proposal_action_index=self.last_model_reference['proposal_action_index'],
                field='raw_actions', gripper_indices=[6, 13]),
            base_clipped_model_target={arm: float(self.last_model_action[i * 7 + 6])
                                       for i, arm in enumerate(ARMS)},
            base_model_action_semantics='last_executed_original_native_command_after_clip',
            closing_arms=arms, contact_actors=actors, actual_count=0, tighten_steps=0,
            steps=[], samples=[], collision_checks=[], original_action_count=original_count,
            discarded_proposal_range=[index, original_count], settings=asdict(self.config),
            single_finger_scale_m=SINGLE_FINGER_SCALE_M,
            dual_arm_chunk_paused=True, reinfer_required=True, grasp_verified=False,
            allowed_object_paths_resolved=sorted(self.allowed_actors),
            trigger_action=np.asarray(action).tolist(), trigger_evidence=self.trigger_evidence,
            trigger_before_observation=f'observations/{session.step_id:06d}.npz',
            trigger_before=dict(step_id=session.step_id,
                finger_positions_m=None if _fingers(session) is None else _fingers(session).tolist(),
                force_window_step_id=(session.gripper_contact_cache or {}).get('step_id'),
                force_window_episode_id=(session.gripper_contact_cache or {}).get('episode_id')))
        rows, previous = [], _fingers(session).copy() if reuse_native else None
        stable = {arm: native[arm]['stable'] if reuse_native else 0 for arm in arms}
        continuous = {arm: native[arm]['count'] if reuse_native else 0 for arm in arms}
        increments = {arm: 0 for arm in arms}
        requested = {arm: 0. for arm in arms}
        previous_actual = {arm: 0. for arm in arms}
        increase_stop_reason = {arm: None for arm in arms}
        session._write_force_hold_event(event)
        try:
            for tick in range(1, self.config.max_extra_steps + 1):
                if session.terminated or session.truncated:
                    event['end_reason'] = 'native_terminal'
                    break
                if session.step_id >= session.metadata['max_episode_steps']:
                    event['end_reason'] = 'step_budget'
                    break
                contacts = {arm: self.contact(session, arm) for arm in arms}
                if any(not c['valid'] for c in contacts.values()):
                    event.update(end_reason='contact_lost_or_invalid', final_contact=contacts)
                    break
                if any(contacts[arm]['actor'] != actors[arm] for arm in arms):
                    event.update(end_reason='contact_object_changed', final_contact=contacts)
                    break
                pre_step_id = session.step_id
                pre_positions = _fingers(session).copy()
                pre_internal_state = np.asarray(session.obs['states']).copy()
                command = base.copy()
                compensation = {}
                increased = False
                for arm in arms:
                    increments[arm] = min(tick, ramp_steps) if ramp_active else 0
                    # Fixed event reference: neither physical feedback nor force rebases this target.
                    requested[arm] = plans[arm]['remaining_extra_gap_m'] * increments[arm] / ramp_steps
                    offset = ARMS.index(arm) * 7 + 6
                    reduction = requested[arm] / (2 * SINGLE_FINGER_SCALE_M)
                    command[offset] = np.clip(float(base[offset]) - reduction, 0, 1)
                    if self.zero_latch is not None:
                        command[offset] = 0.
                    actual_norm = max(0., float(base[offset]) - float(command[offset]))
                    if self.zero_latch is None and actual_norm > reduction:
                        # Round toward less tightening so float32 never exceeds the cap.
                        command[offset] = np.nextafter(command[offset], np.float32(np.inf))
                        actual_norm = max(0., float(base[offset]) - float(command[offset]))
                    if actual_norm > previous_actual[arm] + 1e-10:
                        increased = True
                    previous_actual[arm] = actual_norm
                    if command[offset] <= 0:
                        increase_stop_reason[arm] = 'clipped_target_exhausted'
                    elif not ramp_active or tick >= ramp_steps:
                        increase_stop_reason[arm] = 'margin_target_reached'
                    compensation[arm] = dict(requested_cumulative_gap_reduction_m=requested[arm],
                        actual_cumulative_gap_reduction_m=actual_norm * 2 * SINGLE_FINGER_SCALE_M,
                        requested_normalized_reduction=reduction, actual_normalized_reduction=actual_norm,
                        base_model_target=float(base[offset]), issued_target=float(command[offset]),
                        increase_count=increments[arm], increase_stop_reason=increase_stop_reason[arm])
                actual = measured_joints(session)
                collision = dict(next_step_id=session.step_id + 1, arms={})
                for i, arm in enumerate(ARMS):
                    collision['arms'][arm] = session.kinematics.check_self_collision(
                        arm, np.vstack([actual[i], command[i * 7:i * 7 + 6]]))
                event['collision_checks'].append(collision)
                session._write_force_hold_event(event)
                result = session._chunk_step_native([command], hold_event_id=event_id,
                    proposal_action_index=None, trigger_proposal_action_index=index,
                    hold_base_action=base.tolist(), hold_compensation=compensation,
                    hold_tightening_increased=increased, **kwargs)
                rows.extend(result['steps'])
                if result['steps']:
                    for arm in arms:
                        if plans[arm]['tightening_enabled']:
                            self.tightened_actors[arm] = actors[arm]
                if increased:
                    event['tighten_steps'] += len(result['steps'])
                event.update(steps=[row['step_id'] for row in rows], actual_count=len(rows))
                positions = _fingers(session)
                post = {arm: self.contact(session, arm) for arm in arms}
                sample = dict(step_id=session.step_id, actual_command=np.asarray(result['steps'][-1]['executed_action']).tolist() if result['steps'] else command.tolist(),
                    pre_observation_step_id=pre_step_id, post_observation_step_id=session.step_id,
                    pre_observation_ref=f'observations/{pre_step_id:06d}.npz',
                    post_observation_ref=f'observations/{session.step_id:06d}.npz',
                    pre_contact=contacts, post_contact=post,
                    pre_finger_positions_m=pre_positions.tolist(),
                    post_finger_positions_m=None if positions is None else positions.tolist(),
                    pre_internal_target_state=pre_internal_state.tolist(),
                    post_internal_target_state=np.asarray(session.obs['states']).tolist(),
                    compensation=compensation, force_window_step_id=(session.gripper_contact_cache or {}).get('step_id'),
                    contacts=post, stable_intervals=dict(stable))
                event['samples'].append(sample)
                if any(not c['valid'] for c in post.values()) or positions is None:
                    event['end_reason'] = 'contact_lost_or_invalid'
                elif any(post[arm]['actor'] != actors[arm] for arm in arms):
                    event['end_reason'] = 'contact_object_changed'
                else:
                    for arm in arms:
                        continuous[arm] += 1
                        a = ARMS.index(arm)
                        # Margin: compare samples 2->3->4; zero: 1->2->3.
                        eligible = not ramp_active or tick >= ramp_steps
                        stable[arm] = (stable[arm] + 1 if eligible and previous is not None and
                            np.max(np.abs(positions[a] - previous[a])) < self.config.stable_delta_m else 0)
                    previous = positions.copy()
                    sample.update(stable_intervals=dict(stable), continuous_contact_windows=dict(continuous),
                        increase_stop_reason=dict(increase_stop_reason),
                        compensation_stopped=all(reason is not None for reason in increase_stop_reason.values()))
                    if all(stable[a] >= self.config.stable_intervals and
                             continuous[a] >= self.config.contact_windows for a in arms):
                        event['end_reason'] = 'contact_and_finger_stability_confirmed'
                session._write_force_hold_event(event)
                if session.terminated or session.truncated:
                    event['end_reason'] = 'native_terminal'
                if event['end_reason']:
                    break
            else:
                event['end_reason'] = 'contact_tighten_timeout'
        except SelfCollisionError as error:
            event.update(end_reason='self_collision', error=str(error), collision_failure=dict(
                error.details, arm=arm, stage='pi_contact_tighten_measured_to_held_chord',
                next_step_id=session.step_id + 1))
        except Exception as error:
            event.update(end_reason='execution_or_persistence_error', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            event['actual_count'] = session.step_id - event['insert_after_step_id']
            event['steps'] = [row['step_id'] for row in rows]
            event['ack_rows_missing'] = event['actual_count'] - len(rows)
            event['increase_stop_reason'] = dict(increase_stop_reason)
            event['status'] = 'ended'
            if event['end_reason'] is None:
                event['end_reason'] = 'interrupted'
            session._write_force_hold_event(event)
        return rows, event
