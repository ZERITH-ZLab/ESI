"""Read-only PhysX contact-report telemetry; never a grasp/control predicate.

The native report contains a world-frame normal contact impulse. This module preserves
that raw quantity and its actor ordering. Force is the impulse divided by the
actual physics timestep, not the action period. Friction forces are not included in this report. Empty valid samples count as zero in action-window statistics.
"""
from copy import deepcopy
import logging

import numpy as np

ARMS = ('left', 'right')
FINGERS = ('link7', 'link8')
SCHEMA = 'gripper_contact_impulse_v1'


def require_supported_backend(device):
    """Reject unvalidated GPU reports before changing any reporting settings."""
    if str(device) != 'cpu':
        raise RuntimeError(f'unsupported_physics_backend:{device}; contact telemetry requires cpu PhysX')


class ContactWindow:
    """Accumulate every physics step of one acknowledged control action."""

    def __init__(self, fingers, physics_dt, control_dt):
        self.fingers = deepcopy(fingers)
        self.physics_dt, self.control_dt = float(physics_dt), float(control_dt)
        if self.physics_dt <= 0 or self.control_dt <= 0:
            raise ValueError('Positive native physics/control timesteps required')
        self.active = False
        self.cache = self._empty(0, 'no_action_window')

    def _empty(self, step_id, reason):
        return dict(schema=SCHEMA, step_id=step_id, valid=False, reason=reason,
                    frame='world', physics_dt_s=self.physics_dt,
                    expected_control_dt_s=self.control_dt, samples=[], arms={},
                    force_semantics='normal_contact_impulse_divided_by_physics_dt',
                    force_kind='normal_contact', friction_included=False, supported_physics_backend='cpu',
                    vector_convention='native_impulse_on_actor0; negated_for_actor1',
                    normal_friction_decomposition=False, grasp_verified=False,
                    contact_ratio_semantics='fraction of substeps with contact points; not load threshold',
                    valid_semantics='complete report window; all fingers in query scope awake in every substep',
                    backend_buffer_overflow_observable=False,
                    sleeping_contact_force_reconstruction=False)

    def clear(self):
        """Discard reset/settling samples without advancing simulation."""
        self.active = False
        self.cache = self._empty(0, 'no_action_window')

    def begin(self, step_id):
        """Begin an action window, excluding all prior settling samples."""
        self.pending = self._empty(int(step_id), 'collecting')
        self.errors = []
        self.active = True

    def record_error(self, error):
        """Record a diagnostic failure without affecting robot execution."""
        if self.active:
            self.errors.append(f'{type(error).__name__}: {error}')

    def record_reports(self, headers, data, path_decoder=str, sleeping=None):
        """Copy one native post-physics report before its buffers are reused."""
        if not self.active:
            return
        sample = dict(substep=len(self.pending['samples']) + 1, contacts=[],
                      sleeping=dict(sleeping or {}))
        paths = {path for values in self.fingers.values() for path in values.values()}
        for header in headers:
            actor0, actor1 = (str(path_decoder(getattr(header, name)))
                              for name in ('actor0', 'actor1'))
            if actor0 not in paths and actor1 not in paths:
                continue
            start, count = int(header.contact_data_offset), int(header.num_contact_data)
            if start < 0 or count < 0 or start + count > len(data):
                raise ValueError('Native contact report range is incomplete')
            for index in range(start, start + count):
                contact = data[index]
                impulse = np.asarray(contact.impulse, float)
                normal = np.asarray(contact.normal, float)
                position = np.asarray(contact.position, float)
                if (impulse.shape != (3,) or normal.shape != (3,) or position.shape != (3,)
                        or not np.isfinite(np.r_[impulse, normal, position, contact.separation]).all()):
                    raise ValueError('Nonfinite or malformed native contact report')
                sample['contacts'].append(dict(actor0=actor0, actor1=actor1,
                    collider0=str(path_decoder(header.collider0)),
                    collider1=str(path_decoder(header.collider1)),
                    impulse_world_Ns=impulse.tolist(), normal_world=normal.tolist(),
                    position_world_m=position.tolist(), separation_m=float(contact.separation),
                    event_type=str(header.type)))
        self.pending['samples'].append(sample)

    def finish(self):
        """Seal the last action; invalid windows never fabricate zero forces."""
        self.active = False
        count = len(self.pending['samples'])
        duration = count * self.physics_dt
        expected = self.control_dt / self.physics_dt
        if not np.isclose(count, expected, rtol=0, atol=1e-6):
            self.errors.append(f'physics_sample_count:{count}; expected:{expected}')
        self.pending.update(sample_count=count, duration_s=duration,
                            valid=not self.errors, reason='ok' if not self.errors else '; '.join(self.errors))
        self.pending['reports_valid'] = not self.errors
        self.pending['arms'] = self._summarize(self.pending['samples']) if not self.errors else {}
        if not self.errors:
            self.pending['valid'] = self._all_known(self.pending['arms'])
            if not self.pending['valid']:
                self.pending['reason'] = 'one_or_more_fingers_sleeping_force_unknown'
        self.cache = deepcopy(self.pending)
        return self.query()

    def _summarize(self, samples):
        result = {}
        for arm, fingers in self.fingers.items():
            result[arm] = {}
            for name, path in fingers.items():
                asleep = any(sample.get('sleeping', {}).get(path, False) for sample in samples)
                if asleep:
                    result[arm][name] = dict(prim_path=path, valid=False,
                        reason='sleeping_contact_impulses_not_reported', objects={})
                    continue
                objects = {}
                for index, sample in enumerate(samples):
                    for contact in sample['contacts']:
                        if path not in (contact['actor0'], contact['actor1']):
                            continue
                        first = contact['actor0'] == path
                        other = contact['actor1'] if first else contact['actor0']
                        entry = objects.setdefault(other, dict(
                            forces=np.zeros((len(samples), 3)), contact=np.zeros(len(samples), bool)))
                        sign = 1.0 if first else -1.0
                        entry['forces'][index] += sign * np.asarray(contact['impulse_world_Ns']) / self.physics_dt
                        entry['contact'][index] = True
                pairs = {}
                total = np.zeros((len(samples), 3))
                any_contact = np.zeros(len(samples), bool)
                for other, entry in objects.items():
                    total += entry['forces']
                    any_contact |= entry['contact']
                    pairs[other] = self._statistics(entry['forces'], entry['contact'])
                result[arm][name] = dict(prim_path=path, objects=pairs, valid=True, reason='ok',
                                        **self._statistics(total, any_contact))
        return result

    @staticmethod
    def _all_known(arms):
        return bool(arms) and all(entry['valid'] for fingers in arms.values() for entry in fingers.values())

    @staticmethod
    def _statistics(forces, contacts):
        magnitude = np.linalg.norm(forces, axis=1)
        return dict(force_world_mean_N=forces.mean(axis=0).tolist(),
                    force_magnitude_mean_N=float(magnitude.mean()),
                    force_magnitude_peak_N=float(magnitude.max()),
                    contact_ratio=float(contacts.mean()),
                    force_world_substeps_N=forces.tolist())

    def query(self, arm=None, object_id=None):
        """Return a detached cached receipt; object_id is a native contact actor/collider path."""
        if arm is not None and arm not in self.fingers:
            raise ValueError(f'Unknown arm: {arm}')
        value = deepcopy(self.cache)
        if arm is not None:
            value['arms'] = {arm: value['arms'][arm]} if arm in value['arms'] else {}
        if object_id is not None:
            selected_paths = {path for values in self.fingers.values() for path in values.values()}
            for sample in value['samples']:
                sample['contacts'] = [contact for contact in sample['contacts']
                    if ((contact['actor0'] == object_id and contact['actor1'] in selected_paths)
                        or (contact['actor1'] == object_id and contact['actor0'] in selected_paths))]
            if value.get('reports_valid'):
                value['arms'] = self._summarize(value['samples'])
                if arm is not None:
                    value['arms'] = {arm: value['arms'][arm]}
            value['object_filter'] = object_id
            value['aggregate_scope'] = 'requested rigid object only'
        if value.get('reports_valid'):
            value['valid'] = self._all_known(value['arms'])
            value['reason'] = 'ok' if value['valid'] else 'one_or_more_fingers_sleeping_force_unknown'
        return value

    def observation(self):
        """Return numeric NPZ-safe summaries, never pickle-requiring dictionaries."""
        mask = np.zeros((2, 2), bool)
        result = dict(gripper_contact_valid=self.cache['valid'],
                      gripper_contact_step_id=self.cache['step_id'])
        for a, arm in enumerate(ARMS):
            for f, finger in enumerate(FINGERS):
                entry = self.cache['arms'].get(arm, {}).get(finger, {})
                if not entry.get('valid'):
                    continue
                mask[a, f] = True
                prefix = f'gripper_contact_{arm}_{finger}_'
                result[prefix + 'force_world_mean_N'] = np.asarray(entry['force_world_mean_N'])
                result[prefix + 'force_magnitude_peak_N'] = entry['force_magnitude_peak_N']
                result[prefix + 'contact_ratio'] = entry['contact_ratio']
        result['gripper_contact_valid_mask'] = mask
        return result


class NativeContactMonitor(ContactWindow):
    """Attach reporting to four native fingers and sample after each real step."""

    def __init__(self, sim_context, fingers, control_dt):
        require_supported_backend(sim_context.device)
        import carb
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, PhysicsSchemaTools
        super().__init__(fingers, sim_context.get_physics_dt(), control_dt)
        self.sim_context = sim_context
        self._original_step = sim_context.step
        self._had_step_attribute = 'step' in vars(sim_context)
        self._prior_step_attribute = vars(sim_context).get('step')
        self._reports = get_physx_simulation_interface()
        self._decode = PhysicsSchemaTools.intToSdfPath
        self._stage_id = omni.usd.get_context().get_stage_id()
        self._encoded_fingers = {path: PhysicsSchemaTools.sdfPathToInt(path)
            for values in fingers.values() for path in values.values()}
        self._report_prims = []
        self._settings = carb.settings.get_settings()
        self._contact_disabled_before = self._settings.get_as_bool('/physics/disableContactProcessing')
        self._settings.set_bool('/physics/disableContactProcessing', False)
        stage = omni.usd.get_context().get_stage()
        try:
            self._enable_reports(stage, fingers, PhysxSchema)
        except Exception:
            self._restore_reports()
            raise
        def monitored_step(*args, **kwargs):
            value = self._original_step(*args, **kwargs)
            if self.active:
                try:
                    if self._settings.get_as_bool('/physics/disableContactProcessing'):
                        raise RuntimeError('Native contact reporting is disabled')
                    sleeping = {path: self._reports.is_sleeping(self._stage_id, encoded)
                        for path, encoded in self._encoded_fingers.items()}
                    self.record_reports(*self._reports.get_contact_report(), self._decode, sleeping)
                except Exception as error:
                    logging.exception('Gripper contact telemetry failed after native physics step')
                    self.record_error(error)
            return value
        self._installed_step = monitored_step
        sim_context.step = monitored_step
        self.reporting_configuration = dict(contact_report_api_fingers=deepcopy(fingers),
            report_threshold=0.0, contact_processing_previously_disabled=self._contact_disabled_before,
            contact_processing_enabled=True, sampling='after original SimulationContext.step')

    def _enable_reports(self, stage, fingers, PhysxSchema):
        for values in fingers.values():
            for path in values.values():
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise ValueError(f'Contact finger prim not found: {path}')
                existed = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                attr = api.GetThresholdAttr()
                authored, previous = attr.HasAuthoredValueOpinion(), attr.Get()
                self._report_prims.append((prim, existed, authored, previous))
                api.CreateThresholdAttr(0.0)
    def close(self):
        """Restore this instance's hook and reporting attributes without stepping."""
        from pxr import PhysxSchema
        self.active = False
        if self.sim_context.step is self._installed_step:
            if self._had_step_attribute:
                self.sim_context.step = self._prior_step_attribute
            else:
                del self.sim_context.step
        self._restore_reports()

    def _restore_reports(self):
        from pxr import PhysxSchema
        for prim, existed, authored, previous in self._report_prims:
            if not prim.IsValid():
                continue
            if not existed:
                prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
            else:
                attr = PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr()
                attr.Set(previous) if authored else attr.Clear()
        self._report_prims.clear()
        self._settings.set_bool('/physics/disableContactProcessing', self._contact_disabled_before)


def for_environment(env, control_dt):
    """Resolve actual X5 rigid finger prims for the single-environment session."""
    import omni.usd
    from pxr import Usd, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    fingers = {}
    manager = env.robot_manager
    for index, robot in enumerate(manager.robot_list):
        if robot.type != 'target':
            continue
        arm = robot.arm_name.split('_')[0]
        key = manager.robot_key[index]
        roots = key.root_physx_view.prim_paths
        if len(roots) != 1:
            raise ValueError('Contact telemetry requires exactly one environment')
        root = stage.GetPrimAtPath(roots[0])
        matches = {name: [str(prim.GetPath()) for prim in Usd.PrimRange(root)
                          if prim.GetName() == name and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                   for name in FINGERS}
        if any(len(paths) != 1 for paths in matches.values()):
            raise ValueError(f'Ambiguous native finger prims: {matches}')
        fingers[arm] = {name: paths[0] for name, paths in matches.items()}
    if set(fingers) != set(ARMS):
        raise ValueError('Both target arms required for contact telemetry')
    return NativeContactMonitor(env.sim.sim, fingers, control_dt)
