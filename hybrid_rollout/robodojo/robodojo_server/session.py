"""One RoboDojo episode, native action stepping and success/timeout checks."""
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import uuid
import numpy as np
from PIL import Image

from ..io import write_json, require, sha256
from ..robodojo_server.decision_log import record_decision
from .kinematics import DualKinematics

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def register_native_evaluation(env):
    """Mirror EvalEnv.run_eval's setup, without entering its policy loop.

    Native reset clears the checks. With no registered conditions the native
    RewardManager reports success immediately, so fail closed on that state.
    These remain simulator-only predicates, never teacher planning inputs.
    """
    env.run_reward()
    if hasattr(env, 'get_score'):
        env.get_score()
    names = ('check_list', 'final_check_list', 'trigger_check_list')
    counts = {name: [len(group) for group in getattr(env.reward_manager, name)] for name in names}
    require(all(sum(counts[name][i] for name in names) > 0 for i in range(env.num_envs)),
            'Native task completion conditions are empty; refusing vacuous success')
    support_envs = []
    if getattr(env, 'interact', False) and hasattr(env, 'query_support_arm_traj'):
        support_envs = list(env.get_running_env_idx_list())
        for env_idx in support_envs:
            env.query_support_arm_traj(env_idx=env_idx)
    return dict(registered=True, source='native EvalEnv.run_eval preamble',
                condition_group_counts=counts, initial_support_arm_query_envs=support_envs)


class RoboDojoSession:
    def __init__(self, env, output, task, evaluation_identity=None, *, scene_query_enabled=False,
                 motion_enabled=False, step_budget=None, pi05_collision_shadow=False,
                 gripper_hold_config=None, gripper_contact_enabled=True,
                 pi_force_hold_enabled=False, pi_force_hold_config=None):
        from .gripper_hold import configured
        self.gripper_hold_config = configured(gripper_hold_config)
        if pi_force_hold_enabled and not gripper_contact_enabled:
            raise ValueError('pi force hold requires gripper contact telemetry enabled')
        from .pi_force_hold import PiForceHold, configured as force_configured, execution_version
        self.pi_force_hold = PiForceHold(force_configured(pi_force_hold_config)) if pi_force_hold_enabled else None
        self.gripper_contact_enabled = bool(gripper_contact_enabled)
        self.pi05_collision_shadow = pi05_collision_shadow
        self.motion_enabled = motion_enabled
        self.scene_query_enabled = scene_query_enabled
        self.env, self.output = env, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode_id, self.step_id = None, 0
        self.terminated = self.truncated = self.success = False
        self.poisoned, self.finished = True, False
        self.video_writer, self.video_frames = None, 0
        self.source, self.control_epoch = 'student', 0
        self.evaluation_identity = evaluation_identity
        self.finish_reason = None
        self.step_budget = step_budget or dict(original_native_step_limit=int(env.step_lim),
            effective_step_limit=int(env.step_lim), step_limit_overridden=False, budget_protocol='native')
        self.metadata = dict(task=task, instruction=task, simulator='RoboDojo',
            robot_adapter='dual_arx_x5', control_dt=1/env.obs_manager.collect_freq,
            max_episode_steps=int(env.step_lim), action_dim=14, action_horizon=50,
            frame='environment_origin', eef_link='link6', no_rollback=True,
            supports_fk_preview=True, supports_codex_decision_recording=True,
            cameras=list(CAMERAS), gripper_semantics='continuous_0_closed_1_open')
        if self.step_budget['step_limit_overridden']:
            self.metadata['step_budget'] = self.step_budget
        self.metadata['pi05_collision_shadow'] = dict(enabled=pi05_collision_shadow,
            method='sampled_spheres_v1', gating=False, extra_physics_steps=0,
            scope='sampled_configured_spheres_not_PhysX_contact_truth')
        self.metadata['gripper_hold_config'] = (None if self.gripper_hold_config is None
            else dict(vars(self.gripper_hold_config)))
        self.gripper_contact = None
        self.gripper_contact_error = None
        self.gripper_contact_cache = None
        self.metadata['gripper_contact'] = dict(enabled=self.gripper_contact_enabled, gating=pi_force_hold_enabled,
            source='native PhysX contact reports', frame='world', extra_physics_steps=0,
            force_semantics='normal contact impulse / physics dt; friction excluded',
            archive='episode/gripper_contact.jsonl', schema='gripper_contact_impulse_v1')
        self.metadata['pi_force_hold'] = dict(enabled=bool(pi_force_hold_enabled),
            version=execution_version(force_configured(pi_force_hold_config)), experimental_initial_thresholds=True, config=None if self.pi_force_hold is None else
            dict(vars(self.pi_force_hold.config)), dual_arm_chunk_pause=True,
            archive='episode/pi_force_hold_events.jsonl', schema='pi_force_hold_event_v1',
            grasp_verified=False)
        self.metadata['short_motion_executor'] = 'complete_target_quintic_v1'
        self.metadata['short_motion_steps_semantics'] = 'requested_minimum_expand_to_60_or_remaining'
        if motion_enabled:
            from .joint_motion import VERSION
            self.metadata['motion_executor'] = VERSION
        if scene_query_enabled:
            from ..manipulation.simulator_query import INFORMATION_ACCESS
            from ..manipulation.simulator_query import FIELDS
            self.metadata['scene_query'] = dict(enabled=True,
                information_access=INFORMATION_ACCESS,
                fields=list(FIELDS), renders=True, advances_physics=False)
        # Our recorder owns exactly one frame per actual control ACK. Disable
        # only upstream duplicate video capture, not its native task checks.
        env._stream_vision = lambda *args, **kwargs: None

    def _check_identity(self, episode_id, step_id):
        require(not self.poisoned and episode_id == self.episode_id and step_id == self.step_id,
                'Stale episode/tick or uncertain simulator state')

    def _observe(self, record=False):
        raw = self.env.get_obs()
        state = raw['state']
        states = np.concatenate([np.r_[state[f'{arm}_arm_joint_state'], state[f'{arm}_ee_joint_state']]
                                 for arm in ('left', 'right')]).astype(np.float32)
        require(states.shape == (14,) and np.isfinite(states).all(), 'Invalid native dual-arm state')
        images = {}
        aliases = dict(cam_high=('cam_high', 'cam_head', 'head_camera', 'top_camera'),
                       cam_left_wrist=('cam_left_wrist', 'left_camera'),
                       cam_right_wrist=('cam_right_wrist', 'right_camera'))
        for key, names in aliases.items():
            source = next((n for n in names if n in raw['vision']), None)
            require(source is not None, f'Missing {key}; native cameras={list(raw["vision"])}')
            images[key] = np.asarray(raw['vision'][source]['color'], np.uint8)[..., :3]
        poses = np.asarray([state[f'{arm}_ee_pose'] for arm in ('left', 'right')], np.float32)
        self.obs = dict(**images, states=states, eef_positions=poses[:, :3],
            eef_quaternions_wxyz=poses[:, 3:], instruction=raw['instruction'],
            remaining_steps=max(0, self.metadata['max_episode_steps']-self.step_id))
        from .gripper_hold import telemetry
        self.obs.update(telemetry(self.env, self.kinematics, self.step_id))
        if self.gripper_contact is not None:
            self.obs.update(self.gripper_contact.observation())
        else:
            self.obs.update(gripper_contact_valid=False, gripper_contact_step_id=self.step_id)
        if record:
            directory = self.episode_dir/'observations'
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/f'{self.step_id:06d}.npz', **self.obs)
            if self.video_writer is None:
                import imageio.v2 as imageio
                self.video_writer = imageio.get_writer(str(self.output/'sensors.mp4'),
                    fps=1/self.metadata['control_dt'], codec='libx264', pixelformat='yuv420p',
                    macro_block_size=2, output_params=['-movflags', '+faststart'])
            frames = [np.asarray(Image.fromarray(images[k]).resize((640, 360))) for k in CAMERAS]
            self.video_writer.append_data(np.concatenate(frames, axis=1))
            self.video_frames += 1
        return self.obs

    def reset(self, seed, source, policy_version):
        require(self.episode_id is None, 'One fresh episode only; no rollback/reset')
        if self.pi_force_hold is not None and self.pi_force_hold.config.tighten_enabled:
            from .pi_force_hold import validate_tightening_policy
            validate_tightening_policy(self.pi_force_hold.config, policy_version)
        if self.evaluation_identity:
            require(seed == self.evaluation_identity['layout_id'], 'Reset seed differs from the frozen layout_id')
            actual_layout = self.env.seed_manager.seed_info[seed]['scene_layout']
            require(sha256(actual_layout) == self.evaluation_identity['layout_sha256'],
                    'Native layout changed before reset')
        try:
            self.env.reset(seed=[seed])
        except Exception as error:
            # Only the native named exception is classified as an invalid layout;
            # infrastructure/model errors are never silently removed from evaluation.
            write_json(self.output/'evaluation_outcome.json', dict(
                evaluation_case=self.evaluation_identity, complete=False, valid_for_success_rate=False,
                status='invalid_native_layout' if type(error).__name__ == 'UnStableError' else 'reset_error',
                error_type=type(error).__name__, error=str(error)))
            raise
        # Native reset builds robot_key; install reporting afterwards without a hidden step.
        if self.gripper_contact_enabled:
            from .gripper_contact import for_environment
            try:
                self.gripper_contact = for_environment(self.env, self.metadata['control_dt'])
                self.gripper_contact.clear()
                self.metadata['gripper_contact']['reporting_configuration'] = (
                    self.gripper_contact.reporting_configuration)
            except Exception as error:
                logging.exception('Gripper contact telemetry initialization failed')
                self.gripper_contact_error = f'{type(error).__name__}: {error}'
        registration = register_native_evaluation(self.env)
        require(int(self.env.step_lim) == self.metadata['max_episode_steps'], 'Native reset changed the step budget')
        if self.scene_query_enabled:
            self.metadata['scene_query']['cameras'] = list(
                self.env.camera_manager.camera_names[0])
        write_json(self.output/'native_evaluation.json', registration)
        self.episode_id = uuid.uuid4().hex
        self.episode_dir = self.output/self.episode_id
        self.episode_dir.mkdir()
        self.poisoned = False
        self.kinematics = DualKinematics(self.env)
        # Fail startup before any policy ACK if either geometry/backend is invalid.
        write_json(self.output/'collision_geometry.json',
                   self.kinematics.initialize_self_collision())
        write_json(self.output/'fk_validation.json', self.kinematics.check())
        self._observe(record=True)
        if self.pi_force_hold is not None:
            self.pi_force_hold.initialize(self)
        if self.gripper_hold_config is not None:
            from .gripper_hold import sample
            sample(self)  # Validate real feedback before any policy control.
        fingerprints = {}
        for key, value in self.obs.items():
            arr = np.asarray(value)
            fingerprints[key] = dict(shape=list(arr.shape), dtype=str(arr.dtype),
                sha256=hashlib.sha256(arr.tobytes()).hexdigest())
        write_json(self.output/'initial_observation_fingerprint.json', dict(
            evaluation_case=self.evaluation_identity, fields=fingerprints,
            scope='Recorded RGB, proprio, robot EEF and instruction; not a complete physics state',
            bitwise_physics_reproducibility_guaranteed=False))
        self.metadata['instruction'] = self.obs['instruction']
        identity = hashlib.sha256(self.obs['states'].tobytes()).hexdigest()
        result = dict(episode_id=self.episode_id, step_id=0, initial_state_hash=identity,
            instruction=self.obs['instruction'], metadata=dict(self.metadata, layout_id=seed,
            eval_seed=getattr(self.env, 'eval_seed', None), evaluation_case=self.evaluation_identity,
            initial_state_hash_scope='robot_proprio_only_not_complete_scene',
            policy_version=policy_version, native_reset_settling_not_counted_as_policy_controls=True))
        write_json(self.output/'reset.json', result)
        self.record_native_stages()
        return result

    def _record_gripper_contact(self):
        """Seal and durably record one action window without changing controls."""
        if not self.gripper_contact_enabled:
            return None
        try:
            if self.gripper_contact is None:
                receipt = dict(schema='gripper_contact_impulse_v1', valid=False,
                               reason=self.gripper_contact_error or 'monitor_unavailable',
                               step_id=self.step_id, arms={}, samples=[])
            else:
                receipt = self.gripper_contact.finish()
            receipt['episode_id'] = self.episode_id
            self.gripper_contact_cache = receipt
            with (self.episode_dir/'gripper_contact.jsonl').open('a') as stream:
                stream.write(json.dumps(receipt, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            return None
        except Exception as error:
            logging.exception('Gripper contact recording failed after ACK %s', self.step_id)
            self.gripper_contact_error = f'{type(error).__name__}: {error}'
            self.gripper_contact_cache = dict(valid=False, step_id=self.step_id,
                episode_id=self.episode_id, reason=self.gripper_contact_error, arms={}, samples=[])
            if self.gripper_contact is not None:
                self.gripper_contact.cache.update(valid=False, reports_valid=False,
                    archive_valid=False, reason=self.gripper_contact_error,
                    step_id=self.step_id, arms={}, samples=[])
            if self.pi_force_hold is not None:
                self.poisoned = True
            return self.gripper_contact_error if self.pi_force_hold is not None else None

    def get_gripper_contact(self, arm=None, object_id=None):
        """Read the latest exact post-ACK contact window; no physics or rendering."""
        from copy import deepcopy
        if not self.gripper_contact_enabled:
            if arm is not None and arm not in ('left', 'right'):
                raise ValueError(f'Unknown arm: {arm}')
            return dict(enabled=False, valid=False, reason='disabled', step_id=self.step_id,
                        episode_id=self.episode_id, arms={}, samples=[])
        if self.gripper_contact is None:
            if arm is not None and arm not in ('left', 'right'):
                raise ValueError(f'Unknown arm: {arm}')
            return deepcopy(self.gripper_contact_cache or dict(valid=False,
                episode_id=self.episode_id, step_id=self.step_id, arms={}, samples=[],
                reason=self.gripper_contact_error or 'no_action_window'))
        value = self.gripper_contact.query(arm=arm, object_id=object_id)
        value['episode_id'] = self.episode_id
        return value

    def close(self):
        """Remove contact-only instrumentation before the native environment closes."""
        if self.gripper_contact is not None:
            try:
                self.gripper_contact.close()
            except Exception:
                logging.exception('Gripper contact monitor cleanup failed')

    def record_native_stages(self):
        """Persist offline-only native predicates without modifying reward checks."""
        if self.metadata['task'] != 'deposit_coin':
            return
        rm = self.env.reward_manager
        predicates = dict(lifted=rm.is_lift('coin0', z_threshold=0.08),
            inserted=rm.is_A_bbox_in_B_bbox('coin0', 'piggy_bank',
                B_bottom_functional_tag='bottom', B_top_functional_tag='center'),
            arms_home=rm.all_robot_back_to_origin())
        values = {k: bool(rm.call_func_parser(v, 0)) for k, v in predicates.items()}
        with (self.output/'native_stages.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step_id=self.step_id, **values))+'\n')

    def _write_force_hold_event(self, event):
        """Append and fsync intent/progress/final evidence before another action."""
        try:
            with (self.episode_dir/'pi_force_hold_events.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            self.poisoned = True
            raise

    def _write_force_hold_transition(self, transition):
        """Persist explicit cancellation/rearming without manufacturing a hold."""
        try:
            with (self.episode_dir/'pi_force_hold_state.jsonl').open('a') as stream:
                stream.write(json.dumps(transition, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            self.poisoned = True
            raise

    def chunk_step(self, actions, original_action_count=None, **kwargs):
        """Preserve normal pi commands; opt-in pauses always end this chunk."""
        if self.pi_force_hold is None or self.source != 'student':
            return self._chunk_step_native(actions, **kwargs)
        require(not self.poisoned, 'Uncertain simulator state')
        a = np.asarray(actions, np.float32)
        require(a.ndim == 2 and a.shape[1] == 14 and 1 <= len(a) <= 15 and np.isfinite(a).all(),
                'Expected 1..15 finite 14D absolute actions')
        require(np.all((a[:, [6, 13]] >= 0) & (a[:, [6, 13]] <= 1)), 'Invalid gripper opening')
        original_count = len(a) if original_action_count is None else original_action_count
        require(type(original_count) is int and original_count >= len(a), 'Invalid original proposal count')
        rows, events, consumed = [], [], 0
        transition_start = len(self.pi_force_hold.transitions)
        for index, action in enumerate(a):
            arms = self.pi_force_hold.consider(self, action)
            if arms and not self.pi_force_hold.prefer_model_hold(self, action, arms):
                self.pi_force_hold.commit_consider(self, arms)
                held_rows, event = self.pi_force_hold.run_hold(self, action, arms, index, original_count, kwargs)
                rows.extend(held_rows)
                events.append(event)
                break
            result = self._chunk_step_native([action], proposal_action_index=index, **kwargs)
            self.pi_force_hold.commit_consider(self, ('left', 'right') if result['steps'] else ())
            rows.extend(result['steps'])
            consumed += len(result['steps'])
            if self.terminated or self.truncated or self.poisoned:
                break
        return dict(episode_id=self.episode_id, step_id=self.step_id, steps=rows,
            force_hold_enabled=True, proposal_steps_consumed=consumed,
            force_hold_steps=sum(row.get('hold_event_id') is not None for row in rows),
            tighten_steps=sum(bool(row.get('hold_tightening_increased')) for row in rows),
            force_hold_events=events, reinfer_required=bool(events),
            force_hold_state_transitions=self.pi_force_hold.transitions[transition_start:],
            recording_error=self.gripper_contact_error if self.poisoned else None)

    def _chunk_step_native(self, actions, **kwargs):
        require(not self.poisoned, 'Uncertain simulator state')
        require(not self.finished and not self.terminated and not self.truncated, 'Episode finished')
        a = np.asarray(actions, np.float32)
        require(a.ndim == 2 and a.shape[1] == 14 and 1 <= len(a) <= 15 and np.isfinite(a).all(),
                'Expected 1..15 finite 14D absolute actions')
        require(np.all((a[:, [6, 13]] >= 0) & (a[:, [6, 13]] <= 1)), 'Invalid gripper opening')
        rows = []
        for action in a:
            require(self.step_id < self.metadata['max_episode_steps'], 'Native step budget exhausted')
            original_action = action.copy()
            override = None
            if self.pi_force_hold is not None and self.pi_force_hold.zero_latch is not None:
                action, override = self.pi_force_hold.zero_latch.prepare(self, action,
                    auxiliary=kwargs.get('hold_event_id') is not None)
            force_before = None if self.gripper_contact_cache is None else self.gripper_contact_cache['step_id']
            shadow_before = (np.asarray(self.obs['states']).copy()
                             if self.pi05_collision_shadow and self.source == 'student' else None)
            command = {key: value for arm, offset in (('left', 0), ('right', 7))
                for key, value in ((f'{arm}_arm_joint_state', action[offset:offset+6]),
                                   (f'{arm}_ee_joint_state', action[offset+6:offset+7]))}
            before = int(self.env.take_action_cnt[0])
            if self.gripper_contact is not None:
                self.gripper_contact.begin(self.step_id + 1)
            self.env.take_action(command)
            require(int(self.env.take_action_cnt[0]) == before+1, 'Native action was not executed exactly once')
            self.step_id += 1
            contact_recording_error = self._record_gripper_contact()
            from .motion_boundary import record_action
            record_action(self, action)
            if self.pi_force_hold is not None:
                self.pi_force_hold.last_action = action.copy()
            ended = bool(self.env.end_flag[0])
            self.success = bool(ended and self.env.success[0])
            self.truncated = bool(ended and not self.success and self.step_id >= self.env.step_lim)
            self.terminated = bool(ended and not self.truncated)
            obs = self._observe(record=True)
            self.record_native_stages()
            finger_obs = {key: value for key, value in obs.items()
                          if key.startswith(('finger_positions_', 'gripper_contact_'))}
            if self.pi_force_hold is not None:
                self.pi_force_hold.observe_ack(self, action,
                    original=self.source == 'student' and kwargs.get('hold_event_id') is None,
                    student_prediction_id=kwargs.get('student_prediction_id'),
                    proposal_action_index=kwargs.get('proposal_action_index'),
                    original_action=original_action)
            row = dict(valid=True, executed_action=action.copy(), obs=dict(states=obs['states'], **finger_obs),
                step_id=self.step_id, terminated=self.terminated, truncated=self.truncated, success=self.success,
                source=self.source, control_epoch=self.control_epoch, **kwargs)
            if override is not None:
                row['gripper_override'] = override
            if self.pi_force_hold is not None and self.source == 'student':
                row.update(force_window_before_step_id=force_before,
                           force_window_after_step_id=self.step_id)
            if contact_recording_error:
                row['recording_error'] = contact_recording_error
            rows.append(row)
            write_json(self.episode_dir/f'action_{self.step_id-1:06d}.json',
                dict(row, executed_action=action.tolist(), obs=dict(states=obs['states'].tolist(),
                    **{key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in finger_obs.items()})))
            if shadow_before is not None:
                try:
                    self._record_pi05_collision_shadow(shadow_before, action, obs['states'])
                except Exception as error:
                    # Shadow diagnostics never gate an acknowledged pi chunk.
                    # Native execution and primary ACK writes stay outside this block.
                    logging.exception('pi05 collision shadow diagnostic failed after ACK %s', self.step_id)
                    diagnostic_error = dict(step_id=self.step_id, source=self.source,
                        error_type=type(error).__name__, message=str(error))
                    try:
                        with (self.episode_dir/'pi05_collision_shadow_errors.jsonl').open('a') as stream:
                            stream.write(json.dumps(diagnostic_error) + '\n')
                    except OSError:
                        logging.exception('Cannot persist pi05 collision shadow diagnostic error')
            if contact_recording_error:
                break
            if ended:
                self._write_summary('terminal')
                break
        return dict(episode_id=self.episode_id, step_id=self.step_id, steps=rows)

    def _record_pi05_collision_shadow(self, actual_before, issued, actual_after):
        """Observe cached states after durable ACK recording; never gate controls."""
        began = time.perf_counter()
        cpu_began = time.process_time()
        arms = {}
        for arm, offset in (('left', 0), ('right', 7)):
            guard = self.kinematics.collision_checks[arm]
            arms[arm] = {}
            for label, state in (('actual_before', actual_before), ('issued', issued),
                                 ('actual_after', actual_after)):
                witness = guard.clearance_details(state[offset:offset + 6])
                arms[arm][label] = dict(witness, margin_m=.001,
                                       at_or_below_margin=bool(witness['gap_m'] <= .001))
        record = dict(step_id=self.step_id, source=self.source, control_epoch=self.control_epoch,
            method='sampled_spheres_v1', arms=arms, geometry_cpu_seconds=time.process_time() - cpu_began,
            geometry_wall_seconds=time.perf_counter() - began, timing_scope='six_clearance_queries_excludes_json_io',
            gating=False, extra_physics_steps=0, continuous_clearance_proven=False,
            scope='sampled_configured_spheres_not_PhysX_contact_truth')
        with (self.episode_dir/'pi05_collision_shadow.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    def _write_summary(self, reason):
        if self.episode_id is None:
            return
        if self.video_writer is not None:
            self.video_writer.close(); self.video_writer = None
        if self.finish_reason is not None:
            reason = self.finish_reason  # Do not overwrite a budget/error outcome during server cleanup.
        complete = self.terminated or self.truncated
        invalid = 0 in getattr(self.env, 'unstable_envs', set())
        eligible = complete and not invalid and not self.poisoned
        score = None
        if eligible:
            score = 1.0 if self.success else (float(self.env.reward_manager.get_score()[0])/100
                     if hasattr(self.env, 'get_score') else 0.0)
        abandonment = {}
        if reason == 'teacher_abandon' and not complete:
            # Read existing score only; do not run final checks or step physics.
            partial_score = None
            try:
                if not invalid and not self.poisoned and hasattr(self.env, 'get_score'):
                    partial_score = float(self.env.reward_manager.get_score()[0])/100
            except Exception:
                pass
            abandonment = dict(task_success=False, native_complete=False,
                               collection_complete=True, partial_score_at_abandon=partial_score)
        write_json(self.output/'evaluation_outcome.json', dict(
            evaluation_case=self.evaluation_identity, complete=complete,
            valid_for_success_rate=eligible, native_success=self.success if eligible else None,
            native_score=score, status=('teacher_abandoned' if abandonment else 'invalid_native_layout' if invalid else
                'native_completed' if eligible else 'budget_censored' if reason == 'decision_budget' else 'incomplete'),
            reason=reason, native_control_steps=self.step_id,
            native_step_limit=self.metadata['max_episode_steps'], **self.step_budget,
            valid_for_official_success_rate=eligible and not self.step_budget['step_limit_overridden'],
            success_within_original_limit=bool(self.success and eligible and
                self.step_id <= self.step_budget['original_native_step_limit']),
            success_after_original_limit=bool(self.success and eligible and
                self.step_id > self.step_budget['original_native_step_limit']), **abandonment))
        write_json(self.output/'summary.json', dict(episode_id=self.episode_id, step_id=self.step_id,
            success=self.success, terminated=self.terminated, truncated=self.truncated,
            complete=self.terminated or self.truncated, reason=reason, video_frames=self.video_frames,
            control_dt=self.metadata['control_dt'], no_rollback=True,
            evaluation_case=self.evaluation_identity))

    def dispatch(self, op, args):
        if op == 'metadata':
            return self.metadata
        if op == 'reset':
            return self.reset(**args)
        self._check_identity(args['episode_id'], args['step_id'])
        data = {k: v for k, v in args.items() if k not in ('episode_id', 'step_id')}
        if op == 'scene_query' and self.scene_query_enabled:
            from ..manipulation.simulator_query import simulator_query
            return simulator_query(self, data)
        if op == 'get_gripper_contact':
            return self.get_gripper_contact(**data)
        if op == 'teacher_observation':
            return self.obs  # Cached exact post-ACK observation; no hidden step/render.
        if op == 'record_codex_decision':
            return record_decision(self, args)
        if op == 'begin_combination':
            write_json(self.output/'combination.json', data)
            return dict(physical_steps=0)
        if op == 'switch_control_source':
            require(data['source'] in ('student', 'gpt_eef', 'gpt_joint'), 'Invalid source')
            if self.pi_force_hold is not None and data['source'] != self.source:
                self.pi_force_hold.control_switch(self, data['source'])
            self.source = data['source']; self.control_epoch += 1
            write_json(self.episode_dir/f'source_{self.control_epoch:04d}.json', dict(step_id=self.step_id, **data))
            return dict(physical_steps=0, source=self.source, control_epoch=self.control_epoch)
        if op == 'fk_preview':
            return self.kinematics.preview(data['actions'])
        if op == 'eef_short_motion':
            # Ordinary short corrections remain available without long macros.
            from .short_motion import execute
            return execute(self, **data)
        if op == 'eef_joint_target':
            return self.kinematics.target(data['targets'])
        if op == 'execute_joint_motion' and self.motion_enabled:
            from .joint_motion import execute
            return execute(self, **data)
        if op == 'chunk_step':
            return self.chunk_step(**data)
        if op == 'finish_pilot':
            self.finished = True
            self.finish_reason = data['reason']
            self._write_summary(data['reason'])
            return dict(episode_id=self.episode_id, step_id=self.step_id, success=self.success,
                terminated=self.terminated, truncated=self.truncated, video_frames=self.video_frames,
                control_dt=self.metadata['control_dt'], reason=data['reason'],
                evaluation_case=self.evaluation_identity)
        raise ValueError('Unsupported no-rollback operation: '+op)
