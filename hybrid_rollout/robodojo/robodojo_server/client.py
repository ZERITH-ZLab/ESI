"""File-backed blocking rollout API, using the copied native recorder/EEF contract.

One owner holds the simulator connection. Codex, not this class, schedules every
start -> infer -> execute cycle. No inference, retry or correction runs implicitly.
"""
import hashlib
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image

from ..io import InputError, write_json
from ..settings import MODEL, EFFORT, PROVIDER
from .protocol import RPCClient
from .gate_assessment import GATE_INSTRUCTION, previous_observation, validate_assessment
from .validation import angles, validate_response
from .action_edit_kinematics import edited_targets
from .debug_recorder import DebugVideoRecorder
from .decision_log import decision_event, emit
from .proposal_diagnostics import action_diagnostics


ARMS = ('left', 'right')


def validate_dual_response(response, request):
    if response.get('mode') in ('eef', 'edit'):
        field = 'target' if response['mode'] == 'eef' else 'edit'
        if set(response.get(field, {})) != set(ARMS):
            raise ValueError(f'{field} requires explicit left and right arms')
        for arm in ARMS:
            local = dict(response, **{field: response[field][arm]})
            local_request = dict(request, current_eef=request['current_eef'][arm],
                student_eef_trajectory=[p[arm] for p in request['student_eef_trajectory']])
            validate_response(local, local_request)
    else:
        validate_response(response, request)


class RoboDojoTools:
    profile = 'robodojo'
    camera_keys = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
    observation_keys = (*camera_keys, 'states', 'eef_positions', 'eef_quaternions_wxyz')
    validate = staticmethod(validate_dual_response)

    def current_eef(self):
        return {arm: angles(dict(position=self.obs['eef_positions'][i].tolist(),
            quaternion_wxyz=self.obs['eef_quaternions_wxyz'][i].tolist(),
            frame='environment_origin', link='link6',
            gripper_opening_command=float(self.obs['states'][i*7+6])))
            for i, arm in enumerate(ARMS)}

    def trajectory(self):
        return [dict(index=i, **{arm: angles(p[arm]) for arm in ARMS})
                for i, p in enumerate(self.fk['trajectory'])]

    def correction_targets(self, response):
        if response['mode'] == 'eef':
            return [response['target']]*response['steps']
        targets = {}
        for arm in ARMS:
            trajectory = [p[arm] for p in self.fk['trajectory']]
            targets[arm] = edited_targets(trajectory, response['steps'], response['edit'][arm])
            if response['edit'][arm]['gripper'] == 'keep':
                for target, original in zip(targets[arm], trajectory):
                    target['gripper_opening'] = original['gripper_opening']
        return [{arm: targets[arm][i] for arm in ARMS} for i in range(response['steps'])]

    def target_proposal(self, target):
        return self._rpc('eef_joint_target', targets=target)

    def execute_correction(self, response):
        """Execute complete short goals with additional GPT ticks within the budget."""
        targets = ([response['target']] if response['mode'] == 'eef'
                   else self.correction_targets(response))
        intent = ({arm: 'closed' if response['target'][arm]['gripper_closed'] else 'open'
                   for arm in ARMS} if response['mode'] == 'eef' else
                  {arm: response['edit'][arm]['gripper'] for arm in ARMS})
        result = self._rpc('eef_short_motion', targets=targets, gripper_intent=intent,
                           minimum_steps=response['steps'], decision=len(self.history),
                           teacher_request_id=self.request['request_id'])
        self.correction_result = result['correction_result']
        self.tick = result['step_id']
        return result['steps']

    def gripper_history(self, executed):
        return [{arm: bool(row[i*7+6] < .5) for i, arm in enumerate(ARMS)} for row in executed]

    def __init__(self, output, task, student, *, sim_port=19113, seed=0,
                 max_decisions=180, prompt_sha256='', rpc_factory=RPCClient):
        self.output = Path(output).resolve()
        from ..timing import Timing
        self.timing = Timing(self.output.parent/'timing/controller.jsonl')
        self.task, self.student = task, student
        self.sim_port, self.seed = sim_port, seed
        if max_decisions < 0:
            raise ValueError('max_decisions must be non-negative; 0 uses native termination only')
        self.max_decisions, self.prompt_sha256 = max_decisions, prompt_sha256
        self.rpc_factory = rpc_factory
        self.sim = None
        self.episode = None
        self.require_native_termination = False
        self.tick = 0
        self.phase = 'start'
        self.history = []
        self.preceding = None
        self.counters = dict(student_steps=0, edited_steps=0, recovery_steps=0, predictions=0)
        self.source = 'student'
        self.started = time.monotonic()
        self.recorder = DebugVideoRecorder(self.output)

    def next_call(self):
        index = len(self.history)
        if self.phase == 'start':
            return dict(tool='robodojo_start', task=self.task,
                        output_dir=str(self.output/'observations'/'000'))
        if self.phase == 'infer':
            return dict(tool='pi05_infer', observation_path=str(self.observation_path),
                        output_dir=str(self.output/'proposals'/f'{index:03d}'))
        if self.phase == 'execute':
            return dict(tool='robodojo_execute', proposal_path=str(self.proposal_path),
                        output_dir=str(self.output/'observations'/f'{index+1:03d}'),
                        response=('Supply your gate assessment and student/edit/eef decision; '
                                  'this benchmark waits for native termination'
                                  if self.require_native_termination else
                                  'Supply your gate assessment and student/edit/eef/stop decision'))
        return None

    def _check(self, name, arguments):
        expected = self.next_call()
        if expected is None or name != expected['tool']:
            raise InputError(f'Wrong order or completed rollout; next call: {expected}')
        for key, value in expected.items():
            if key not in ('tool', 'response') and arguments.get(key) != value:
                raise InputError(f'{key} must equal {value!r}')
        destination = Path(arguments['output_dir'])
        if destination.exists() or not destination.resolve().is_relative_to(self.output):
            raise InputError('Output must be a fresh directory under this rollout root')
        return destination

    def _rpc(self, op, **kwargs):
        with self.timing.span('rpc.'+op, step_id=self.tick):
            return self.sim.request(op, episode_id=self.episode, step_id=self.tick, **kwargs)

    def _observation(self, directory, *, result=None):
        directory.mkdir(parents=True)
        self.obs = self._rpc('teacher_observation')
        np.savez_compressed(directory/'observation.npz', **{
            k: self.obs[k] for k in self.observation_keys})
        images = []
        for camera in self.camera_keys:
            path = directory/f'{camera}.png'
            Image.fromarray(self.obs[camera]).save(path)
            images.append(dict(camera=camera, path=str(path)))
        self.observation_path = directory/'observation.json'
        packet = dict(episode_id=self.episode, step_id=self.tick, task=self.task,
            max_episode_steps=self.meta['max_episode_steps'],
            instruction=self.obs['instruction'], remaining_steps=self.obs['remaining_steps'],
            require_native_termination=self.require_native_termination,
            current_state=self.obs['states'].tolist(), images=images,
            current_eef=self.current_eef(), robot_profile=self.profile,
            observation_path=str(self.observation_path), arrays_path=str(directory/'observation.npz'),
            history_path=str(self.output/'history.json'),
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters), result=result, rollout_finished=self.phase == 'done',
            next_call=self.next_call())
        self.observation_packet = packet
        from ..prompt_profile import guidance_disabled
        if self.tick == 0 and not guidance_disabled():
            from ..prompt_context import task_context
            packet['task_context'] = task_context(self.task)
        write_json(self.observation_path, packet)
        return packet

    def start(self, **arguments):
        directory = self._check('robodojo_start', arguments)
        self.sim = self.rpc_factory('127.0.0.1', self.sim_port, timeout=600)
        self.meta = self.sim.request('metadata')
        if self.meta['task'] != self.task:
            raise RuntimeError('Launched RoboDojo task differs from requested task')
        from ..regression_eval import enabled as eval_enabled, NATIVE_BUDGET
        if eval_enabled():
            expected_budget = int(os.environ.get('ROLLOUT_EVAL_NATIVE_BUDGET', NATIVE_BUDGET))
            if (self.meta['max_episode_steps'] != expected_budget
                    or self.meta.get('scene_query', {}).get('enabled')
                    or self.meta.get('step_budget', {}).get('step_limit_overridden')):
                raise RuntimeError('Eval simulator has an unexpected budget or privileged scene access')
        identity = self.student.metadata['checkpoint_sha256']
        hold_config = self.meta.get('pi_force_hold', {})
        if hold_config.get('enabled') and hold_config.get('version') in ('contact_margin_v3', 'contact_zero_v4'):
            from .pi_force_hold import validate_tightening_context
            validate_tightening_context(hold_config['config'],
                controller_route=os.environ.get('ROLLOUT_CONTROLLER_ROUTE', 'baseline'),
                checkpoint_sha256=identity)
        config = self.student.metadata['config']
        version = f'OpenPI-JAX/{config}/{identity[:16]}'
        with self.timing.span('rpc.reset'):
            reset = self.sim.request('reset', seed=self.seed, source='student', policy_version=version)
        if reset.get('instruction'):
            self.meta['instruction'] = reset['instruction']
        self.episode, self.tick = reset['episode_id'], reset['step_id']
        if 'pi_force_hold' in reset.get('metadata', {}):
            self.meta['pi_force_hold'] = reset['metadata']['pi_force_hold']
        self.force_hold_enabled = bool(self.meta.get('pi_force_hold', {}).get('enabled', False))
        from ..downsampling import configuration, HybridDownsampling
        downsample_config = configuration()
        if downsample_config is not None and self.force_hold_enabled:
            raise RuntimeError('Downsampling cannot execute with simulator pi_force_hold enabled')
        self.downsampling = (HybridDownsampling(downsample_config, self._rpc)
                             if downsample_config is not None else None)
        self.force_hold_intervened = False
        if self.force_hold_enabled:
            self.counters.update(student_proposal_steps=0, force_hold_steps=0)
            if self.meta['pi_force_hold'].get('version') in ('contact_tighten_v2', 'contact_margin_v3', 'contact_zero_v4'):
                self.counters['tighten_steps'] = 0
        # A frozen benchmark case must use its native horizon, not a model's
        # judgement that further attempts are unlikely to help. Operator and
        # infrastructure aborts still use finish() and remain incomplete.
        self.require_native_termination = bool(reset.get('metadata', {}).get('evaluation_case'))
        if eval_enabled():
            case = reset.get('metadata', {}).get('evaluation_case') or {}
            if not case or case.get('policy_rng_seed') != self.student.metadata.get('policy_rng_seed'):
                raise RuntimeError('Eval reset requires frozen case identity and matching policy RNG seed')
        from ..prompt_context import CONTEXT_VERSION
        self._rpc('begin_combination', teacher_model=MODEL, teacher_model_provider=PROVIDER,
                  context_version=CONTEXT_VERSION,
                  prompt_sha256=self.prompt_sha256,
                  student_policy_version=version, student_policy_sha256=identity)
        self.run = dict(schema='robodojo_rollout.run.v1', teacher='codex_tools',
            context_version=CONTEXT_VERSION,
            teacher_model=MODEL, teacher_model_provider=PROVIDER,
            teacher_reasoning_effort=EFFORT,
            task=self.task, instruction=self.meta['instruction'], seed=self.seed,
            evaluation_case=reset.get('metadata', {}).get('evaluation_case'),
            config=config, robot_profile=self.profile, checkpoint=self.student.metadata['checkpoint'],
            action_horizon=self.student.metadata.get('action_horizon', 15),
            action_dim=self.student.metadata.get('action_dim', 8),
            student_backend='OpenPI/JAX', student_identity_sha256=identity,
            student_server_metadata=self.student.metadata, teacher_off=False, split=None,
            gate_policy='failure-or-intent', no_rollback=True,
            gate_instruction_sha256=hashlib.sha256(GATE_INSTRUCTION.encode()).hexdigest(),
            teacher_prompt_sha256=self.prompt_sha256, initial_state_hash=reset.get('initial_state_hash'),
            reset_metadata=reset.get('metadata'), max_episode_steps=self.meta['max_episode_steps'],
            control_dt=self.meta['control_dt'], max_decisions=self.max_decisions,
            require_native_termination=self.require_native_termination,
            training_steps=0, training_bundle_ready=False)
        from ..downsampling import identity as downsampling_identity
        self.run.update(downsampling_identity())
        if self.downsampling is not None:
            self.run.update(force_hold_enabled=False, pi_force_hold=self.meta.get('pi_force_hold'))
        if self.force_hold_enabled:
            self.run['pi_force_hold'] = self.meta['pi_force_hold']
            self.run.update(force_hold_enabled=True, control_class='pi05_gpt_force_assisted')
            self.run['student_steps_semantics'] = 'actual student-source ACKs including force hold; student_proposal_steps excludes hold'
        write_json(self.output/'run.json', self.run)
        write_json(self.output/'history.json', self.history)
        self.phase = 'infer'
        return self._observation(directory)

    def policy_observation(self):
        return self.obs

    def infer(self, **arguments):
        directory = self._check('pi05_infer', arguments)
        directory.mkdir(parents=True)
        decision = len(self.history)
        self.proposal_path = directory/'actions.npz'
        with self.timing.span('pi05_infer_and_save', step_id=self.tick, decision_id=decision):
            self.actions, self.prediction = self.student.infer(self.policy_observation(), self.proposal_path)
        self.timing.emit('pi05_inference_receipt', step_id=self.tick,
                         inference_seconds=self.prediction.get('inference_seconds'))
        # Preserve the downstream record layout without copying arrays.
        (self.output/f'proposal_{decision:03d}.npz').hardlink_to(self.proposal_path)
        self.counters['predictions'] += 1
        self.fk = self._rpc('fk_preview', actions=self.actions)
        self.request = dict(request_id=uuid.uuid4().hex, episode_id=self.episode,
            step_id=self.tick, decision=decision, gate_policy='failure-or-intent',
            gate_instruction=GATE_INSTRUCTION, task=self.task, instruction=self.obs['instruction'],
            sim_time_s=self.tick*self.meta['control_dt'],
            max_episode_steps=self.meta['max_episode_steps'],
            remaining_steps=self.obs['remaining_steps'],
            require_native_termination=self.require_native_termination,
            images=self.observation_packet['images'],
            current_state=self.observation_packet['current_state'],
            current_eef=self.observation_packet['current_eef'],
            student_eef_trajectory=self.trajectory(), robot_profile=self.profile,
            action_diagnostics=action_diagnostics(self.actions),
            fk_check=self.fk['measured_fk_check'],
            previous_observation=previous_observation(self.preceding, self.history[-1]) if self.history else None,
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters), **self.prediction)
        if getattr(self, 'downsampling', None) is not None:
            from ..downsampling import PROMPT
            self.request['student_execution_protocol'] = PROMPT
        write_json(self.output/f'request_{decision:03d}.json', self.request)
        self.phase = 'execute'
        # The unchanged gate prompt is already in the persistent Codex workspace.
        packet = {k: v for k, v in self.request.items() if k != 'gate_instruction'}
        packet.update(proposal_path=str(self.proposal_path), rollout_finished=False,
                      request_path=str(self.output/f'request_{decision:03d}.json'), next_call=self.next_call())
        write_json(directory/'proposal.json', packet)
        return packet

    def execute(self, **arguments):
        directory = self._check('robodojo_execute', arguments)
        response = arguments.get('response')
        try:
            self.validate(response, self.request)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise InputError(str(error)) from error
        if response['mode'] == 'stop' and self.require_native_termination:
            raise InputError(
                'This benchmark requires native termination; model stop is not allowed. '
                f"There are {self.obs['remaining_steps']} native control steps remaining. "
                'No action or stop was executed. Reassess the current observation and '
                'fresh proposal, then submit a student/edit/eef decision under the unchanged '
                'gate. Failed grasp attempts alone are not an episode termination. '
                'Do not reset or fabricate progress; operator/transport/physics aborts '
                'are handled separately by the runner.')
        decision = len(self.history)
        write_json(self.output/f'response_{decision:03d}.json', response)
        # RoboDojo persists every decision (including student/stop), not just
        # source transitions. A failed receipt aborts before physical execution.
        receipt = self._rpc('record_codex_decision', decision=decision,
                            prediction_id=self.prediction['prediction_id'], response=response)
        emit(dict(decision_event(episode_id=self.episode, step_id=self.tick,
            decision=decision, prediction_id=self.prediction['prediction_id'], response=response),
            simulator_receipt=receipt))
        if response['mode'] == 'stop':
            return self._observation(directory, result=self.finish('model_stop'))
        trigger = validate_assessment(response, self.request)
        wanted = 'student' if response['mode'] == 'student' else 'gpt_eef'
        if wanted != self.source:
            self._rpc('switch_control_source', source=wanted, reason=response['reason'])
            self.source = wanted
        trace = []
        self.correction_result = None
        start_tick = self.tick
        sampler = getattr(self, 'downsampling', None)
        selection = None
        selected_indices = None
        if response['mode'] == 'student':
            hold_args = ({'original_action_count': len(self.actions)}
                         if self.force_hold_enabled else {})
            selected_actions = self.actions[:response['steps']]
            if sampler is not None:
                selected_indices, selection = sampler.select(self.actions, response['steps'],
                    self.obs['states'], self.tick, self.output/f'selection_{decision:03d}.json')
                selected_actions = selected_actions[selected_indices]
            result = self._rpc('chunk_step', actions=selected_actions,
                               student_prediction_id=self.prediction['prediction_id'], **hold_args)
            trace = result['steps']
            self.tick = result['step_id']
        else:
            trace = self.execute_correction(response)
        valid = [r for r in trace if r['valid']]
        if sampler is not None:
            sampler.acknowledge(trace, self.obs['states'], self.tick)
        selection_record = {}
        if selection is not None:
            original_indices = [selected_indices[r.get('proposal_action_index', i)]
                                for i, r in enumerate(valid)]
            selection.update(status='executed', actual_ack_count=len(valid),
                actual_ack_original_indices=original_indices, end_tick=self.tick,
                closing_hold_remaining_after_ack=sampler.selector.hold_remaining.tolist())
            write_json(self.output/f'selection_{decision:03d}.json', selection)
            selection_record = dict(action_downsampling=dict(mode='error',
                approved_original_steps=response['steps'], retained_steps=len(selected_indices),
                executed_steps=len(valid), executed_original_indices=original_indices,
                selector_wall_seconds=selection['selector_wall_seconds'],
                fk_wall_seconds=selection['fk_wall_seconds']))
        self.force_hold_intervened |= any(r.get('gripper_override') for r in valid)
        if response['mode'] != 'student':
            write_json(self.output/f'edit_{decision:03d}.json', [dict(
                target=r['edited_target'], diagnostics=r['ik_diagnostics'], valid=r['valid'],
                executed_action=np.asarray(r['executed_action']).tolist()) for r in trace])
        countkey = dict(student='student_steps', edit='edited_steps', eef='recovery_steps')[response['mode']]
        self.counters[countkey] += len(valid)
        proposal_consumed = len(valid)
        hold_record = {}
        if self.force_hold_enabled and response['mode'] == 'student':
            proposal_consumed = result['proposal_steps_consumed']
            hold_steps = result['force_hold_steps']
            if (proposal_consumed + hold_steps != len(valid)
                    or not 0 <= proposal_consumed <= response['steps']):
                raise RuntimeError('Force-hold receipt counts do not match actual ACKs')
            self.counters['student_proposal_steps'] += proposal_consumed
            self.counters['force_hold_steps'] += hold_steps
            self.force_hold_intervened |= bool(result['force_hold_events'])
            hold_record = dict(student_proposal_steps=proposal_consumed, force_hold_steps=hold_steps,
                force_hold_events=result['force_hold_events'],
                reinfer_required=result.get('reinfer_required', False),
                force_hold_state_transitions=result.get('force_hold_state_transitions', []))
            if 'tighten_steps' in self.counters:
                tightening = result['tighten_steps']
                if type(tightening) is not int or not 0 <= tightening <= hold_steps:
                    raise RuntimeError('Tighten count is not a subset of extra ACKs')
                self.counters['tighten_steps'] += tightening
                hold_record['tighten_steps'] = tightening
        executed = np.asarray([r['executed_action'] for r in valid], np.float32).reshape(-1, 14)
        mapping = {}
        if selection is not None:
            mapping = dict(proposal_action_index=np.asarray(original_indices, np.int64),
                native_selected_action_index=np.arange(len(valid), dtype=np.int64),
                selected_original_index=np.asarray(selected_indices, np.int64),
                original_window=self.actions[:response['steps']], initial_measured_state=self.obs['states'])
        if self.force_hold_enabled:
            mapping = {key: np.asarray([r.get(key) if r.get(key) is not None else -1 for r in valid], np.int64)
                       for key in ('proposal_action_index', 'trigger_proposal_action_index')}
            mapping['hold_event_id'] = np.asarray([r.get('hold_event_id') or '' for r in valid], dtype=str)
        np.savez_compressed(self.output/f'execution_{decision:03d}.npz', actions=executed,
                            states=np.asarray([r['obs']['states'] for r in valid], np.float32).reshape(-1, 14),
                            **mapping)
        done = bool(valid and (valid[-1]['terminated'] or valid[-1]['truncated']))
        record = dict(decision=decision, start_tick=start_tick, end_tick=self.tick, response=response,
            simulator_decision_receipt=receipt,
            executed_steps=len(valid), source=self.source, source_detail=response['mode'],
            prediction_id=self.prediction['prediction_id'],
            selected_student_steps=response['steps'] if response['mode'] == 'student' else 0,
            discarded_student_steps=len(self.actions)-proposal_consumed if response['mode'] == 'student' else len(self.actions),
            **hold_record, **selection_record,
            executed_gripper_closed=self.gripper_history(executed),
            assessment=response['assessment'], takeover_trigger=trigger, terminal=done,
            native_success=bool(valid and valid[-1]['success']))
        if self.correction_result is not None:
            record['correction_result'] = self.correction_result
        self.history.append(record)
        self.preceding = self.request
        write_json(self.output/'history.json', self.history)
        write_json(self.output/'progress.json', dict(episode_id=self.episode, step_id=self.tick, **self.counters))
        self.phase = 'infer'
        budget_exhausted = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        final = self.finish('terminal' if done else 'decision_budget') if done or budget_exhausted else None
        return self._observation(directory, result=final)

    def finish(self, reason):
        final = self._rpc('finish_pilot', reason=reason)
        self.phase = 'done'
        from ..prompt_context import CONTEXT_VERSION
        final.update(self.counters, decisions=len(self.history), context_version=CONTEXT_VERSION,
                     complete=bool(final['terminated'] or final['truncated']),
                     wall_seconds=time.monotonic()-self.started, training_bundle_ready=False,
                     trajectory_path=str(self.output.parent/'sim'),
                     history_path=str(self.output/'history.json'),
                     debug_video_path=str(self.recorder.output/'debug_rollout.mp4'),
                     debug_video_manifest=str(self.recorder.output/'manifest.json'))
        if getattr(self, 'force_hold_enabled', False):
            final.update(pi_force_hold=self.meta['pi_force_hold'],force_hold_enabled=True,
                         force_hold_intervened=self.force_hold_intervened,
                         control_class='pi05_gpt_force_assisted')
        if getattr(self, 'downsampling', None) is not None:
            receipts = [r['action_downsampling'] for r in self.history if r.get('action_downsampling')]
            approved = sum(r['approved_original_steps'] for r in receipts)
            retained = sum(r['retained_steps'] for r in receipts)
            final.update(action_downsampling=self.run['action_downsampling'], force_hold_enabled=False,
                downsampling_summary=dict(approved_original_steps=approved, selected_steps=retained,
                    actual_executed_steps=sum(r['executed_steps'] for r in receipts),
                    selected_removed_fraction=1-retained/approved if approved else 0,
                    selector_wall_seconds=sum(r['selector_wall_seconds'] for r in receipts),
                    fk_wall_seconds=sum(r['fk_wall_seconds'] for r in receipts)))
        write_json(self.output/'result.json', final)
        return final

    def close(self):
        if self.sim is not None:
            self.sim.close()
        # finish_pilot has finalized sensors.mp4; this runs after the Codex turn
        # so its public final message is available too. No extra physics steps.
        if self.episode is not None:
            self.recorder.finalize()
