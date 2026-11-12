"""Two rollout tools with controller-owned inference for a persistent Codex agent."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from ..io import InputError, write_json
from ..prompt_context import CONTEXT_VERSION
from ..task_success_conditions import append_task_success_conditions
from ..settings import MODEL, EFFORT, PROVIDER, DEFAULT_CODEX_IMAGE_MAX_EDGE
from ..robodojo_server.client import RoboDojoTools
from ..method import evaluation_method
from ..skill.schema import _obj, response_schema, direct_response_schema
from ..skill.transport import StdioAppServer
from .workspace import prepare_workspace
from .cycle import dispatch
from .input_view import INPUT_VIEW_VERSION, teacher_view
from .output_contract import OUTPUT_CONTRACT_VERSION, compact_response_schema, expand_arguments, next_call
from ..skill.image_preview import image_max_edge, prepare_image
from ..skill.network_recovery import NETWORK_CONTINUE_DELAYS, closed_network_turn, is_network_error, probe_failed_turn
from .model_recovery import eligible as recoverable_model_error, DELAYS as MODEL_RETRY_DELAYS, ModelRecoveryExhausted

SKILL_ROOT = Path(__file__).parent
CONTROLLER_VERSION = 'auto_infer_compact_output_v1'
PROMPT_VERSION = 'visual_task_guidance_push_release_v2'
# Original failure + four retries = five consecutive failures, then stop.
CAPACITY_RETRY_DELAYS = (30, 60, 120, 240)
NATIVE_WORK_ITEMS = frozenset(('commandExecution', 'fileChange', 'imageView',
    'webSearch', 'mcpToolCall', 'collabAgentToolCall'))


def rejected_input(error, rollout, arguments, count):
    """Feedback only for InputError's pre-execution rejection contract.

    Never change arguments, clip a target, retry a handler or reset simulation.
    Other exceptions (including uncertain RPC outcomes) remain fatal.
    """
    packet = dict(error=str(error), error_type='recoverable_tool_input',
        no_execution=True, retryable=True, controller_version=CONTROLLER_VERSION,
        rejected_calls=count, step_id=rollout.tick, next_call=next_call(rollout),
        correction='Correct the arguments and retry next_call in this same episode. '
                   'Do not reset or replay already executed actions.')
    if getattr(rollout, 'observation_path', None):
        packet['observation_path'] = str(rollout.observation_path)
    request = getattr(rollout, 'request', None)
    if isinstance(request, dict) and rollout.phase in ('act', 'execute'):
        packet['request_id'] = request.get('request_id')
        packet['current_eef'] = request.get('current_eef')
        # Explain the existing Euclidean bound without changing its validator.
        response = arguments.get('response') if isinstance(arguments, dict) else None
        if ('exceeds 5 cm' in str(error) and isinstance(response, dict)
                and response.get('mode') == 'eef'):
            distances = {}
            for arm in ('left', 'right'):
                try:
                    target = response['target'][arm]['position']
                    current = request['current_eef'][arm]['position']
                    if len(target) != 3 or len(current) != 3:
                        continue
                    distance = math.dist(target, current)
                    if math.isfinite(distance):
                        distances[arm] = distance
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue  # Diagnostic only; the original error is authoritative.
            packet['eef_translation'] = dict(distance_m=distances, max_distance_m=0.05,
                metric='Euclidean distance from the current EEF, not a per-axis limit')
    return packet


def is_capacity_error(error):
    text = json.dumps(error, ensure_ascii=False).lower()
    return 'serveroverloaded' in text or 'selected model is at capacity' in text


def toml_value(value):
    """Codex -c values are TOML; JSON objects/quoted dotted keys are not equivalent."""
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + toml_value(v) for k, v in value.items()) + '}'
    return json.dumps(value)


def agent_config(audit, agent):
    # Keep Codex's normal tool/skill configuration. These two capabilities are
    # essential, even if an inherited profile had disabled them previously.
    from .memory import compact_limit
    from ..regression_eval import enabled as eval_enabled
    return dict(model=MODEL, model_provider=PROVIDER, model_reasoning_effort=EFFORT,
        model_auto_compact_token_limit=compact_limit(os.environ),
        sqlite_home=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_db'),
        log_dir=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_logs'),
        default_permissions='rollout_agent', **{
            # Eval receives RGB via tool attachments and history via a dedicated
            # read-only tool. Native file tools could bypass the score filter by
            # opening host audit files or the simulator's mounted scene assets.
            'features.shell_tool': not eval_enabled(), 'features.view_image': not eval_enabled(),
            'shell_environment_policy.inherit': 'all',
            'permissions.rollout_agent.extends': ':workspace',
            # The parent contains host-owned RPC logs and experiment artifacts.
            # Only the agent directory is a writable runtime workspace root.
            'permissions.rollout_agent.filesystem': ({str(agent): 'write'} if eval_enabled() else {
                str(audit.parent): 'read', str(agent): 'write'}),
        })


def tool_specs(method='pi05_plus_gpt'):
    if method != 'pi05_plus_gpt':
        raise ValueError('auto_infer_v1 supports pi05_plus_gpt only')
    string = {'type': 'string'}
    def spec(name, description, **properties):
        return dict(type='function', name=name, description=description,
                    inputSchema=_obj(properties))
    if method == 'gpt_only':
        return [spec('robodojo_start', 'Blocking: start the requested episode and return RGB/proprio.',
                     task=string, output_dir=string),
                spec('robodojo_act', 'Blocking: execute GPT-authored bounded dual-arm EEF targets '
                     '(1-5 steps), then return RGB/proprio and native outcome. '
                     'No pi05 inference, proposals or direct joint-action commands exist.',
                     observation_path=string, response=direct_response_schema(), output_dir=string)]
    return [
        spec('robodojo_start', 'Blocking: start once, then automatically infer pi05. Return current RGB/proprio, task context, and a fresh H50 proposal with robot-only FK and action_diagnostics. Review execution evidence and proposal intent before executing.',
             task=string),
        spec('robodojo_execute', 'Blocking: execute a reviewed fresh chunk, then automatically infer unless terminal. Use the latest decision_id in response. Paths and request identity are host-managed. Supply edit only in edit mode, target only in eef mode, neither in student/stop. Assessment and reason remain mandatory; diagnostics are not a gate verdict.',
             response=compact_response_schema()),
    ]


def content_items(packet, *, images, max_image_edge=DEFAULT_CODEX_IMAGE_MAX_EDGE,
                  task_references=()):
    # This boundary is downstream of observation capture and policy inference.
    # Copy metadata only: never change packet, source PNG/NPZ or policy arrays.
    from ..regression_eval import enabled as eval_enabled, teacher_packet
    if eval_enabled():
        packet = teacher_packet(packet)
    visible_packet = teacher_view(packet)
    attachments = []
    if images:
        visible_packet['images'] = []
        for item in packet.get('images', []):
            descriptor, payload = prepare_image(item, max_image_edge)
            visible_packet['images'].append(descriptor)
            data = base64.b64encode(payload).decode('ascii')
            attachments.append(dict(type='inputImage', imageUrl='data:image/png;base64,' + data))
        from ..task_references import append_reference_images
        append_reference_images(visible_packet, attachments, task_references, max_image_edge)
    return [dict(type='inputText', text=json.dumps(visible_packet, separators=(',', ':'))), *attachments]


def build_prompt(skill_root, agent_workspace):
    return ((skill_root/'SKILL.md').read_text()
            + '\n\n# Decision gate\n\n' + (skill_root/'gate_prompt.md').read_text()
            + '\n\n' + (skill_root/'context/eef_control.md').read_text()
            + '\n\nOptional reference workspace: ' + str(agent_workspace)
            + '\nAll normal decision instructions are already supplied above.\n')


class CodexPolicy:
    """Persistent GPT policy; CLI startup supplies the current task rules."""

    def __init__(self, workspace, codex, *, timeout=900, transport_factory=StdioAppServer,
                 method='pi05_plus_gpt', scene_query_enabled=False, task=None):
        from ..regression_eval import enabled as eval_enabled
        from .feedforward import enabled as feedforward_enabled
        self.feedforward = feedforward_enabled()
        self.strict_eval = eval_enabled()
        from .. import prompt_profile
        self.prompt_profile = prompt_profile.profile()
        prompt_metadata = prompt_profile.identity(os.environ)
        from ..placement_order import configuration as order_configuration
        placement_order = order_configuration(task=task)
        if placement_order is not None:
            prompt_metadata['placement_order'] = placement_order
        if self.strict_eval:
            from ..regression_eval import required_env
            for name, value in required_env(os.environ.get('ROLLOUT_CONTROLLER_ROUTE')).items():
                if os.environ.get(name) != value:
                    raise ValueError(f'Eval controller requires {name}={value}')
            if scene_query_enabled:
                raise ValueError('Scene query is unavailable in eval')
        self.scene_query_enabled = scene_query_enabled
        from .abandon import enabled as abandon_enabled
        self.abandon_enabled = abandon_enabled(os.environ)
        self.motion_enabled = os.environ.get('ROLLOUT_ENABLE_MOTION_TOOLS', '0') == '1'
        self.method = evaluation_method(method)
        if self.method != 'pi05_plus_gpt':
            raise ValueError('auto_infer_v1 supports pi05_plus_gpt only')
        direct = self.method == 'gpt_only'
        skill_root = SKILL_ROOT.parent/'robodojo-gpt-only-rollout' if direct else SKILL_ROOT
        self.image_max_edge = image_max_edge(os.environ.get(
            'CODEX_IMAGE_MAX_EDGE', str(DEFAULT_CODEX_IMAGE_MAX_EDGE)))
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=False)
        from ..task_references import prepare_task_references
        self.task_references = prepare_task_references(task, self.workspace)
        write_json(self.workspace.parent/'controller_protocol.json', dict(
            scene_query_version='role_query_v2' if scene_query_enabled else None,
            motion_executor='link6_relaxed_quintic_ack_soft_boundary_v1' if self.motion_enabled else None,
            output_contract_version=OUTPUT_CONTRACT_VERSION,
            prompt_version=PROMPT_VERSION,
            input_view_version=INPUT_VIEW_VERSION,
            controller_version=CONTROLLER_VERSION, context_version=CONTEXT_VERSION,
            recoverable_input_error_limit=None, validation_unchanged=True,
            same_thread=True, simulator_reset_on_input_error=False,
            action_replay_on_input_error=False, **prompt_metadata))
        self.agent_workspace = prepare_workspace(self.workspace, skill_root, method=self.method)
        if self.feedforward:
            from .feedforward import ROUTE, DECISION_POLICY_VERSION
            for path in (self.workspace.parent/'controller_protocol.json', self.agent_workspace/'workspace.json'):
                if path.exists():
                    metadata = json.loads(path.read_text())
                    metadata.update(controller_route=ROUTE, conditioning_version=ROUTE,
                                    decision_policy_version=DECISION_POLICY_VERSION)
                    write_json(path, metadata)
        self.timeout = timeout
        skill = (skill_root/'SKILL.md').read_text()
        gate = '' if direct else (skill_root/'gate_prompt.md').read_text()
        self.prompt = build_prompt(skill_root, self.agent_workspace)
        if scene_query_enabled:
            from .role_query import allow_query, PROMPT
            self.prompt = allow_query(self.prompt) + PROMPT
            skill, gate = allow_query(skill), allow_query(gate)
            # Keep discoverable workspace references consistent with injected instructions.
            for path in self.agent_workspace.rglob('*.md'):
                path.write_text(allow_query(path.read_text()))
            (self.agent_workspace/'AGENTS.md').write_text(
                (self.agent_workspace/'AGENTS.md').read_text() + PROMPT)
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        if self.motion_enabled:
            from .motion import PROMPT as motion_prompt, allow_motion
            self.prompt = allow_motion(self.prompt) + motion_prompt
            skill = allow_motion(skill)
            self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
            for path in self.agent_workspace.rglob('*.md'):
                path.write_text(allow_motion(path.read_text()))
            with (self.agent_workspace/'AGENTS.md').open('a') as stream:
                stream.write(motion_prompt)
        if task is not None and not prompt_profile.guidance_disabled():
            self.prompt = append_task_success_conditions(
                self.prompt, task, output_dir=self.workspace)
            from .task_memory import inject
            self.prompt = inject(self.prompt, self.workspace, task, os.environ)
        if self.abandon_enabled:
            from .abandon import PROMPT as abandon_prompt
            self.prompt += abandon_prompt
            with (self.agent_workspace/'AGENTS.md').open('a') as stream:
                stream.write(abandon_prompt)
        if self.feedforward:
            from .feedforward import adapt_prompt
            self.prompt, skill, gate = map(adapt_prompt, (self.prompt, skill, gate))
            for path in self.agent_workspace.rglob('*.md'):
                path.write_text(adapt_prompt(path.read_text()))
        if self.strict_eval:
            from .eval_prompt import adapt, WORKSPACE
            self.prompt, skill, gate = map(adapt, (self.prompt, skill, gate))
            for path in self.agent_workspace.rglob('*.md'):
                path.write_text(adapt(path.read_text()))
            (self.agent_workspace/'AGENTS.md').write_text(WORKSPACE)
        if prompt_profile.guidance_disabled():
            self.prompt, skill, gate = map(prompt_profile.adapt, (self.prompt, skill, gate))
            for path in self.agent_workspace.rglob('*.md'):
                path.write_text(prompt_profile.adapt(path.read_text()))
        from ..downsampling import configuration as downsampling_config, PROMPT as downsampling_prompt
        if downsampling_config() is not None:
            self.prompt += downsampling_prompt
            skill += downsampling_prompt
        if placement_order is not None:
            from ..placement_order import render as render_order
            order_prompt = render_order(placement_order)
            # A task constraint in BOTH profiles, outside the optional guidance filter.
            # Developer instructions retain it across turns/compaction; no VLA rewrite.
            self.prompt += order_prompt
            with (self.agent_workspace/'AGENTS.md').open('a') as stream:
                stream.write(order_prompt)
            write_json(self.workspace/'placement_order.json', dict(
                **placement_order, instruction=order_prompt))
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        (self.workspace/'SKILL.md').write_text(skill)
        if not direct:
            (self.workspace/'gate_prompt.md').write_text(gate)
        (self.workspace/'PROMPT.md').write_text(self.prompt)
        specs = tool_specs(self.method)
        if self.strict_eval:
            from ..regression_eval import history_spec
            specs.append(history_spec())
        if self.abandon_enabled:
            from .abandon import tool_spec as abandon_spec
            specs.append(abandon_spec())
        if scene_query_enabled:
            from .role_query import tool_spec
            specs.append(tool_spec())
        if self.motion_enabled:
            from .motion import tool_specs as motion_specs
            specs.extend(motion_specs())
        if self.feedforward:
            from .feedforward import adapt_specs
            specs = adapt_specs(specs)
        write_json(self.workspace/'tools.json', specs)
        version = subprocess.run([codex, '--version'], capture_output=True, text=True, check=True).stdout.strip()
        config = agent_config(self.workspace, self.agent_workspace)
        argv = [codex, 'app-server', '--stdio', '--strict-config']
        for key, value in config.items():
            argv += ['-c', key + '=' + toml_value(value)]
        write_json(self.workspace/'launch.json', dict(argv=argv, config=config))
        print(json.dumps(dict(event='context_policy', history='memory/decisions.md',
            model_auto_compact_token_limit=config['model_auto_compact_token_limit'])), flush=True)
        self.transport = transport_factory(argv, self.workspace)
        try:
            self.transport.request('initialize', dict(
                clientInfo=dict(name='robodojo-hybrid-policy', version='1'),
                capabilities=dict(experimentalApi=True)), timeout)
            self.transport.notify('initialized', {})
            response = self.transport.request('thread/start', dict(
                cwd=str(self.agent_workspace), model=MODEL, modelProvider=PROVIDER,
                config=dict(model_reasoning_effort=EFFORT), developerInstructions=self.prompt,
                dynamicTools=specs, ephemeral=self.strict_eval,
                allowProviderModelFallback=False, approvalPolicy='never',
                approvalsReviewer='auto_review', permissions='rollout_agent',
                runtimeWorkspaceRoots=[str(self.agent_workspace)]), timeout)
            if response.get('model') != MODEL or response.get('reasoningEffort') != EFFORT:
                raise RuntimeError(f'Codex model/effort differs from the required {MODEL}/{EFFORT}: '
                                   f'{response.get("model")}/{response.get("reasoningEffort")}')
            self.thread_id = response['thread']['id']
            print(json.dumps(dict(event='teacher_model_verified', model=MODEL,
                reasoning_effort=EFFORT, provider=PROVIDER, thread_id=self.thread_id)), flush=True)
            write_json(self.workspace/'worker.json', dict(model=MODEL, model_provider=PROVIDER,
                prompt_version=PROMPT_VERSION,
                input_view_version=INPUT_VIEW_VERSION, output_contract_version=OUTPUT_CONTRACT_VERSION,
                controller_version=CONTROLLER_VERSION,
                evaluation_method=self.method,
                scene_query_enabled=scene_query_enabled,
                motion_tools_enabled=self.motion_enabled,
                scene_query_version='role_query_v2' if scene_query_enabled else None,
                reasoning_effort=EFFORT, context_version=CONTEXT_VERSION,
                codex_version=version, thread_id=self.thread_id, pid=self.transport.process.pid,
                prompt_sha256=self.prompt_sha256, no_rollback=True, tools=[s['name'] for s in specs],
                service_tools_are_additive=True, agent_workspace=str(self.agent_workspace),
                native_tools=('disabled shell/image file access; RGB attachments and same-episode history tool'
                              if self.strict_eval else 'Codex defaults; shell and image viewing explicitly enabled'),
                network_recovery=dict(revision='same_thread_model_continue_v2',
                    backoff_seconds=list(MODEL_RETRY_DELAYS), restart_simulator=False,
                    new_thread=False, replay_actions=False),
                capacity_recovery=dict(max_retries=10,
                    backoff_seconds=list(MODEL_RETRY_DELAYS),
                    terminal_confirmation_required=True, replay_actions=False),
                codex_image_max_edge=self.image_max_edge, policy_images_resized=False,
                baseline_full_conversation_available=False))
        except BaseException:
            self.close()
            raise

    def _turn(self, text):
        result = self.transport.request('turn/start', dict(threadId=self.thread_id,
            model=MODEL, effort=EFFORT, approvalPolicy='never', approvalsReviewer='auto_review',
            permissions='rollout_agent', cwd=str(self.agent_workspace),
            runtimeWorkspaceRoots=[str(self.agent_workspace)], input=[dict(type='text', text=text)]), self.timeout)
        return result['turn']['id']

    def _network_continue(self, rollout, old_turn, error, consecutive, serial):
        if not is_network_error(error) or consecutive >= len(NETWORK_CONTINUE_DELAYS):
            raise RuntimeError(f'Codex network continue exhausted or ineligible: {error}')
        return self._continue_failed_turn(rollout, old_turn, error, consecutive, serial,
                                          'network', NETWORK_CONTINUE_DELAYS[consecutive])

    def _capacity_continue(self, rollout, old_turn, error, consecutive, serial):
        if not is_capacity_error(error) or consecutive >= len(CAPACITY_RETRY_DELAYS):
            raise RuntimeError(f'Codex capacity retry exhausted after five consecutive failures: {error}')
        return self._continue_failed_turn(rollout, old_turn, error, consecutive, serial,
                                          'capacity', CAPACITY_RETRY_DELAYS[consecutive])

    def _continue_failed_turn(self, rollout, old_turn, error, consecutive, serial, kind, delay):
        # Called only after terminal failure/confirmed idle probe. Successful
        # service ACKs already advanced rollout.next_call(); never replay them.
        record = dict(revision='same_thread_continue_v1', thread_id=self.thread_id,
            previous_turn_id=old_turn, error=error, step_id=rollout.tick,
            phase=rollout.phase, next_call=next_call(rollout), delay_seconds=delay,
            consecutive_attempt=consecutive+1, status='waiting', physical_actions_replayed=0)
        path = self.workspace/f'{kind}_continue_{serial:04d}.json'
        write_json(path, record)
        print(json.dumps(dict(event=f'codex_{kind}_continue_wait', **record)), flush=True)
        # Keep Python/the simulator/observation alive. SIGTERM still exits normally.
        from contextlib import nullcontext
        timing = getattr(self, 'timing', None)
        with timing.span('retry_wait.'+kind) if timing else nullcontext():
            for _ in range(delay):
                time.sleep(1)
        text = (f'Continue the same rollout. The previous turn ended because of a {kind} error. '
                'The simulator and recorded actions are unchanged; do not reset or repeat executed actions. '
                f'Current step_id={rollout.tick}. Use the latest recorded observation and next_call: '
                + json.dumps(next_call(rollout)))
        if getattr(rollout, 'observation_path', None):
            text += '\nLatest observation: ' + str(rollout.observation_path)
        record['status'] = 'turn_start_requested'
        write_json(path, record)  # Ambiguous RPC must never be reissued blindly.
        new_turn = self._turn(text)
        record.update(status='continue_sent', new_turn_id=new_turn)
        write_json(path, record)
        print(json.dumps(dict(event=f'codex_{kind}_continue_sent', thread_id=self.thread_id,
            old_turn_id=old_turn, turn_id=new_turn, step_id=rollout.tick)), flush=True)
        return new_turn

    def run(self, rollout):
        from ..timing import Timing
        self.timing = Timing(self.workspace.parent.parent/'timing/controller.jsonl')
        with self.timing.span('controller_loop'):
            try:
                return self._run_measured(rollout)
            finally:
                # Do not close the enclosing loop span here; its context owns it.
                for token in list(self.timing.open.values()):
                    if token[1] != 'controller_loop':
                        self.timing.end(token, status='interrupted')

    def _run_measured(self, rollout):
        teacher_span = self.timing.begin('teacher_to_start', step_id=rollout.tick)
        native_spans = {}
        turn_id = self._turn('Make online control decisions for this single episode under '
            'the injected skill, gate and robot contract. All normal instructions are '
            'already supplied; no startup file-reading or notes checklist is required. '
            'Use service replies as the decision input and preserve the required '
            'assessment and reason. Use English for public output. '
            'The user authorizes sending this episode\'s three RGB images, proprio, robot-only FK, '
            'task text and same-episode history to OpenAI Codex for online decisions. '
            + ('Recorded gated scene truth and numbered images are also authorized for recovery. '
               if getattr(self, 'scene_query_enabled', False) else '')
            + 'First call: ' + json.dumps(next_call(rollout)))
        call_index = 0
        errors = 0
        continuations = 0
        network_error = None
        network_consecutive = network_serial = 0
        capacity_attempts = 0
        capacity_serial = 0
        capacity_error = None
        model_error = None
        model_attempts = model_serial = 0
        completed_turns = 0
        token_limit = int(os.environ.get('CODEX_MAX_TOTAL_TOKENS', '0'))
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError('Codex made no completed service or native tool call within the configured timeout')
                event = self.transport.next_message(remaining)
            except TimeoutError:
                if not network_error and not capacity_error and not model_error:
                    raise
                # A missing terminal notification does not authorize a second active turn.
                turn = probe_failed_turn(self.transport, self.thread_id, turn_id, min(30, self.timeout),
                    recoverable_model_error if model_error else is_capacity_error if capacity_error else is_network_error,
                    ephemeral=getattr(self,'strict_eval',False),
                    network_failure=bool(network_error and not capacity_error and not model_error))
                if turn is None:
                    raise
                event = dict(method='turn/completed', params=dict(threadId=self.thread_id, turn=turn))
            method, params = event.get('method'), event.get('params', {})
            if (method == 'model/rerouted' and params.get('threadId') == self.thread_id
                    and params.get('toModel') != MODEL):
                write_json(self.workspace/'model_identity_error.json', params)
                raise RuntimeError(f'Model rerouted away from pinned {MODEL}: {params}')
            if method == 'thread/tokenUsage/updated' and params.get('threadId') == self.thread_id:
                usage = params.get('tokenUsage', {})
                write_json(self.workspace.parent/'token_usage.json', dict(
                    model=MODEL, effort=EFFORT, provider=PROVIDER, usage=usage,
                    total_tokens_include_cached_input=True, configured_limit=token_limit))
                print(json.dumps(dict(event='token_usage', total=usage.get('total', {}))), flush=True)
                if token_limit and usage.get('total', {}).get('totalTokens', 0) >= token_limit:
                    raise RuntimeError('Configured total-token budget reached; stopping this episode')
            elif method == 'item/tool/call':
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    raise RuntimeError('Tool call belongs to another thread or turn')
                name, arguments = params['tool'], params['arguments']
                visible_args = arguments
                if isinstance(visible_args, str):
                    try:
                        visible_args = json.loads(visible_args)
                    except ValueError:
                        visible_args = {}
                response = visible_args.get('response', {}) if isinstance(visible_args, dict) else {}
                mode = response.get('mode') if isinstance(response, dict) else None
                if name in ('robodojo_move', 'robodojo_grasp'):
                    mode = 'eef'
                if teacher_span is not None:
                    self.timing.end(teacher_span, next_tool=name, mode=mode)
                    teacher_span = None
                print(json.dumps(dict(event='tool_start', tool=name, call=call_index,
                                      phase=rollout.phase, step_id=rollout.tick)), flush=True)
                audit = dict(call_id=params['callId'], tool=name, arguments=arguments)
                write_json(self.workspace/f'call_{call_index:04d}_request.json', audit)
                try:
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as error:
                            raise InputError(f'Tool arguments must be a JSON object: {error}') from error
                    motion_call = name in ('robodojo_move', 'robodojo_grasp') and getattr(self, 'motion_enabled', False)
                    abandon_call = name == 'robodojo_abandon' and getattr(self, 'abandon_enabled', False)
                    history_call = name == 'robodojo_history' and getattr(self, 'strict_eval', False)
                    infer_call = name == 'pi05_infer' and getattr(self, 'feedforward', False)
                    if name not in ('robodojo_start', 'robodojo_execute') and not infer_call and not motion_call and not abandon_call and not history_call and not (
                            name == 'robodojo_query' and getattr(self, 'scene_query_enabled', False)):
                        raise InputError('Unknown host service; native Codex tools are executed by app-server')
                    if not isinstance(arguments, dict):
                        raise InputError('Tool arguments must be a JSON object')
                    expanded = (arguments if name == 'robodojo_query' or motion_call or abandon_call or history_call or infer_call else
                                expand_arguments(rollout, name, arguments))
                    write_json(self.workspace/f'call_{call_index:04d}_expanded_request.json',
                               dict(call_id=params['callId'], tool=name, arguments=expanded))
                    if history_call:
                        from ..regression_eval import history_packet
                        packet = history_packet(getattr(rollout, 'history', []), expanded)
                    elif abandon_call:
                        from .abandon import execute as abandon_execute
                        with self.timing.span('service.robodojo_abandon', step_id=rollout.tick, call_id=call_index):
                            packet = abandon_execute(rollout, expanded)
                    elif motion_call:
                        packet = rollout.execute_motion(name, expanded, self.timing, call_index)
                    elif name == 'robodojo_query':
                        with self.timing.span('service.robodojo_query', step_id=rollout.tick, call_id=call_index):
                            packet = rollout.query_scene(expanded)
                    else:
                        packet = dispatch(rollout, name, expanded, self.timing, call_index)
                except InputError as error:
                    errors += 1
                    packet = rejected_input(error, rollout, arguments, errors)
                    success = False
                else:
                    success = True
                    # Reset only network wakeups; rejected-call counts remain audit-only.
                    network_consecutive = 0
                    network_error = None
                    capacity_attempts = 0
                    capacity_error = None
                    model_error = None
                    model_attempts = 0
                finally:
                    # Also persist executed decisions when automatic infer or
                    # transport fails afterwards. Never replay the action.
                    from .memory import save_history
                    save_history(self.workspace.parent, getattr(rollout, 'history', []))
                # Persist every rejection, including the fifth and later, before
                # replying. A failed transport reply must not repeat a handler.
                if getattr(self, 'abandon_enabled', False):
                    from .abandon import gate as abandon_gate
                    packet['abandon_gate'] = abandon_gate(getattr(rollout, 'history', []))
                write_json(self.workspace/f'call_{call_index:04d}_result.json', packet)
                with self.timing.span('tool_result_prepare', call_id=call_index):
                    items = content_items(packet, images=success,
                                          max_image_edge=self.image_max_edge,
                                          task_references=getattr(self, 'task_references', ()))
                with self.timing.span('tool_result_send', call_id=call_index):
                    self.transport.reply(event['id'], dict(success=success, contentItems=items))
                teacher_span = self.timing.begin('teacher_to_'+rollout.phase,
                    step_id=rollout.tick, previous_tool=name, call_id=call_index)
                call_index += 1
                print(json.dumps(dict(event='tool_done', tool=name, step_id=rollout.tick,
                    phase=rollout.phase, counters=rollout.counters)), flush=True)
                deadline = time.monotonic() + self.timeout
            elif method in ('item/started', 'item/completed'):
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    continue
                item = params.get('item', {})
                kind = item.get('type')
                if kind in NATIVE_WORK_ITEMS:
                    key = (turn_id, item.get('id'))
                    if method == 'item/started' and key not in native_spans:
                        native_spans[key] = self.timing.begin('native.'+kind, step_id=rollout.tick,
                                                             item_id=item.get('id'))
                    elif method == 'item/completed':
                        token = native_spans.pop(key, None)
                        if token is not None:
                            self.timing.end(token, item_status=item.get('status'),
                                            exit_code=item.get('exitCode'))
                        else:
                            self.timing.emit('unmatched_native_completion', item_type=kind,
                                             item_id=item.get('id'))
                if kind in NATIVE_WORK_ITEMS or (kind == 'agentMessage' and method == 'item/completed'):
                    # Public outputs only: never copy private reasoning items.
                    with (self.workspace/'agent_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(method=method, **params), ensure_ascii=False)+'\n')
                    event_record = dict(event='agent_activity', phase=method.split('/')[-1],
                        item_type=kind, item_id=item.get('id'), step_id=rollout.tick,
                        status=item.get('status'))
                    if kind == 'agentMessage':
                        event_record['text'] = item.get('text', '')
                    elif kind == 'commandExecution':
                        event_record['command'] = item.get('command')
                        event_record['exit_code'] = item.get('exitCode')
                    print(json.dumps(event_record, ensure_ascii=False), flush=True)
                    if kind in NATIVE_WORK_ITEMS:
                        # Real analysis is progress, not a missing rollout decision.
                        deadline = time.monotonic() + self.timeout
            elif method == 'turn/completed' and params.get('threadId') == self.thread_id:
                turn = params.get('turn', {})
                if turn.get('id') != turn_id:
                    continue
                if teacher_span is not None and (rollout.phase == 'done' or turn.get('status') == 'completed'):
                    self.timing.end(teacher_span, status='ok' if turn.get('status') == 'completed' else 'error', terminal=True)
                    teacher_span = None
                write_json(self.workspace/f'turn_{completed_turns:02d}_completed.json', turn)
                completed_turns += 1
                failure = turn.get('error') or model_error or capacity_error or network_error
                if turn.get('status') == 'failed' and recoverable_model_error(failure):
                    if rollout.phase == 'done':
                        return
                    if model_attempts >= len(MODEL_RETRY_DELAYS):
                        raise ModelRecoveryExhausted('Ten same-thread model recovery attempts exhausted')
                    turn_id = self._continue_failed_turn(rollout, turn_id, failure,
                        model_attempts, model_serial, 'model', MODEL_RETRY_DELAYS[model_attempts])
                    model_attempts += 1
                    model_serial += 1
                    model_error = capacity_error = network_error = None
                    deadline = time.monotonic() + self.timeout
                    continue
                if turn.get('status') == 'failed' and is_network_error(failure) and recoverable_model_error(failure):
                    if rollout.phase == 'done':
                        return  # Native completion needs no further paid model response.
                    turn_id = self._network_continue(rollout, turn_id, failure,
                        network_consecutive, network_serial)
                    network_serial += 1
                    network_consecutive += 1
                    network_error = None
                    deadline = time.monotonic() + self.timeout
                    continue
                if turn.get('status') == 'failed' and is_capacity_error(failure) and recoverable_model_error(failure):
                    if rollout.phase == 'done':
                        return
                    turn_id = self._capacity_continue(rollout, turn_id, failure,
                        capacity_attempts, capacity_serial)
                    capacity_attempts += 1
                    capacity_serial += 1
                    capacity_error = None
                    deadline = time.monotonic() + self.timeout
                    continue
                if turn.get('status') != 'completed':
                    raise RuntimeError(f'Codex turn ended: {turn.get("status")}')
                if rollout.phase == 'done':
                    return
                if continuations >= 2:
                    raise RuntimeError('Codex ended repeatedly before completing the episode')
                # Context-only continuation, never a human action judgement.
                continuations += 1
                teacher_span = self.timing.begin('teacher_to_'+rollout.phase, step_id=rollout.tick)
                turn_id = self._turn('The same episode is unfinished. Continue the skill; next call: ' +
                                     json.dumps(next_call(rollout)))
                deadline = time.monotonic() + self.timeout
            elif method in ('error', 'turn/failed'):
                if (params.get('threadId') not in (None, self.thread_id)
                        or params.get('turnId') not in (None, turn_id)):
                    continue
                error_value = params.get('error', params)
                if is_network_error(error_value):
                    network_error = error_value
                if method == 'error' and params.get('willRetry') is True:
                    # App-server owns model transport retry; this does not
                    # repeat inference, a tool handler, or physical execution.
                    print(json.dumps(dict(event='codex_transport_retry',
                        error=params.get('error'), step_id=rollout.tick)), flush=True)
                    continue
                if recoverable_model_error(error_value):
                    if rollout.phase == 'done':
                        return
                    model_error = error_value
                    deadline = min(deadline, time.monotonic() + 30)
                    print(json.dumps(dict(event='codex_model_wait_terminal',
                        error=error_value, step_id=rollout.tick)), flush=True)
                    continue
                if is_capacity_error(error_value) and recoverable_model_error(error_value):
                    if rollout.phase == 'done':
                        return
                    capacity_error = error_value
                    deadline = min(deadline, time.monotonic() + 30)
                    print(json.dumps(dict(event='codex_capacity_wait_terminal',
                        error=error_value, step_id=rollout.tick)), flush=True)
                    continue
                if network_error and is_network_error(error_value) and recoverable_model_error(error_value):
                    if rollout.phase == 'done':
                        return
                    # Wait for turn/completed, or verify idle/failed via the timeout probe.
                    # Do not send continue while Codex still owns an active turn.
                    deadline = min(deadline, time.monotonic() + 30)
                    print(json.dumps(dict(event='codex_network_wait_terminal',
                        error=network_error, step_id=rollout.tick)), flush=True)
                    continue
                raise RuntimeError(f'Codex error: {params}')
            elif 'id' in event and 'method' in event:
                raise RuntimeError(f'Unexpected Codex capability request: {method}')

    def close(self):
        self.transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--evaluation-method', default='pi05_plus_gpt', choices=('pi05_plus_gpt',))
    parser.add_argument('--codex', required=True)
    parser.add_argument('--sim-port', type=int, default=19113)
    parser.add_argument('--student-port', type=int, default=18830)
    parser.add_argument('--max-decisions', type=int, default=180, help='0 disables the decision-count limit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--enable-scene-query', action='store_true')
    args = parser.parse_args()
    from ..regression_eval import enabled as eval_enabled
    if eval_enabled() and (args.max_decisions != 0 or args.enable_scene_query):
        raise ValueError('Eval requires unlimited decision count and no scene query')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from ..timing import Timing
    startup_timing = Timing(args.output.parent/'timing/controller.jsonl')
    student = worker = rollout = None
    def terminate(signum, frame):
        raise KeyboardInterrupt('Stopping only this owned rollout and Codex process')
    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.evaluation_method == 'pi05_plus_gpt':
            if args.checkpoint is None:
                raise ValueError('Hybrid evaluation requires an explicit checkpoint')
            from ..pi05_server.client import Pi05Client
            with startup_timing.span('controller_connect_pi05'):
                student = Pi05Client(args.student_port, args.checkpoint)
        with startup_timing.span('controller_start_codex'):
            worker = CodexPolicy(args.output/'codex_workspace', args.codex, method=args.evaluation_method,
                                 scene_query_enabled=args.enable_scene_query, task=args.task)
        if args.evaluation_method == 'gpt_only':
            from ..robodojo_server.gpt_only_client import GPTOnlyTools
            rollout = GPTOnlyTools(args.output, args.task, sim_port=args.sim_port,
                seed=args.seed, max_decisions=args.max_decisions, prompt_sha256=worker.prompt_sha256)
        else:
            tools_class = RoboDojoTools
            if args.enable_scene_query:
                from .role_query import RoleQueryTools
                tools_class = RoleQueryTools
            if worker.motion_enabled:
                from .motion import MotionMixin
                tools_class = type('MotionTools', (MotionMixin, tools_class), {})
            if worker.feedforward:
                from .feedforward import FeedforwardMixin
                tools_class = type('FeedforwardTools', (FeedforwardMixin, tools_class), {})
            rollout = tools_class(args.output, args.task, student, sim_port=args.sim_port,
                seed=args.seed, max_decisions=args.max_decisions, prompt_sha256=worker.prompt_sha256)
        worker.run(rollout)
        from .task_memory import enabled, summarize
        if enabled(os.environ):
            with startup_timing.span('episode_summary'):
                summarize(worker, rollout, args.task)
    except BaseException:
        write_json(args.output/'failure.json', dict(error=traceback.format_exc(),
            controller_version=CONTROLLER_VERSION,
            episode_id=rollout.episode if rollout else None, step_id=rollout.tick if rollout else 0,
            counters=rollout.counters if rollout else {}, completed=False))
        if rollout is not None and rollout.episode is not None and rollout.phase != 'done':
            try:
                rollout.finish('controller_error')
            except Exception:
                pass  # Never reconnect or retry an action with an uncertain ACK.
        raise
    finally:
        from .task_memory import enabled
        if enabled(os.environ) and not (args.output/'memory/episode_summary.json').exists():
            (args.output/'memory').mkdir(exist_ok=True)
            write_json(args.output/'memory/episode_summary.json', dict(
                status='skipped_interrupted', reason='Control did not reach a summary turn'))
        if rollout is not None:
            rollout.close()
        if student is not None:
            student.close()
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
