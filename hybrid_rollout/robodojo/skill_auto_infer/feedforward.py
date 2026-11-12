"""Opt-in teacher-first conditioning; legacy auto-inference is unchanged."""
from copy import deepcopy
import os
import hashlib
from pathlib import Path
import uuid

from ..io import InputError, write_json
from .subtask_instruction import online_schema, render_online, format_prompt

ROUTE = 'subtask_feedforward_v1'
DECISION_POLICY_VERSION = 'goal_directed_v2'


def enabled(env=None):
    return (os.environ if env is None else env).get('ROLLOUT_CONTROLLER_ROUTE') == ROUTE


def adapt_prompt(text):
    # Replace the complete gate in both injected prompt and discoverable references.
    # Never edit shared source prompts used by auto_infer_v1.
    legacy_gate = Path(__file__).with_name('gate_prompt.md').read_text()
    text = text.replace(legacy_gate, Path(__file__).with_name('feedforward_gate_prompt.md').read_text())
    if '## Host-managed loop' in text:
        start = text.index('## Host-managed loop')
        end = text.index('## Default workflow', start)
        text = text[:start] + Path(__file__).with_name('feedforward_prompt.md').read_text() + '\n' + text[end:]
    text = (text.replace('robodojo-auto-infer-rollout', 'robodojo-subtask-feedforward-rollout')
        .replace('Keep the original task instruction as the student\'s input.',
                 'Always preserve the episode instruction; the host may append the selected canonical subtask.')
        .replace('Never manually call infer.', 'After execution, select current_intent and subtask or null and call pi05_infer once.')
        .replace('Nonterminal completion returns observations AND a fresh pi05 proposal automatically;\nreview it and hand back to student as soon as appropriate.',
                 'Nonterminal completion returns observations only. Select subtask or null, infer once,\nthen review against the saved current_intent; hand back only when the prefix advances that goal and preserves the correction.')
        .replace('student remains preferred when its proposal is aligned.',
                 'student execution is preferred when its prefix advances the preselected goal and meets the current operation requirements.')
        .replace('prefer aligned student proposals.',
                 'prefer student prefixes that advance the preselected goal and preserve needed corrections.')
        .replace('Keep aligned\nstudent actions;',
                 'Keep student prefixes that advance the preselected goal and meet its execution requirements;')
        .replace('Assess the last execution, judge the fresh\nstudent\'s intent against the current subtask,',
                 'Select the active goal before inference, then review the fresh student prefix against that saved intent,')
        .replace('Then auto infer.', 'Then return observations for subtask selection; do not auto infer.')
        .replace('Two additional tools expose', 'Host service tools expose'))
    from ..prompt_profile import freeform_subtask
    if freeform_subtask():
        text = adapt_freeform_prompt(text)
    return text


def adapt_freeform_prompt(text):
    """Remove the canonical vocabulary, preserving the goal-directed control loop."""
    import re
    replacement = Path(__file__).with_name('feedforward_freeform_prompt.md').read_text()
    text = re.sub(r'## Training-compatible skill fields\n.*?(?=\n## |\Z)',
                  lambda _: replacement, text, flags=re.S)
    text = text.replace('a structured skill or null', 'a concise English instruction or null')
    text = text.replace('optional canonical subtask', 'optional free-text subtask')
    text = text.replace('selected canonical subtask', 'selected free-text subtask')
    text = text.replace('If no predefined skill fits, use a concise custom\nonly for a clearly grounded operation; otherwise use null.',
                        'Use your own concise wording for a clearly grounded operation; otherwise use null.')
    return text


def adapt_specs(specs):
    specs = deepcopy(specs)
    for spec in specs:
        if spec['name'] == 'robodojo_start':
            spec['description'] = 'Start once; return RGB/state and observation_id, without pi05 inference. Next select current_intent and subtask or null.'
        elif spec['name'] == 'robodojo_execute':
            spec['description'] = 'Review against conditioning.current_intent and execute a useful prefix or targeted correction under the goal-directed gate. Return fresh observations; next select current_intent and subtask or null.'
        else:
            spec['description'] = adapt_prompt(spec['description'])
    specs.append(dict(type='function', name='pi05_infer',
        description='Once per observation: select current_intent before seeing the proposal, then a grounded subtask or null. The host echoes intent for review but never adds it to pi05 input. Predict H50 once; review against that same intent, with no candidate resampling.',
        inputSchema=dict(type='object', properties=dict(observation_id={'type':'string'}, subtask=online_schema(),
                             current_intent={'type':'string', 'minLength':1, 'maxLength':600,
                                 'description':'Concise pre-inference goal, immediate operation and critical requirement/error; required even with null subtask.'}),
                         required=['observation_id','subtask','current_intent'], additionalProperties=False)))
    return specs


class FeedforwardMixin:
    feedforward = True

    def start(self, **kwargs):
        packet = super().start(**kwargs)
        gate = Path(__file__).with_name('feedforward_gate_prompt.md').read_text()
        self.run.update(decision_policy_version=DECISION_POLICY_VERSION,
                        gate_instruction_sha256=hashlib.sha256(gate.encode()).hexdigest())
        write_json(self.output/'run.json', self.run)
        return packet

    def _observation(self, directory, **kwargs):
        packet = super()._observation(directory, **kwargs)
        self.observation_id = uuid.uuid4().hex
        self.conditioning = None
        packet['observation_id'] = self.observation_id
        if self.phase == 'infer':
            packet['next_call'] = self.planning_call()
        write_json(self.observation_path, packet)
        return packet

    def planning_call(self):
        from ..prompt_profile import freeform_subtask
        return dict(tool='pi05_infer', observation_id=self.observation_id,
                    subtask=('Concise free-text instruction or null; episode instruction is always preserved'
                             if freeform_subtask() else 'Structured skill or null; episode instruction is always preserved'),
                    current_intent='One concise line describing the active goal and immediate execution requirement before inference; required even with null subtask')

    def policy_observation(self):
        if self.conditioning is None:
            raise RuntimeError('Feedforward inference requires a validated subtask selection')
        return dict(self.obs, instruction=self.conditioning['effective_prompt'])

    def infer_subtask(self, arguments):
        if self.phase != 'infer':
            raise InputError('Inference already consumed or wrong phase; review/execute the latest proposal')
        if set(arguments) != {'observation_id','subtask','current_intent'} or arguments['observation_id'] != self.observation_id:
            raise InputError('Supply latest observation_id, subtask and current_intent; stale, missing or extra fields rejected')
        intent = arguments['current_intent']
        if (not isinstance(intent, str) or not intent.strip() or len(intent) > 600
                or intent.splitlines() != [intent]):
            raise InputError('current_intent must be one nonempty line of at most 600 characters')
        text = render_online(arguments['subtask'])
        prompt = format_prompt(self.obs['instruction'], text)
        self.conditioning = dict(version=ROUTE, decision_policy_version=DECISION_POLICY_VERSION,
            current_intent=intent, observation_id=self.observation_id,
            episode_id=self.episode, step_id=self.tick, subtask=deepcopy(arguments['subtask']),
            subtask_text=text, episode_instruction=self.obs['instruction'], effective_prompt=prompt)
        from ..prompt_profile import freeform_subtask
        if freeform_subtask():
            self.conditioning['subtask_format'] = 'free_text_v1'
        write_json(self.observation_path.parent/'conditioning_selection.json', self.conditioning)
        call = self.next_call()
        try:
            packet = self.infer(**{k:v for k,v in call.items() if k!='tool'})
        except Exception as error:
            self.phase = 'failed'
            raise RuntimeError('Feedforward inference failed; abort without replay or candidate resampling') from error
        self.conditioning['prediction_id'] = self.prediction['prediction_id']
        self.request['gate_instruction'] = Path(__file__).with_name('feedforward_gate_prompt.md').read_text()
        self.request['conditioning'] = deepcopy(self.conditioning)
        packet['conditioning'] = deepcopy(self.conditioning)
        write_json(self.proposal_path.parent/'conditioning.json', self.conditioning)
        write_json(self.proposal_path.parent/'proposal.json', packet)
        write_json(self.output/f'request_{len(self.history):03d}.json', self.request)
        return dict(self.observation_packet, **packet)

    def execute(self, **kwargs):
        conditioning = deepcopy(self.conditioning)
        before = len(self.history)
        try:
            return super().execute(**kwargs)
        finally:
            if len(self.history) > before:
                self.history[-1]['conditioning'] = conditioning
                write_json(self.output/'history.json', self.history)
                if self.observation_packet.get('previous_result') is self.history[-1]:
                    write_json(self.observation_path, self.observation_packet)

    def dispatch_feedforward(self, name, arguments, timing, call_id):
        handlers = {'robodojo_start': self.start, 'robodojo_execute': self.execute}
        with timing.span('service.'+name, step_id=self.tick, call_id=call_id, automatic=False):
            if name == 'pi05_infer':
                return self.infer_subtask(arguments)
            if name not in handlers:
                raise InputError('Unknown feedforward service')
            return handlers[name](**arguments)
