"""Latched, executed-step gate for an explicit unsuccessful collection ending."""
from ..io import InputError, write_json
from ..skill.schema import _obj
from .output_contract import decision_id

VERSION = 'teacher_abandon_streak120_v1'
THRESHOLD = 120


def enabled(env):
    value = env.get('ROLLOUT_ENABLE_TEACHER_ABANDON', '0')
    if value not in ('0', '1'):
        raise ValueError('ROLLOUT_ENABLE_TEACHER_ABANDON must be 0 or 1')
    if value == '1' and (env.get('ROLLOUT_CONTROLLER_ROUTE') not in ('auto_infer_v1', 'subtask_feedforward_v1')
                         or env.get('ROLLOUT_EVALUATION_METHOD') != 'pi05_plus_gpt'):
        raise ValueError('Teacher abandonment requires auto_infer_v1/subtask_feedforward_v1 + pi05_plus_gpt')
    return value == '1'


def gate(history):
    current = maximum = 0
    for row in history:
        steps = row.get('executed_steps', 0)
        # Rejected/zero-motion calls and queries neither count nor break an
        # uninterrupted physical correction. Count actual ACKed controls only.
        if steps <= 0:
            continue
        mode = row.get('response', {}).get('mode')
        if mode in ('eef', 'edit'):
            current += steps
            maximum = max(maximum, current)
        else:
            current = 0
    return dict(version=VERSION, unlocked=maximum >= THRESHOLD,
                consecutive_correction_steps=current, max_consecutive_correction_steps=maximum,
                threshold=THRESHOLD, latched_for_episode=True)


def tool_spec():
    field = {'type': 'string', 'minLength': 1, 'maxLength': 1000}
    return dict(type='function', name='robodojo_abandon',
        description='End this episode unsuccessfully without further physics. Only after a past '
                    'uninterrupted run of >=120 actually executed GPT correction steps. Use only '
                    'when other feasible work is exhausted and repeated recovery shows no progress.',
        inputSchema=_obj(dict(decision_id={'type':'string'}, reason=field,
            no_progress_evidence=field, unresolved={'type':'array', 'minItems':1,
                'maxItems':8, 'items':{'type':'string', 'minLength':1, 'maxLength':300}})))


def execute(rollout, arguments):
    state = gate(rollout.history)
    if rollout.phase != 'execute' or not state['unlocked']:
        raise InputError('Abandon locked: requires >=120 consecutive executed GPT correction steps in this episode')
    if not isinstance(arguments, dict) or set(arguments) != {'decision_id','reason','no_progress_evidence','unresolved'}:
        raise InputError('Abandon requires decision_id, reason, no_progress_evidence and unresolved only')
    if arguments['decision_id'] != decision_id(rollout.request['request_id']):
        raise InputError('Stale abandon decision_id')
    for key in ('reason', 'no_progress_evidence'):
        if not isinstance(arguments[key], str) or not arguments[key].strip() or len(arguments[key]) > 1000:
            raise InputError('Abandon reason/evidence must be nonempty strings of at most 1000 characters')
    items = arguments['unresolved']
    if (not isinstance(items, list) or not 1 <= len(items) <= 8 or
            any(not isinstance(x, str) or not x.strip() or len(x) > 300 for x in items)):
        raise InputError('Abandon requires 1..8 short unresolved items')
    audit = dict(episode_id=rollout.episode, step_id=rollout.tick,
                 decision=len(rollout.history), gate=state, arguments=arguments,
                 status='requested', physical_steps=0)
    write_json(rollout.output/'abandon.json', audit)
    # Existing finish closes recording, not physics stepping. Never retry after
    # an uncertain finish ACK, and never bypass the normal action validator.
    result = rollout.finish('teacher_abandon')
    if result.get('complete') or result.get('success'):
        raise RuntimeError('Unexpected native terminal state while abandoning')
    result.update(termination_reason='teacher_abandon', task_success=False,
                  collection_complete=True, native_complete=False, abandon_gate=state,
                  abandon_reason=arguments['reason'], unresolved=items)
    write_json(rollout.output/'result.json', result)
    audit['status'] = 'acknowledged'
    write_json(rollout.output/'abandon.json', audit)
    return dict(episode_id=rollout.episode, step_id=rollout.tick, rollout_finished=True,
                next_call=None, result=result, images=[], counters=dict(rollout.counters))


def acknowledged(result, outcome):
    """Distinguish a deliberate failed sample from arbitrary incomplete runs."""
    state = result.get('abandon_gate') or {}
    return (result.get('termination_reason') == result.get('reason') == 'teacher_abandon'
        and result.get('collection_complete') is True and result.get('complete') is False
        and result.get('success') is False and result.get('task_success') is False
        and state.get('version') == VERSION and state.get('unlocked') is True
        and state.get('max_consecutive_correction_steps', 0) >= THRESHOLD
        and outcome.get('reason') == 'teacher_abandon' and outcome.get('complete') is False
        and outcome.get('status') == 'teacher_abandoned' and outcome.get('task_success') is False)


PROMPT = '''

## Gated early abandonment (collection only)
Completing the whole task remains the normal goal. A missed grasp, one planning
failure, uncertainty, or low confidence is not a reason to abandon. If the same
unresolved problem has resisted several different attempts with substantial
effort and no observable improvement, you may defer it and do other independently
feasible work. Do not mark deferred work complete, break required task ordering,
or damage completed work. Safely release or place a held object when feasible;
there is no special forced arm-return action before abandoning. Revisit deferred
work if a genuinely new opportunity appears. Partial work does not guarantee score.

The host reports abandon_gate. It unlocks after this episode has EVER contained
at least 120 consecutive actually executed GPT correction controls (edit or eef,
including move/grasp). Student controls break the current streak; queries, failed
planning and zero-execution calls do not add steps. Once unlocked it stays unlocked
for the rest of this episode, even after student control resumes. Unlocking is
permission, not advice to stop. Never consume steps or split actions to unlock it.

Only when other feasible work is done or cannot proceed and the remaining hard
problem still shows no progress, you may call robodojo_abandon with the latest
decision_id, a short reason, unresolved items and observed no-progress evidence.
Continue when there is real progress. No extra per-step reflection checklist is
required: reuse your existing assessments. This is the sole gated exception to
the native-termination requirement below/above; mode=stop remains disallowed.
Abandon ends collection as a FAILURE, not native completion or success, and does
not execute further actions. Do not claim a final native score. After its receipt,
stop controlling and follow any host request for an episode summary.
'''
