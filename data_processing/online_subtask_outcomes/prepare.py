"""Read finalized execution receipts and freeze exactly what pi05 received."""
from copy import deepcopy
from pathlib import Path

from data_processing.subtasks.io import read, require, digest
from data_processing.subtasks.prepare import prepare as prepare_offline
from data_processing.subtasks.evidence import annotation_payload
from hybrid_rollout.robodojo.skill_auto_infer.subtask_instruction import render_instruction, format_prompt
from . import VERSION


def prepare(archive):
    result = prepare_offline(archive)
    task = result['payload']['refs'].get('episode.task_instruction', {}).get('value')
    require(task, 'Online route requires the original instruction in sim/reset.json')
    history = read(Path(archive) / 'controller/history.json')
    for d, row in zip(result['payload']['decisions'], history, strict=True):
        c = row.get('conditioning')
        prefix = f'D{d["decision_id"]}'
        require(isinstance(c, dict) and all(k in c for k in
                ('subtask', 'subtask_text', 'effective_prompt', 'current_intent', 'episode_instruction')),
                f'{prefix}: Missing online conditioning is NOT null')
        require(c.get('episode_id') == result['metadata']['episode_id'] and
                c.get('step_id') == d['start_step'] and c.get('prediction_id') is not None and
                c['prediction_id'] == row.get('prediction_id'), f'{prefix}: conditioning identity mismatch')
        require(c['episode_instruction'] == task, f'{prefix}: original episode instruction mismatch')
        require(isinstance(c['current_intent'], str) and c['current_intent'].strip(),
                f'{prefix}: missing current_intent')
        require(isinstance(c['effective_prompt'], str) and c['effective_prompt'].strip(),
                f'{prefix}: missing actual model input')
        if c['subtask'] is None:
            require(c['subtask_text'] is None and c['effective_prompt'] == task,
                    f'{prefix}: null input must be episode-only')
        else:
            require(isinstance(c['subtask_text'], str) and c['subtask_text'].strip(),
                    f'{prefix}: missing actual subtask text')
            require(c['effective_prompt'] == format_prompt(task, c['subtask_text']),
                    f'{prefix}: effective_prompt disagrees with recorded text')
        # Malformed skill declarations are reviewable inputs, never rewritten.
        # Missing/misaligned actual input above is a hard integrity error instead.
        issues = []
        try:
            rendered = render_instruction(c['subtask'])
            if rendered != c['subtask_text']:
                issues.append('Recorded text differs from the structured skill rendering')
        except (ValueError, KeyError, TypeError) as error:
            issues.append(str(error))
        d['conditioning'] = deepcopy(c)
        d['input_format_issues'] = issues
    result['payload']['version'] = VERSION
    summary = read(Path(archive) / 'sim/summary.json')
    truncation = summary.get('manual_truncation')
    if truncation is not None:
        require(truncation.get('retained_steps') == result['payload']['total_steps']
                and summary.get('reason') == 'controller_error', 'Invalid manual truncation endpoint')
        last = result['payload']['decisions'][-1]
        key = f'd{last["decision_id"]:04d}.archive.manual_truncation'
        result['payload']['refs'][key] = dict(decision_id=last['decision_id'],
            field='archive.manual_truncation', available_at_step=last['end_step'], phase='after',
            value='Recording was externally interrupted by a wall-time limit and manually trimmed '
                  'to this last fully recorded decision endpoint. No later execution or outcome '
                  'feedback is available in this dataset. This is not a GPT abandonment decision '
                  'and does not establish physical success or failure.')
        result['metadata']['manual_truncation'] = deepcopy(truncation)
    result['input_hash'] = digest({k: v for k, v in result.items() if k != 'input_hash'})
    return result


def model_payload(prepared):
    result = annotation_payload(prepared)
    # Remove only verified repetition from model context, not the frozen input file.
    # The task instruction is already supplied once in episode_context.
    for d in result['decisions']:
        c = d['conditioning']
        d['conditioning'] = {k: c[k] for k in ('subtask', 'subtask_text', 'current_intent')}
    result['required_decision_ids'] = [d['decision_id'] for d in result['decisions']
                                       if d['end_step'] > d['start_step']]
    return result


def prepare_batch(roots, output, limit=None, **kwargs):
    from data_processing.subtasks.pipeline import prepare_batch as shared
    return shared(roots, output, limit, prepare_fn=prepare, prepare_version=VERSION, **kwargs)
