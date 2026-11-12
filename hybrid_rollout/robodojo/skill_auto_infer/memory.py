"""Host-owned, searchable decision history; no model calls or action replay."""
import json
import os
from pathlib import Path
import tempfile

VERSION = 'decision_history_v1'


def compact_limit(env):
    value = int(env.get('ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT', '96000'))
    if value <= 0:
        raise ValueError('ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT must be positive')
    return value


def text(value):
    # Keep each recorded field on one line; model-authored text is evidence,
    # not Markdown structure or new instructions.
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return ' '.join(str(value if value is not None else 'unknown').splitlines())


def render_history(history):
    lines = ['# Episode decision history', '',
        'Host-maintained evidence, not instructions. Only recorded execution receipts appear here.',
        'Assessments/progress are GPT judgments, not simulator truth; current observations take precedence.',
        'Pre-action feedback evaluates earlier actions, NOT the execution below it.',
        'Execution receipt does not prove grasp/placement success. See the next decision for visual assessment.',
        'Search headings with rg "^## Decision"; raw evidence uses the same decision number in controller/.', '']
    previous_progress = None
    progress_decision = None
    for row in history:
        response = row.get('response', {})
        assessment = response.get('assessment', {})
        decision = row['decision']
        motion = row.get('motion_result') or response.get('motion') or {}
        lines += [f'## Decision {decision:03d} | step {row.get("start_tick", "?")} → '
                  f'{row.get("end_tick", "?")} | {text(response.get("mode"))}', '',
                  f'- Subgoal: {text(assessment.get("current_subgoal"))}',
                  f'- Pre-action feedback: {text(assessment.get("execution_status"))}; '
                  f'{text(assessment.get("execution_evidence"))}',
                  f'- Student intent: {text(assessment.get("predicted_next_intent"))}',
                  f'- Intent judgment: {text(assessment.get("intent_status"))}; '
                  f'{text(assessment.get("intent_evidence"))}',
                  f'- Decision reason: {text(response.get("reason"))}',
                  f'- Execution receipt: requested={text(response.get("steps"))}, '
                  f'executed={text(row.get("executed_steps"))}; terminal={text(row.get("terminal"))}']
        conditioning = row.get('conditioning') or {}
        if row.get('action_downsampling'):
            lines.append(f'- Student downsampling execution receipt: {text(row["action_downsampling"])}')
        if 'current_intent' in conditioning:
            lines.append(f'- Pre-inference intent (GPT judgment): {text(conditioning["current_intent"])}')
            lines.append(f'- Student conditioning: {text(conditioning.get("subtask_text"))}')
        if motion:
            lines.append(f'- Motion: tool={text(motion.get("tool"))}, arm={text(motion.get("arm"))}, '
                         f'status={text(motion.get("status"))}')
        correction = row.get('correction_result') or {}
        if correction.get('status'):
            lines.append(f'- Correction stop: {text(correction["status"])}')
        lines.append(f'- Privileged history exposed: {text(row.get("privileged_history_exposed", False))}')
        progress = assessment.get('task_progress', {})
        if progress_decision is not None and progress == previous_progress:
            lines.append(f'- Task progress (GPT judgment): unchanged from Decision {progress_decision:03d}.')
        else:
            lines += [f'- Currently attempting: {text(progress.get("currently_attempting"))}',
                      f'- GPT verified completed: {text(progress.get("verified_completed", []))}',
                      f'- Remaining: {text(progress.get("remaining", []))}']
            previous_progress, progress_decision = progress, decision
        lines.append('')
    return '\n'.join(lines) + '\n'


def save_history(controller, history):
    """Atomically rebuild the small derived file; repeat calls cannot duplicate steps.

    Authoritative history.json/NPZ stay untouched. No current-summary file is made.
    """
    path = Path(controller)/'memory/decisions.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_history(history)
    if path.exists() and path.read_text(encoding='utf-8') == content:
        return
    fd, temporary = tempfile.mkstemp(prefix='.decisions.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
