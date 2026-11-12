"""Eval-only tool availability wording; keep task/manipulation instructions."""
import re


def adapt(text):
    text = re.sub(r'Image viewing and calculation remain available.*?task_progress in your response\.',
        'Use attached camera previews and robot observations. Shell, filesystem image '
        'inspection and private simulator queries are unavailable. Preserve accurate task_progress.',
        text, flags=re.S)
    text = re.sub(r'## Searchable episode history\n.*?(?=\n## Images)',
        '## Searchable episode history\n\nUse robodojo_history for earlier decisions of THIS '
        'episode, including after compaction. Assessments are GPT judgments, not ground truth. '
        'Current RGB/state and actual execution receipts take precedence.\n', text, flags=re.S)
    text = re.sub(r'Do not routinely reopen all three originals\..*?Uncertainty alone is not a takeover justification\.',
        'Use the supplied same-frame views and before/after observations. Image paths do not '
        'grant filesystem access. Occlusion or uncertainty does not prove grasp or justify takeover alone.',
        text, flags=re.S)
    text = text.replace('When only difficult unresolved corrections remain, end the episode only if the existing early-abandon tool is available and its gate permits it.',
        'There is no early-abandon tool in evaluation; continue within the native budget until native termination.')
    text = re.sub(r'If relevant details were lost, consult ../manipulation_guidance.md and\s*../task_success_conditions.md[^\n]*',
        'After compaction retain these injected manipulation and task requirements.', text)
    return text


WORKSPACE = '''# Strict evaluation workspace
Follow the injected skill, gate, manipulation guidance and task-specific requirements.
Use the latest tool reply, attached camera previews and next_call. No shell, private
scene queries, filesystem image reads, cross-episode memory or early abandonment.
Use robodojo_history for same-episode evidence after compaction. Preserve unresolved
recovery goals and distinguish intentions from observed results. Native termination
and the task's native control-step budget determine when the episode ends.
'''
