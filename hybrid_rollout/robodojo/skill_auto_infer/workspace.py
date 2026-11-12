"""Give the policy agent a real working directory beside the host's audit files."""
import hashlib
from pathlib import Path
import shutil
import sys

from ..io import write_json


def prepare_workspace(audit, skill_root, *, method='pi05_plus_gpt'):
    if method != 'pi05_plus_gpt':
        raise ValueError('auto_infer_v1 supports pi05_plus_gpt only')
    audit, skill_root = Path(audit).resolve(), Path(skill_root).resolve()
    agent = audit/'agent'
    agent.mkdir()
    (agent/'scratch').mkdir()
    context = agent/'context'
    shutil.copytree(skill_root/'context', context)
    # A discoverable local skill, in addition to the host's explicit injection.
    skill = agent/'.agents/skills'/'robodojo-auto-infer-rollout'
    skill.mkdir(parents=True)
    for name in ('SKILL.md', 'gate_prompt.md'):
        shutil.copy2(skill_root/name, skill/name)
    shutil.copytree(context, skill/'context')
    controller = audit.parent
    from .memory import save_history
    save_history(controller, [])
    workspace = dict(
        controller_route='auto_infer_v1', controller_output=str(controller), source_root=str(skill_root.parent.parent), robot_profile='robodojo',
        python_executable=sys.executable, history_path=str(controller/'history.json'),
        decision_history_path=str(controller/'memory/decisions.md'),
        observations_path=str(controller/'observations'), proposals_path=str(controller/'proposals'),
        execution_files=str(controller/'execution_NNN.npz'),
        eef_diagnostics_files=str(controller/'edit_NNN.json'),
        scratch=str(agent/'scratch'), notes_path=str(agent/'NOTES.md'),
        context_files={str(p.relative_to(agent)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in context.rglob('*') if p.is_file()},
        baseline_full_conversation_available=False)
    write_json(agent/'workspace.json', workspace)
    instructions = (
        '# RoboDojo policy agent workspace\n\n'
        'Act as the online decision module under the injected skill, gate and robot '
        'contract, all supplied in full. Two additional tools expose the host-managed '
        'control loop. Use the latest service reply and its next_call. '
        'Workspace copies are references, not a reading checklist. '
        'No routine notes or source/file exploration are required. '
        'Auxiliary tools remain available for decision-relevant information gaps. '
        f'The host maintains read-only step history at {controller / "memory/decisions.md"}. '
        'After compaction or when earlier events matter, read/search it as needed; '
        'no routine reading or manual maintenance is required. '
        'Shared manipulation strategies and task requirements survive compaction. '
        'If relevant guidance is missing, consult ../manipulation_guidance.md and '
        '../task_success_conditions.md when present. Preserve the unresolved recovery '
        'goal, object/supporting arm, observed attempt outcomes and next unfinished '
        'step; distinguish intentions from verified results. '
        'workspace.json is optional path/interpreter reference; scratch/ and NOTES.md '
        'are optional writable aids, not deliverables. Use English for public output. '
        'Preserve host-owned evidence and executing source; do not use shell scripts '
        'to open a second simulator connection, reset, or bypass the recorded control '
        'tools. Recorded evidence is read-only; private simulator scene '
        'snapshots/object truth are not planning inputs.\n')
    (agent/'AGENTS.md').write_text(instructions)
    (agent/'NOTES.md').write_text(
        '# Episode working memory\n\n'
        'Optional scratch memory. No routine updates required; keep required '
        'task_progress accurate in decision responses.\n')
    return agent
