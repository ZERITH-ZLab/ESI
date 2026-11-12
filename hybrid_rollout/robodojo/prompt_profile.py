"""Explicit prompt profiles; legacy guided and baseline behavior stays unchanged."""
import os
import re

GUIDED = 'guided_v1'
BASELINE = 'baseline_task_only_v1'
NO_GUIDANCE = 'ablation_no_guidance_v1'
FREEFORM = 'ablation_freeform_subtask_v1'
PRETRAINED = 'ablation_pretrained_vla_v1'
TRANSFER_BASELINE = 'transfer_pretrained_freeform_no_guidance_v1'
ENV_KEY = 'ROLLOUT_PROMPT_PROFILE'


def profile(env=None):
    value = (os.environ if env is None else env).get(ENV_KEY, GUIDED)
    if value not in (GUIDED, BASELINE, NO_GUIDANCE, FREEFORM, PRETRAINED, TRANSFER_BASELINE):
        raise ValueError(f'Unknown prompt profile: {value}')
    return value


def baseline(env=None):
    return profile(env) == BASELINE


def guidance_disabled(env=None):
    return profile(env) in (BASELINE, NO_GUIDANCE, TRANSFER_BASELINE)


def freeform_subtask(env=None):
    return profile(env) in (FREEFORM, TRANSFER_BASELINE)


def identity(env):
    selected = profile(env)
    if selected in (NO_GUIDANCE, FREEFORM, PRETRAINED, TRANSFER_BASELINE):
        required = dict(ROLLOUT_STRICT_EVAL='1', ROLLOUT_CONTROLLER_ROUTE='subtask_feedforward_v1',
            ROLLOUT_ENABLE_SCENE_QUERY='0', ROLLOUT_ENABLE_TASK_MEMORY='0',
            ROLLOUT_ENABLE_TEACHER_ABANDON='0', ROBODOJO_PI_FORCE_HOLD_ENABLED='0',
            ROLLOUT_TASK_REFERENCE_IMAGES='0' if guidance_disabled(env) else '1')
        for key, value in required.items():
            if env.get(key) != value:
                raise ValueError(f'Ablation profile requires {key}={value}')
        guided = not guidance_disabled(env)
        return dict(prompt_profile=selected, task_guidance=guided,
                    manipulation_guidance=guided, task_context=guided, task_reference_images=guided,
                    subtask_format='free_text_v1' if freeform_subtask(env) else 'canonical_skill_v1')
    if not baseline(env):
        return {}  # Preserve existing frozen guided-evaluation identities.
    required = dict(ROLLOUT_STRICT_EVAL='1', ROLLOUT_CONTROLLER_ROUTE='auto_infer_v1',
        ROLLOUT_ENABLE_SCENE_QUERY='0', ROLLOUT_ENABLE_TASK_MEMORY='0',
        ROLLOUT_ENABLE_TEACHER_ABANDON='0', ROLLOUT_ACTION_DOWNSAMPLING='off',
        ROBODOJO_PI_FORCE_HOLD_ENABLED='0', ROLLOUT_TASK_REFERENCE_IMAGES='0')
    for key, value in required.items():
        if env.get(key) != value:
            raise ValueError(f'Baseline profile requires {key}={value}')
    return dict(prompt_profile=BASELINE, task_guidance=False,
                manipulation_guidance=False, task_context=False, task_reference_images=False)


def adapt(text):
    """Remove embedded heuristics as well as the separately injected guidance.

    Call only when guidance_disabled. Tool geometry, limits and execution semantics
    (including explicit GPT close waits) stay accurate and unchanged.
    """
    text = re.sub(r'## Rigid-object pick-and-place preferences\n.*?(?=\n## |\Z)', '', text, flags=re.S)
    text = text.replace('Apply the rigid-object pick-and-place preferences in the robot contract; these\n'
                        'also apply to ordinary EEF corrections, not only to motion tools.\n', '')
    text = text.replace('current images/state, initial task_context,', 'current images/state, original task instruction,')
    text = text.replace('Use the native instruction and task_context, not a hard-coded script.',
                        'Use the original environment task instruction as the objective.')
    text = re.sub(r'If requires_arm_return=true, release,.*?before moving sideways\.',
        'Follow any finishing or arm-return requirements stated in the original task instruction.',
        text, flags=re.S)
    text = text.replace('Follow the injected skill, gate, manipulation guidance and task-specific requirements.',
                        'Follow the injected tool protocol, decision gate and original environment task instruction.')
    text = re.sub(r'Shared manipulation strategies and task requirements survive compaction\. '
                  r'.*?when present\. ', '', text, flags=re.S)
    return text
