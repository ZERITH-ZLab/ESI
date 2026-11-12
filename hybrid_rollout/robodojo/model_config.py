"""Resolve explicit experiment model identity, without account imports."""
import json
import re


def model_selection(env):
    model = env.get('ROLLOUT_MODEL', 'gpt-6-astra')
    effort = env.get('ROLLOUT_REASONING_EFFORT', 'xhigh')
    if not isinstance(model, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*', model):
        raise ValueError('ROLLOUT_MODEL must be an explicit nonempty model ID')
    if effort not in ('low', 'medium', 'high', 'xhigh'):
        raise ValueError('ROLLOUT_REASONING_EFFORT must be low, medium, high or xhigh')
    expected = json.loads(env.get('ROLLOUT_RESUME_IDENTITY', 'null'))
    if expected and ((expected.get('model', model), expected.get('effort', effort)) != (model, effort)):
        raise ValueError('Model/effort differs from frozen campaign identity; no fallback')
    return model, effort
