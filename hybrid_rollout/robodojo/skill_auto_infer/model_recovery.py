"""Allowlisted model failures only; never action/validation failures."""
import json
from ..skill.network_recovery import is_network_error

DELAYS = (60,) * 10


class ModelRecoveryExhausted(RuntimeError):
    pass


def eligible(error):
    if not isinstance(error, dict):
        return False
    text = json.dumps(error).lower()
    if any(s in text for s in ('quota', 'usage', 'unauthorized', 'authentication',
                              'safety', 'invalid_api_key', 'permission')):
        return False
    if is_network_error(error):
        return True
    if error.get('codexErrorInfo') == 'serverOverloaded':
        return True
    # Only the observed opaque error, not arbitrary invalid request/schema errors.
    message = error.get('message', '')
    try:
        opaque = json.loads(message) == {'detail': 'Bad Request'}
    except (ValueError, TypeError):
        opaque = False
    return opaque and error.get('codexErrorInfo') == 'other'
