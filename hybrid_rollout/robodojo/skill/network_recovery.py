"""Same-thread wakeup eligibility. No simulator restart or action replay."""
import json

NETWORK_CONTINUE_DELAYS = (10,) * 20


class CodexTransportRecoveryError(RuntimeError):
    """A disconnected ephemeral turn cannot be proven terminal; restart episode."""


def probe_failed_turn(transport, thread_id, turn_id, timeout, error_check,
                      *, ephemeral=False, network_failure=False):
    # includeTurns requires stored history. Do not start a second turn on an
    # ambiguous ephemeral session, and do not convert it to a persistent thread.
    if ephemeral:
        if network_failure:
            raise CodexTransportRecoveryError('Disconnected ephemeral Codex turn timed out; terminal state unavailable')
        raise TimeoutError('Ephemeral Codex recovery cannot confirm a terminal turn')
    response = transport.request('thread/read', dict(threadId=thread_id, includeTurns=True), timeout)
    return closed_network_turn(response, thread_id, turn_id, error_check)


def is_network_error(error):
    text = json.dumps(error, ensure_ascii=False).lower()
    # Account/config/safety failures must not become automatic model calls.
    if any(s in text for s in ('usage limit', 'quota', 'unauthorized', 'authentication',
                               'invalid api key', 'invalid_api_key', 'permission denied', 'safety',
                               'five rejected', 'exceeds 5 cm')):
        return False
    # Codex may omit the low-level cause when a remote compact request fails.
    # Recognize its typed connection failure, without retrying HTTP client errors
    # such as bad credentials, quota limits or invalid requests.
    info = error.get('codexErrorInfo') if isinstance(error, dict) else None
    if isinstance(info, dict) and 'httpConnectionFailed' in info:
        details = info['httpConnectionFailed']
        status = details.get('httpStatusCode') if isinstance(details, dict) else None
        if status is not None:
            try:
                status = int(status)
            except (TypeError, ValueError):
                return False
            return status == 408 or 500 <= status < 600
        return True
    return any(s in text for s in ('stream disconnected', 'responsestreamdisconnected',
        'error decoding response body', 'network error', 'connection reset',
        'connection closed', 'connection timed out', 'connection refused',
        'error sending request', 'tls close_notify'))


def closed_network_turn(response, thread_id, turn_id, error_check=is_network_error):
    """Timeout probe is read-only. Never wake an active or interrupted turn."""
    thread = (response or {}).get('thread', {})
    if thread.get('id') != thread_id or thread.get('status', {}).get('type') != 'idle':
        return None
    turns = thread.get('turns', [])
    if not turns or turns[-1].get('id') != turn_id:
        return None
    turn = turns[-1]
    return turn if turn.get('status') == 'failed' and error_check(turn.get('error')) else None
