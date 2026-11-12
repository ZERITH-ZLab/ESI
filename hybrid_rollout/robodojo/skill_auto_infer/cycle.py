"""Controller-owned inference; never replay a physical service on failure."""
from ..io import InputError


def dispatch(rollout, name, arguments, timing, call_id):
    if getattr(rollout, 'feedforward', False):
        return rollout.dispatch_feedforward(name, arguments, timing, call_id)
    handlers = {'robodojo_start': rollout.start, 'robodojo_execute': rollout.execute}
    if name not in handlers:
        raise InputError('Only robodojo_start and robodojo_execute are available; inference is automatic')
    # Keep service timing comparable to baseline: automatic inference is a
    # separate sibling span, not hidden inside the execute service duration.
    with timing.span('service.' + name, step_id=rollout.tick, call_id=call_id):
        observation = handlers[name](**arguments)
    if rollout.phase == 'done':
        if not observation.get('rollout_finished'):
            raise RuntimeError('Terminal state and observation disagree')
        return observation
    try:
        if rollout.phase != 'infer' or observation.get('rollout_finished'):
            raise RuntimeError('Expected a fresh nonterminal observation')
        next_call = rollout.next_call()
        if not next_call or next_call['tool'] != 'pi05_infer':
            raise RuntimeError('Expected internal pi05_infer next_call')
        with timing.span('service.pi05_infer', step_id=rollout.tick,
                         call_id=call_id, automatic=True):
            proposal = rollout.infer(**{k: v for k, v in next_call.items() if k != 'tool'})
        for key in ('episode_id', 'step_id'):
            if proposal[key] != observation[key]:
                raise RuntimeError(f'Observation/proposal {key} mismatch')
        if (rollout.phase != 'execute' or
                proposal['next_call']['tool'] != 'robodojo_execute' or
                proposal['request_id'] != rollout.request['request_id'] or
                proposal['proposal_path'] != str(rollout.proposal_path)):
            raise RuntimeError('Fresh proposal identity or next_call mismatch')
        # Observation-only fields (task_context, result, arrays/history paths)
        # survive. Proposal supplies fresh counters/request/next_call. Images
        # appear once and come from this same observation.
        return dict(observation, **proposal)
    except InputError as error:
        # start/execute already succeeded. A recoverable no_execution reply
        # would be false and could cause a reset or action replay.
        raise RuntimeError('Automatic inference failed after service completion; '
                           'do not replay start/execute') from error
