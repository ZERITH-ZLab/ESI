"""Select starts without removing time from an episode. All intervals are half-open."""


def select_windows(valid, controls, horizon=50, min_valid=15):
    if not 1 <= min_valid <= horizon or len(valid) != len(controls):
        raise ValueError('Require 1 <= min_valid <= horizon and matching timelines')
    n, start = len(valid), 0
    gpt_prefix = [0]
    for control in controls:
        gpt_prefix.append(gpt_prefix[-1] + (control == 'gpt'))
    runs, windows = [], []
    while start < n:
        if not valid[start]:
            start += 1
            continue
        end = start + 1
        while end < n and valid[end]:
            end += 1
        last_gpt = next((t for t in range(end - 1, max(start, end - horizon) - 1, -1)
                         if controls[t] == 'gpt'), None)
        cutoff = end - horizon
        if last_gpt is not None:
            cutoff = max(cutoff, min(last_gpt, end - min_valid))
        runs.append(dict(start_step=start, end_step=end, last_gpt_step=last_gpt,
                         last_allowed_start=cutoff if cutoff >= start else None))
        for t in range(start, cutoff + 1):
            length = min(horizon, end - t)
            windows.append(dict(start_step=t, valid_end_step=end, valid_length=length,
                                real_length=min(horizon, n - t),
                                pad_length=max(0, t + horizon - n),
                                gpt_valid_steps=gpt_prefix[t + length] - gpt_prefix[t],
                                selection='full' if length == horizon else 'gpt_tail'))
        start = end
    return runs, windows


def window_positions(start, total_steps, valid_length, horizon=50):
    """For a future loader: preserve real tails, repeat only past episode end.

    This does not apply a mask to the model. The loader/loss integration remains
    separate; masks must never resume after the first invalid position.
    """
    if not (0 <= start < total_steps and 1 <= valid_length <= min(horizon, total_steps-start)):
        raise ValueError('Invalid window')
    return ([min(start+k, total_steps-1) for k in range(horizon)],
            [k < valid_length for k in range(horizon)])
