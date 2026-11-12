"""Finite closing protection: net onset, new-low renewals, explicit re-arming.

There is no pending/closing mode that can retain rows indefinitely. Only the
finite ACK countdown and local opening confirmation rows require retention.
"""
from dataclasses import dataclass, asdict


@dataclass
class ArmClosing:
    peak: float
    progress_anchor: float
    low: float
    mode: str = 'ready'
    remaining: int = 0
    opening_count: int = 0
    opening_peak: float = 0.

    def advance(self, value, config):
        protected = self.remaining > 0
        self.remaining = max(0, self.remaining - 1)
        events = []
        threshold = config.closing_hold_epsilon
        if self.mode == 'ready':
            self.peak = max(self.peak, value)
            if self.peak - value > threshold + 1e-12:
                self.mode = 'latched'
                self.progress_anchor = self.low = value
                self.remaining = config.closing_hold_ticks
                protected = True
                events.append('closing_confirmed')
        else:
            self.low = min(self.low, value)
            if value - self.low > threshold + 1e-12:
                self.opening_count += 1
                self.opening_peak = max(self.opening_peak, value)
                protected = True
                if (self.opening_count >= config.closing_settle_ticks or
                        value - self.low > 2 * threshold + 1e-12):
                    self.mode = 'ready'
                    self.peak = self.opening_peak
                    self.progress_anchor = self.low = value
                    self.opening_count = 0
                    self.opening_peak = 0.
                    events.append('rearmed_on_opening')
            else:
                self.opening_count = 0
                self.opening_peak = 0.
                if self.progress_anchor - value > config.closing_refresh_epsilon + 1e-12:
                    self.progress_anchor = value
                    self.remaining = config.closing_hold_ticks
                    protected = True
                    events.append('significant_closing_progress')
        return protected, events


class SignificantClosingPhases:
    def __init__(self, opening):
        self.arms = [ArmClosing(float(g), float(g), float(g)) for g in opening]

    def advance(self, opening, config):
        protected, events = [], []
        for arm, (state, value) in enumerate(zip(self.arms, opening)):
            keep, changes = state.advance(float(value), config)
            protected.append(keep)
            events.extend(dict(arm=arm, event=event) for event in changes)
        return protected, events

    def snapshot(self):
        return [asdict(arm) for arm in self.arms]
