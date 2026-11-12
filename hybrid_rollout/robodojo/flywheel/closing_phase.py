"""Command-only closing phases. No contact or successful-grasp inference.

Only ACKs commit this state; selection runs an independent copy. Pending slow
closes are protected conservatively until confirmed or settled. A latched phase
uses a running minimum, so a bounce back to an old minimum cannot extend it.
"""
from dataclasses import dataclass, asdict


@dataclass
class ArmClosure:
    peak: float
    low: float
    progress_anchor: float
    mode: str = 'ready'
    quiet: int = 0
    remaining: int = 0
    opening_count: int = 0
    opening_peak: float = 0.

    def advance(self, value, *, threshold, onset_epsilon, settle_ticks, hold_ticks):
        before = self.mode
        protected = self.remaining > 0 or before in ('pending', 'closing')
        self.remaining = max(0, self.remaining - 1)
        events = []

        # Require persistent opening to re-arm small excursions; a large single
        # opening must still support a one-tick release/regrasp. These rows are
        # protected so ACK replay observes the same persistence as selection.
        if before in ('closing', 'latched') and value - self.low > threshold:
            self.opening_count += 1
            self.opening_peak = max(self.opening_peak, value)
            protected = True
            if self.opening_count >= settle_ticks or value - self.low > 2 * threshold:
                self.mode = 'ready'
                self.peak = self.opening_peak
                self.low = self.progress_anchor = value
                self.quiet = self.opening_count = 0
                self.opening_peak = 0.
                events.append('rearmed_on_opening')
                # Preserve any already owed tail, but opening cannot extend it.
                return protected, events
        else:
            self.opening_count = 0
            self.opening_peak = 0.

        if self.mode == 'ready':
            self.peak = max(self.peak, value)
            if self.peak - value > onset_epsilon:
                self.mode = 'pending'
                self.low = self.progress_anchor = value
                self.quiet = 0
                protected = True
                events.append('closing_candidate')
                if self.peak - value > threshold:
                    self.mode = 'closing'
                    self.remaining = hold_ticks
                    events.append('closing_confirmed')
        elif self.mode in ('pending', 'closing'):
            # Once a phase has begun, do not mistake a slow monotonic tail for
            # settlement merely because each step is below the onset deadband.
            # Returning to an old minimum still is not new progress.
            advancing = self.low - value > 1e-9
            self.low = min(self.low, value)
            if advancing:
                self.progress_anchor = value
                self.quiet = 0
                events.append('closing_progress')
                if self.mode == 'closing':
                    self.remaining = hold_ticks
            else:
                self.quiet += 1
            if self.mode == 'pending' and self.peak - value > threshold:
                self.mode = 'closing'
                self.remaining = hold_ticks
                events.append('closing_confirmed')
            if self.quiet >= settle_ticks:
                if self.mode == 'closing':
                    self.mode = 'latched'
                    events.append('closing_settled')
                else:
                    self.mode = 'ready'
                    self.peak = self.low = self.progress_anchor = value
                    events.append('candidate_cancelled')
        elif self.mode == 'latched':
            # A new material reduction below the old progress minimum may be a
            # renewed close. Oscillation around an already reached target is not.
            if self.progress_anchor - value > threshold:
                self.mode = 'closing'
                self.low = self.progress_anchor = value
                self.quiet = 0
                self.remaining = hold_ticks
                protected = True
                events.append('renewed_closing')
        return protected, events


class ClosingPhases:
    def __init__(self, opening):
        self.arms = [ArmClosure(float(g), float(g), float(g)) for g in opening]

    def advance(self, opening, config):
        protected, events = [], []
        for arm, (state, value) in enumerate(zip(self.arms, opening)):
            keep, changes = state.advance(float(value),
                threshold=config.closing_hold_epsilon,
                onset_epsilon=config.closing_onset_epsilon,
                settle_ticks=config.closing_settle_ticks,
                hold_ticks=config.closing_hold_ticks)
            protected.append(keep)
            events.extend(dict(arm=arm, event=change) for change in changes)
        return protected, events

    def snapshot(self):
        return [asdict(arm) for arm in self.arms]
