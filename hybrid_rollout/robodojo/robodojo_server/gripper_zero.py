"""Contact-triggered zero latch across flywheel rounds; physics/arm targets unchanged."""
from copy import deepcopy

import numpy as np

ARMS = ('left', 'right')
VERSION = 'contact_zero_v4'


class ZeroLatch:
    def __init__(self, owner):
        self.owner = owner
        self.active = {}
        self.student_latest = {}
        self.gpt_closing = {}
        self.gpt_blocked = set()

    def event(self, session, arm, reason, **details):
        row = dict(schema='pi_force_hold_state_v1', version=VERSION,
                   episode_id=session.episode_id, step_id=session.step_id,
                   arm=arm, reason=reason, **details)
        session._write_force_hold_transition(row)
        self.owner.transitions.append(row)

    def activate(self, session, arm, actor, original_target, *, source='student'):
        if arm in self.active:
            return
        lows = {source: float(original_target)}
        if source != 'student' and arm in self.student_latest:
            lows['student'] = self.student_latest[arm]
        self.active[arm] = dict(actor=actor, activation_step_id=session.step_id,
                               original_low=lows)
        self.event(session, arm, 'zero_target_activated', actor=actor,
                   original_target=float(original_target), issued_target=0.)

    def cancel(self, session, arm, reason):
        state = self.active.pop(arm)
        self.event(session, arm, reason, actor=state['actor'])
        # Require a fresh closure after a failed grasp, not repeated activation
        # from the same stale closing history. A future opening rearms used.
        self.owner.arms[arm]['used'] = True
        self.gpt_closing.pop(arm, None)
        self.gpt_blocked.add(arm)

    def refresh(self, session):
        # contact() uses the full action window, not individual low-force ticks.
        for arm, state in list(self.active.items()):
            contact = self.owner.contact(session, arm)
            if not contact['valid']:
                self.cancel(session, arm, 'zero_target_contact_lost_or_unknown')
            elif contact['actor'] != state['actor']:
                self.cancel(session, arm, 'zero_target_object_changed')

    def prepare(self, session, action, *, auxiliary=False):
        self.refresh(session)
        original = np.asarray(action, np.float32).copy()
        issued = original.copy()
        if not auxiliary:
            for i, arm in enumerate(ARMS):
                value = float(original[i*7+6])
                if session.source == 'student':
                    self.student_latest[arm] = value
                elif value > self.owner.config.opening_trend_delta:
                    self.gpt_closing.pop(arm, None)
                    self.gpt_blocked.discard(arm)
                elif value == 0. and arm not in self.gpt_blocked and arm not in self.gpt_closing:
                    from .pi_force_hold import _fingers
                    fingers = _fingers(session)
                    if fingers is not None:
                        self.gpt_closing[arm] = fingers[i].copy()
        overrides = {}
        for arm, state in list(self.active.items()):
            index = ARMS.index(arm)*7+6
            value = float(original[index])
            source = session.source
            # Separate source baselines: GPT keep/close at zero must not turn a
            # returning pi proposal at .7 into a fabricated opening reversal.
            low = state['original_low'].get(source, state['original_low'].get('student', value))
            if not auxiliary and (value >= 1.-1e-7 or
                    value-low+1e-7 >= self.owner.config.opening_trend_delta):
                self.cancel(session, arm, 'zero_target_original_opening')
                continue
            if not auxiliary:
                state['original_low'][source] = min(low, value)
            issued[index] = 0.
            overrides[arm] = dict(actor=state['actor'],
                activation_step_id=state['activation_step_id'],
                original_target=value, issued_target=0.,
                contact=deepcopy(self.owner.contact(session, arm)))
        record = (dict(version=VERSION, observation_step_id=session.step_id,
                       original_action=original.tolist(), arms=overrides) if overrides else None)
        return issued, record

    def observe_gpt_ack(self, session, original):
        """GPT's explicit zero closure also requires measured movement + contact."""
        if session.source == 'student':
            return
        from .pi_force_hold import _fingers
        fingers = _fingers(session)
        if fingers is None:
            return
        for i, arm in enumerate(ARMS):
            if arm in self.active or arm in self.gpt_blocked or arm not in self.gpt_closing:
                continue
            if float(original[i*7+6]) != 0.:
                continue
            closing = ((fingers[i]-self.gpt_closing[arm]) * self.owner.finger_closing_directions[arm])
            contact = self.owner.contact(session, arm)
            if np.all(closing >= self.owner.config.physical_closing_delta_m) and contact['valid']:
                self.activate(session, arm, contact['actor'], 0., source=session.source)
