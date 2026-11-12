"""Polynomial boundary derivatives and acknowledged joint-reference continuity."""

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq


def time_law(t):
    """Return the shared quintic path progress for normalized chunk time."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def extrema(poly):
    """Return numerical extrema of a polynomial on the unit interval."""
    roots = poly.deriv().roots()
    points = [0.0, 1.0] + [
        float(r.real) for r in roots if abs(r.imag) < 1e-7 and 0 < r.real < 1
    ]
    values = poly(np.asarray(points))
    return float(np.min(values)), float(np.max(values))


class BoundaryPath:
    """Apply C2 Hermite endpoint corrections to the shared timed joint path.

    Corrections occupy only the first and last polynomial spans. All geometry
    knots retain their positions. Physical boundary derivatives stay fixed when
    duration changes, so every candidate duration needs its own feasibility check.
    Floating-point polynomial extrema are not interval-arithmetic certificates.
    """

    def __init__(self, path):
        self.path = path
        self.times = np.r_[0.0, [brentq(lambda t: time_law(t) - s, 0, 1)
                                for s in path.x[1:-1]], 1.0]
        progress = Polynomial([0, 0, 0, 10, -15, 6])
        self.polynomials = []
        for i, (begin, end) in enumerate(zip(self.times[:-1], self.times[1:])):
            s = progress(Polynomial([begin, end - begin])) - path.x[i]
            self.polynomials.append([
                sum((path.c[k, i, j] * s ** (3 - k) for k in range(4)),
                    Polynomial([0])) for j in range(path.c.shape[-1])
            ])

    def corrections(self, duration, velocity, acceleration, end_velocity=None, end_acceleration=None):
        """Yield span index and ascending joint-polynomial coefficient arrays."""
        end_velocity = np.zeros_like(velocity) if end_velocity is None else end_velocity
        end_acceleration = np.zeros_like(acceleration) if end_acceleration is None else end_acceleration
        for index, first in ((0, True), (len(self.polynomials) - 1, False)):
            v, a = (velocity, acceleration) if first else (end_velocity, end_acceleration)
            span_time = duration * (self.times[index + 1] - self.times[index])
            if first:
                v_basis = np.array([0, 1, 0, -6, 8, -3], dtype=float)
                a_basis = np.array([0, 0, .5, -1.5, 1.5, -.5])
            else:
                v_basis = np.array([0, 0, 0, -4, 7, -3], dtype=float)
                a_basis = np.array([0, 0, 0, .5, -1, .5])
            yield index, (v_basis[:, None] * (v * span_time)
                          + a_basis[:, None] * (a * span_time**2))

    def sample(self, times, duration, velocity, acceleration, order=0, *, end_velocity=None, end_acceleration=None):
        """Evaluate joint positions or physical time derivatives up to order 3."""
        t = np.atleast_1d(np.asarray(times, float))
        s = time_law(t)
        sd = 30 * t**2 * (1 - t)**2
        sdd = 60 * t - 180 * t**2 + 120 * t**3
        if order == 0:
            result = self.path(s)
        elif order == 1:
            result = self.path(s, 1) * sd[:, None] / duration
        elif order == 2:
            result = (self.path(s, 2) * sd[:, None]**2
                      + self.path(s, 1) * sdd[:, None]) / duration**2
        elif order == 3:
            sddd = 60 - 360 * t + 360 * t**2
            result = (self.path(s, 3) * sd[:, None]**3
                      + 3 * self.path(s, 2) * (sd * sdd)[:, None]
                      + self.path(s, 1) * sddd[:, None]) / duration**3
        else:
            raise ValueError("Expected derivative order 0..3")
        for index, coefficients in self.corrections(duration, velocity, acceleration, end_velocity, end_acceleration):
            begin, end = self.times[index:index + 2]
            # Use the right-hand span at an interior knot. Position/velocity/
            # acceleration agree there, but jerk can have distinct one-sided values.
            selected = (t >= begin) & ((t <= end) if index == len(self.polynomials) - 1 else (t < end))
            local = (t[selected] - begin) / (end - begin)
            derived = np.polynomial.polynomial.polyder(coefficients, order, axis=0)
            result[selected] += np.polynomial.polynomial.polyval(local, derived).T / (
                duration * (end - begin))**order
        return result

    def bounds(self, duration, velocity, acceleration, *, end_velocity=None, end_acceleration=None):
        """Return continuous position bounds and velocity/acceleration/jerk peaks."""
        corrections = {}
        for i, coefficients in self.corrections(duration, velocity, acceleration, end_velocity, end_acceleration):
            corrections[i] = corrections.get(i, 0) + coefficients
        count = self.path.c.shape[-1]
        position = np.array([np.full(count, np.inf), np.full(count, -np.inf)])
        peaks = np.zeros((3, count))
        for i, polynomials in enumerate(self.polynomials):
            span_time = duration * (self.times[i + 1] - self.times[i])
            for j, original in enumerate(polynomials):
                poly = original + (Polynomial(corrections[i][:, j])
                                   if i in corrections else Polynomial([0]))
                lo, hi = extrema(poly)
                position[0, j] = min(position[0, j], lo)
                position[1, j] = max(position[1, j], hi)
                for order in (1, 2, 3):
                    lo, hi = extrema(poly.deriv(order) / span_time**order)
                    peaks[order - 1, j] = max(peaks[order - 1, j], abs(lo), abs(hi))
        return position, peaks


def record_action(session, action):
    """Remember acknowledged absolute joint targets, never unexecuted proposals.

    Called after one native action ACK and before refreshing the observation.
    The pre-action observation seeds the first finite difference. Acceleration
    remains zero until two issued action intervals exist; its source is explicit.
    """
    history = getattr(session, "_motion_action_history", [])
    if not history:
        history = [dict(episode_id=session.episode_id, step_id=session.step_id - 1,
                        joints=np.asarray(session.obs["states"], float).copy())]
    if (history[-1]["episode_id"] != session.episode_id
            or history[-1]["step_id"] != session.step_id - 1):
        raise ValueError("Discontinuous motion action-reference history")
    history.append(dict(episode_id=session.episode_id, step_id=session.step_id,
                        joints=np.asarray(action, float).copy()))
    session._motion_action_history = history[-3:]
    session._motion_derivatives = {}


def initial_derivatives(session, arm):
    """Resolve derivatives from genuine acknowledged absolute command history."""
    history = getattr(session, "_motion_action_history", [])
    if not history:
        if session.step_id != 0:
            raise ValueError("Missing acknowledged motion boundary history")
        return np.zeros(6), np.zeros(6), "initial_zero_no_previous_action"
    if (history[-1]["episode_id"] != session.episode_id
            or history[-1]["step_id"] != session.step_id):
        raise ValueError("Stale motion action-reference history")
    offset = 0 if arm == "left" else 7
    q = np.asarray([row["joints"][offset:offset + 6] for row in history])
    dt = session.metadata["control_dt"]
    velocity = (q[-1] - q[-2]) / dt
    acceleration = (q[-1] - 2 * q[-2] + q[-3]) / dt**2 if len(q) == 3 else np.zeros(6)
    source = "issued_action_differences" if len(q) == 3 else "first_action_difference_zero_acceleration"
    return velocity, acceleration, source


def remember_derivatives(session, arm, velocity, acceleration):
    """Save analytic derivatives at the last actually acknowledged motion sample."""
    if not hasattr(session, "_motion_derivatives"):
        session._motion_derivatives = {}
    session._motion_derivatives[arm] = dict(
        episode_id=session.episode_id, step_id=session.step_id,
        velocity=np.asarray(velocity, float).copy(),
        acceleration=np.asarray(acceleration, float).copy())


def command_differences(session, arm, target):
    """Report proposed command finite differences across real ACK boundaries.

    This is diagnostic only. No proposal is inserted into acknowledged history.
    Missing derivative orders are reported as unavailable rather than invented.
    """
    history = getattr(session, '_motion_action_history', [])
    offset = 0 if arm == 'left' else 7
    dt = float(session.metadata['control_dt'])
    if not history:
        previous = np.asarray(session.obs['states'][offset:offset + 6], float)
        source = 'initial_measured_state_no_ack'
    else:
        if history[-1]['episode_id'] != session.episode_id or history[-1]['step_id'] != session.step_id:
            raise ValueError('Stale ACK history for proposed command diagnostics')
        previous = np.asarray(history[-1]['joints'][offset:offset + 6], float)
        source = 'genuine_ack_command_history'
    velocity = (np.asarray(target, float) - previous) / dt
    acceleration = jerk = None
    if len(history) >= 2:
        prior = np.asarray(history[-2]['joints'][offset:offset + 6], float)
        prior_velocity = (previous - prior) / dt
        acceleration = (velocity - prior_velocity) / dt
    if len(history) >= 3:
        earlier = np.asarray(history[-3]['joints'][offset:offset + 6], float)
        prior_acceleration = (previous - 2 * prior + earlier) / dt**2
        jerk = (acceleration - prior_acceleration) / dt
    return dict(step_id=session.step_id + 1, source=source, velocity=velocity.tolist(),
                acceleration=None if acceleration is None else acceleration.tolist(),
                jerk=None if jerk is None else jerk.tolist(), proposed_not_acknowledged=True)
