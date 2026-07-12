"""Head wobble locked to the speech playout clock.

The Mixer reports a per-frame RMS envelope of what it is playing. Rocky nods with the
envelope, but the sound reaches the ear ~0.2 s after the sample is scheduled, so the
wobble is delayed to the playout clock to stay in sync (plan: wobble_playout_delay_s).
"""

from __future__ import annotations

from collections import deque


class WobbleTracker:
    def __init__(self, gain: float = 8.0, delay_s: float = 0.2) -> None:
        self.gain = gain
        self.delay_s = delay_s
        self._scheduled: deque[tuple[float, float]] = deque()  # (play_time, rms)
        self._current = 0.0

    def feed(self, rms: float, t: float) -> None:
        """Register an envelope value produced at time t; it plays at t + delay."""
        self._scheduled.append((t + self.delay_s, float(rms)))

    def delta(self, t: float) -> dict:
        """Return the wobble pose delta at time t (nod down proportional to loudness)."""
        while self._scheduled and self._scheduled[0][0] <= t:
            self._current = self._scheduled.popleft()[1]
        return {"head_pitch": -self.gain * self._current}
