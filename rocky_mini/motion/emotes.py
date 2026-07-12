"""Emote trajectories and the jazz-hands reflex.

Two kinds of emote:
  - Primary trajectories (nods, tilts) blended over breathing. Trajectory mimics the
    SDK's RecordedMove.evaluate(t), which RAISES at t >= duration (motion/recorded_move.py
    in the SDK). sample() clamps just before the boundary so the 100 Hz loop never trips
    that raise. This de-risks the one unverified SDK surface (plan M1).
  - Reflex deltas (jazz_hands): a fast procedural antenna shake fired somatically by the
    tic linter on a tripled word. Additive, never blended (blending would smear the shake).
"""

from __future__ import annotations

import math

from .pose import FIELDS

TWO_PI = 2.0 * math.pi


class Trajectory:
    """Keyframed additive-delta trajectory with SDK-compatible boundary semantics."""

    def __init__(self, keyframes: list[tuple[float, dict]]) -> None:
        if not keyframes:
            raise ValueError("trajectory needs at least one keyframe")
        keyframes = sorted(keyframes, key=lambda kf: kf[0])
        self._times = [kf[0] for kf in keyframes]
        # Densify each keyframe over the delta fields (missing -> 0.0).
        self._values = [{f: kf[1].get(f, 0.0) for f in FIELDS} for kf in keyframes]
        self.duration = self._times[-1]

    def evaluate(self, t: float) -> dict:
        """Interpolate at t. RAISES at t >= duration, exactly like the SDK helper."""
        if t < 0:
            raise ValueError("t must be >= 0")
        if t >= self.duration:
            raise ValueError(f"t={t} is at or past duration={self.duration}")
        # Find the segment [i, i+1] containing t.
        hi = 0
        while hi < len(self._times) and self._times[hi] <= t:
            hi += 1
        lo = hi - 1
        t0, t1 = self._times[lo], self._times[hi]
        frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        v0, v1 = self._values[lo], self._values[hi]
        return {f: v0[f] + (v1[f] - v0[f]) * frac for f in FIELDS}

    def sample(self, t: float) -> dict:
        """Safe evaluate: clamp t just before the boundary so evaluate never raises."""
        if t <= 0:
            return dict(self._values[0])
        if t >= self.duration:
            return self.evaluate(self.duration - 1e-6)
        return self.evaluate(t)

    def finished(self, t: float) -> bool:
        return t >= self.duration


def jazz_hands_delta(t_since: float, dur: float = 1.2, amp: float = 20.0, freq: float = 6.0) -> dict:
    """Antenna shake +/-20 deg at 6 Hz for 1.2 s. Empty dict once finished."""
    if t_since < 0 or t_since >= dur:
        return {}
    shake = amp * math.sin(TWO_PI * freq * t_since)
    return {"antenna_left": shake, "antenna_right": -shake}


# Primary trajectory emotes (blended over breathing).
PRIMARY_EMOTES: dict[str, Trajectory] = {
    "understand": Trajectory([(0.0, {}), (0.25, {"head_pitch": 12.0}), (0.6, {"head_pitch": 0.0})]),
    "thinking": Trajectory(
        [(0.0, {}), (0.5, {"head_roll": 8.0, "head_yaw": 6.0}), (1.0, {"head_roll": 0.0, "head_yaw": 0.0})]
    ),
    "sleepy": Trajectory([(0.0, {}), (0.8, {"head_pitch": -10.0}), (1.5, {"head_pitch": -12.0})]),
}

# Fast reflex emotes (additive, not blended).
REFLEX_EMOTES = {"jazz_hands": jazz_hands_delta}


def is_reflex(name: str) -> bool:
    return name in REFLEX_EMOTES


def is_primary(name: str) -> bool:
    return name in PRIMARY_EMOTES
