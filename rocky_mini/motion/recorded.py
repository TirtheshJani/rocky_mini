"""Adapter for Pollen RecordedMove trajectories vendored offline.

These are the Reachy Mini expression moves in assets/emotes/*.json: each file is a
dense list of SDK set_target frames (a 4x4 head pose matrix, [right, left] antenna
radians, and an optional body_yaw in radians) on a fixed time grid. load_recorded
converts one file into a RecordedTrajectory that duck-types motion.emotes.Trajectory:
it exposes `.duration`, `.sample(t) -> dict` over the delta FIELDS, and
`.finished(t) -> bool`. No inheritance; just the same sampled surface.

The conversion produces ADDITIVE deltas in Rocky's units (degrees for angles, meters
for head_z) taken relative to the FIRST frame. That way a recorded emote composes
over the breathing base exactly like the procedural Trajectory emotes: at t=0 every
delta is ~0 and the emote departs from wherever breathing currently sits, instead of
snapping to Pollen's absolute start pose.

RecordedTrajectory instances are SAMPLED by MotionThread (MovementManager samples
`.sample(t)` at 100 Hz); they are never handed to the SDK's play_move / RecordedMove
machinery. That keeps footgun 2 (never call play_move / cancel_move / goto_target /
play_sound) structurally unreachable.

Antenna sign convention: Pollen stores antennas as [right, left] radians, so
antennas[0] maps to antenna_right and antennas[1] to antenna_left. Some Pollen moves
swing an antenna well past Rocky's rest pose (~10 deg), so the resulting delta can
push a commanded antenna outside [ANTENNA_MIN, ANTENNA_MAX]; clamp_pose bounds it, so
any left/right asymmetry is a hardware-tuning item, not a safety issue.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .pose import FIELDS


def _head_euler_deg(head: list[list[float]]) -> tuple[float, float, float]:
    """Recover (yaw, pitch, roll) in degrees from a head pose matrix.

    This inverts sdk_targets exactly: there R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    (extrinsic xyz Euler), so the closed form below reads the angles straight off
    the rotation block.
    """
    pitch = math.asin(max(-1.0, min(1.0, -head[2][0])))
    roll = math.atan2(head[2][1], head[2][2])
    yaw = math.atan2(head[1][0], head[0][0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _body_yaw_deg(frame: dict) -> float:
    """Body yaw of a frame in degrees, defaulting a missing value to 0.0."""
    body_yaw = frame.get("body_yaw", 0.0)
    if body_yaw is None:
        body_yaw = 0.0
    return math.degrees(body_yaw)


class RecordedTrajectory:
    """Keyframed additive-delta trajectory matching Trajectory's sampled surface."""

    def __init__(self, times: list[float], deltas: list[dict]) -> None:
        if not times:
            raise ValueError("recorded trajectory needs at least one frame")
        self._times = times
        self._deltas = deltas
        self.duration = times[-1]

    def _interp(self, t: float) -> dict:
        # Find the segment [lo, hi] bracketing t. times are sorted and start at 0.
        hi = 0
        while hi < len(self._times) and self._times[hi] <= t:
            hi += 1
        if hi == 0:
            return dict(self._deltas[0])
        if hi >= len(self._times):
            return dict(self._deltas[-1])
        lo = hi - 1
        t0, t1 = self._times[lo], self._times[hi]
        frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        v0, v1 = self._deltas[lo], self._deltas[hi]
        return {f: v0[f] + (v1[f] - v0[f]) * frac for f in FIELDS}

    def sample(self, t: float) -> dict:
        """Interpolate the delta at t. Clamps just before duration, never raises."""
        if t <= 0:
            return dict(self._deltas[0])
        if t >= self.duration:
            return self._interp(self.duration - 1e-6)
        return self._interp(t)

    def finished(self, t: float) -> bool:
        return t >= self.duration


def load_recorded(path) -> RecordedTrajectory:
    """Load a vendored RecordedMove JSON into a RecordedTrajectory of additive deltas."""
    path = Path(path)
    data = json.loads(path.read_text())
    time = data["time"]
    frames = data["set_target_data"]
    if not time or not frames:
        raise ValueError(f"recorded emote {path} has no frames")

    # Shift timestamps so the trajectory starts at t=0.
    t0 = time[0]
    times = [ti - t0 for ti in time]

    # Everything is measured relative to the first frame, so sample(0) is all ~0.0.
    first = frames[0]
    yaw0, pitch0, roll0 = _head_euler_deg(first["head"])
    z0 = first["head"][2][3]
    right0 = math.degrees(first["antennas"][0])
    left0 = math.degrees(first["antennas"][1])
    body0 = _body_yaw_deg(first)

    deltas: list[dict] = []
    for frame in frames:
        yaw, pitch, roll = _head_euler_deg(frame["head"])
        deltas.append(
            {
                "head_yaw": yaw - yaw0,
                "head_pitch": pitch - pitch0,
                "head_roll": roll - roll0,
                "body_yaw": _body_yaw_deg(frame) - body0,
                "antenna_left": math.degrees(frame["antennas"][1]) - left0,
                "antenna_right": math.degrees(frame["antennas"][0]) - right0,
                "head_z": frame["head"][2][3] - z0,
            }
        )
    return RecordedTrajectory(times, deltas)
