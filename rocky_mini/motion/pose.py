"""Pose primitives and the safety clamp.

A Pose is world-frame head + body + antennas. Layers contribute additive deltas
(dicts of field -> value); apply_deltas sums them, and clamp_pose enforces the hard
safety limits from CLAUDE.md: pitch/roll <= 40 deg, |head yaw - body yaw| <= 65 deg,
antennas never commanded to 0 (rest ~10 deg, floor 3 deg).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ANTENNA_REST = 10.0
ANTENNA_MIN = 3.0
ANTENNA_MAX = 45.0
PITCH_LIMIT = 40.0
ROLL_LIMIT = 40.0
YAW_DIFF_LIMIT = 65.0

FIELDS = (
    "head_yaw",
    "head_pitch",
    "head_roll",
    "body_yaw",
    "antenna_left",
    "antenna_right",
    "head_z",
)


@dataclass
class Pose:
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    body_yaw: float = 0.0
    antenna_left: float = ANTENNA_REST
    antenna_right: float = ANTENNA_REST
    head_z: float = 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def apply_deltas(pose: Pose, deltas: list[dict]) -> Pose:
    vals = {f: getattr(pose, f) for f in FIELDS}
    for d in deltas:
        for k, v in d.items():
            if k in vals:
                vals[k] += v
    return Pose(**vals)


def lerp_pose(a: Pose, b: Pose, f: float) -> Pose:
    return Pose(**{fld: getattr(a, fld) + (getattr(b, fld) - getattr(a, fld)) * f for fld in FIELDS})


def blend_factor(dt: float, tau: float = 0.4) -> float:
    """Fraction to move toward target this tick for a ~tau-second blend (no snap)."""
    if tau <= 0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def sdk_targets(pose: Pose) -> tuple[np.ndarray, list[float], float]:
    """Convert a Pose to the reachy-mini SDK set_target arguments (audit F4).

    SDK 1.9.0 contract: head is a 4x4 float64 pose matrix (rotation + translation in
    meters), antennas is [right, left] in radians, body_yaw is radians. Rocky's Pose
    keeps degrees for angles and meters for head_z; the conversion happens only here.
    """
    roll = math.radians(pose.head_roll)
    pitch = math.radians(pose.head_pitch)
    yaw = math.radians(pose.head_yaw)
    # Extrinsic xyz Euler to rotation matrix, same convention as the SDK's
    # create_head_pose (scipy R.from_euler("xyz", ...)).
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    head = np.eye(4)
    head[:3, :3] = rz @ ry @ rx
    head[2, 3] = pose.head_z
    antennas = [math.radians(pose.antenna_right), math.radians(pose.antenna_left)]
    return head, antennas, math.radians(pose.body_yaw)


def clamp_pose(pose: Pose) -> Pose:
    head_pitch = _clamp(pose.head_pitch, -PITCH_LIMIT, PITCH_LIMIT)
    head_roll = _clamp(pose.head_roll, -ROLL_LIMIT, ROLL_LIMIT)
    body_yaw = pose.body_yaw
    head_yaw = _clamp(pose.head_yaw, body_yaw - YAW_DIFF_LIMIT, body_yaw + YAW_DIFF_LIMIT)
    antenna_left = _clamp(pose.antenna_left, ANTENNA_MIN, ANTENNA_MAX)
    antenna_right = _clamp(pose.antenna_right, ANTENNA_MIN, ANTENNA_MAX)
    return Pose(
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        head_roll=head_roll,
        body_yaw=body_yaw,
        antenna_left=antenna_left,
        antenna_right=antenna_right,
        head_z=pose.head_z,
    )
