"""Idle procedural motion: breathing, sonar sweep, and sleep-watch breathing.

Pure functions of time returning additive pose deltas. Breathing keeps Rocky visibly
alive at rest; the sonar sweep is the blind "listening for echoes" head drift; the
sleep-watch variant slows the breath to 0.05 Hz and lowers the head over TJ.
"""

from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi


def breathing_delta(
    t: float,
    freq: float = 0.25,
    z_amp: float = 0.008,
    pitch_amp: float = 2.0,
    antenna_amp: float = 1.5,
    sweep_amp: float = 4.0,
    sweep_freq: float = 0.07,
) -> dict:
    """Gentle breathing bob + slow blind sonar-sweep yaw drift."""
    phase = TWO_PI * freq * t
    return {
        "head_z": z_amp * math.sin(phase),
        "head_pitch": pitch_amp * math.sin(phase),
        "head_yaw": sweep_amp * math.sin(TWO_PI * sweep_freq * t),
        "antenna_left": antenna_amp * math.sin(phase),
        "antenna_right": antenna_amp * math.sin(phase + 0.3),
    }


def sleepwatch_delta(t: float, freq: float = 0.05, doa_deg: float | None = None) -> dict:
    """Head lowered toward TJ's last DoA, drooped antennas, very slow breath."""
    phase = TWO_PI * freq * t
    delta = {
        "head_pitch": -12.0 + 3.0 * math.sin(phase),
        "antenna_left": -6.0,
        "antenna_right": -6.0,
        "head_z": 0.004 * math.sin(phase),
    }
    if doa_deg is not None:
        delta["head_yaw"] = 0.5 * doa_deg
    return delta
