"""Idle procedural motion: breathing, sonar sweep, and sleep-watch breathing.

Pure functions of time returning additive pose deltas. Breathing keeps Rocky visibly
alive at rest; the sonar sweep is the blind "listening for echoes" head drift; the
sleep-watch variant slows the breath to 0.05 Hz and lowers the head over TJ.

IdlePolicy is the cadence gate for rare idle liveliness (a glance or a calm emote during
true silence). It is polled every motion tick but only ever acts long after the last
activity, so the motion loop can consult it cheaply without spamming emotes.
"""

from __future__ import annotations

import math
import random

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


class IdlePolicy:
    """Cadence gate for rare idle liveliness (glances / calm emotes during silence).

    maybe_act() is polled every motion tick and returns None almost always; only after
    idle_glance_s of uninterrupted silence does it roll a weighted choice. note_activity()
    is called whenever Rocky does something, pushing the next opportunity out so idle
    actions never step on a real interaction. All randomness draws from self.rng, seeded
    for determinism so the motion path stays reproducible.
    """

    _CALM_EMOTES = ("bored", "surprised", "welcome")

    def __init__(self, idle_glance_s: float = 180.0, rng: random.Random | None = None) -> None:
        self.idle_glance_s = idle_glance_s
        self.rng = rng if rng is not None else random.Random(0)
        self._next_eval: float | None = None

    def note_activity(self, t: float) -> None:
        """Reset the cadence: the next possible idle action is idle_glance_s from now."""
        self._next_eval = t + self.idle_glance_s

    def maybe_act(self, t: float) -> str | None:
        """Return an idle action name, or None. Rolls only once per idle_glance_s window.

        Weights: None 0.6, "sweep_look" 0.2, a calm recorded emote 0.2 (chosen uniformly
        from bored/surprised/welcome). The first possible action is idle_glance_s after
        the first poll, arming the clock lazily so construction time never leaks in.
        """
        if self._next_eval is None:
            self._next_eval = t + self.idle_glance_s
            return None
        if t < self._next_eval:
            return None
        self._next_eval = t + self.idle_glance_s
        roll = self.rng.random()
        if roll < 0.6:
            return None
        if roll < 0.8:
            return "sweep_look"
        return self._CALM_EMOTES[self.rng.randrange(len(self._CALM_EMOTES))]
