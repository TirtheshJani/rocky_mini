"""Eridian chord language: base-6 just-intonation scale + additive-synth stinger bank.

Rocky "talks" partly in chords. Eridians count in base six, so the scale has six
degrees tuned to simple integer ratios (just intonation), which gives the bank its
clean, non-Western, slightly alien consonance. Stingers are short additive-synth
chords fired by the conversation loop and the tic linter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Six just-intonation ratios (base-6 scale). Index 0..5.
SCALE_RATIOS: tuple[float, ...] = (1.0, 9 / 8, 5 / 4, 11 / 8, 3 / 2, 5 / 3)


@dataclass(frozen=True)
class StingerSpec:
    """A named chord event.

    degrees: scale-degree indices (into SCALE_RATIOS) sounded together.
    duration_s: total length.
    attack_s / release_s: linear amplitude ramps.
    staccato: if set, the chord is retriggered this many times as short hits
        (mirrors Rocky's word tripling).
    unresolved: leave the chord on a tense interval and do not fade fully to zero
        energy at the tonic (used for the question stinger).
    gain: peak amplitude before mixing.
    """

    degrees: tuple[int, ...]
    duration_s: float = 0.35
    attack_s: float = 0.01
    release_s: float = 0.12
    staccato: int = 1
    unresolved: bool = False
    gain: float = 0.6
    octave: int = 0  # shift the whole chord by this many octaves.


# The ~12 stingers referenced across the app. Question stinger is intentionally
# an unresolved interval (tritone-ish 11/8 over the tonic).
STINGER_BANK: dict[str, StingerSpec] = {
    "wake": StingerSpec(degrees=(0, 2, 4), duration_s=0.45, gain=0.7),
    "listening": StingerSpec(degrees=(0, 4), duration_s=0.25, gain=0.45),
    "thinking": StingerSpec(degrees=(0, 1), duration_s=0.30, gain=0.4),
    "understand": StingerSpec(degrees=(0, 2, 4, 5), duration_s=0.5, gain=0.7),
    "error": StingerSpec(degrees=(0, 3), duration_s=0.35, unresolved=True, gain=0.6),
    "interrupted": StingerSpec(degrees=(3, 0), duration_s=0.18, gain=0.6),
    "remember": StingerSpec(degrees=(0, 2, 5), duration_s=0.4, gain=0.55),
    "jazz_hands": StingerSpec(degrees=(0, 2, 4), duration_s=0.5, staccato=3, gain=0.7),
    "sleepy": StingerSpec(degrees=(0, 1), duration_s=0.7, release_s=0.4, octave=-1, gain=0.4),
    "signal_lost": StingerSpec(degrees=(3,), duration_s=0.6, unresolved=True, gain=0.5),
    "still_watching": StingerSpec(degrees=(0, 4), duration_s=0.8, octave=-1, gain=0.35),
    "question": StingerSpec(degrees=(0, 3), duration_s=0.3, unresolved=True, gain=0.55),
}


@dataclass
class ChordBank:
    """Renders stingers to PCM. base_hz is the tonic (scale degree 0)."""

    base_hz: float = 220.0
    sample_rate: int = 22050
    # Additive partials: (harmonic_number, relative_amplitude).
    partials: tuple[tuple[int, float], ...] = field(
        default_factory=lambda: ((1, 1.0), (2, 0.45), (3, 0.2), (4, 0.1))
    )

    def note_freq(self, degree: int, octave: int = 0) -> float:
        ratio = SCALE_RATIOS[degree % len(SCALE_RATIOS)]
        return self.base_hz * ratio * (2.0 ** octave)

    def _render_chord(self, freqs: list[float], duration_s: float,
                      attack_s: float, release_s: float, unresolved: bool) -> np.ndarray:
        n = max(1, int(duration_s * self.sample_rate))
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        buf = np.zeros(n, dtype=np.float64)
        for f in freqs:
            for harmonic, amp in self.partials:
                buf += amp * np.sin(2.0 * np.pi * f * harmonic * t)
        buf /= max(1, len(freqs))
        env = self._envelope(n, attack_s, release_s, unresolved)
        return (buf * env).astype(np.float32)

    def _envelope(self, n: int, attack_s: float, release_s: float,
                  unresolved: bool) -> np.ndarray:
        env = np.ones(n, dtype=np.float64)
        a = min(n, int(attack_s * self.sample_rate))
        r = min(n, int(release_s * self.sample_rate))
        if a > 0:
            env[:a] = np.linspace(0.0, 1.0, a)
        if r > 0:
            # An unresolved stinger stops short of silence to sound "hanging".
            floor = 0.35 if unresolved else 0.0
            env[n - r:] = np.linspace(1.0, floor, r)
        return env

    def render(self, name: str) -> np.ndarray:
        """Render a named stinger to mono float32 PCM at self.sample_rate."""
        spec = STINGER_BANK[name]
        freqs = [self.note_freq(d, spec.octave) for d in spec.degrees]
        hit = self._render_chord(
            freqs, spec.duration_s / spec.staccato,
            spec.attack_s, spec.release_s, spec.unresolved,
        )
        if spec.staccato > 1:
            gap = np.zeros(int(0.04 * self.sample_rate), dtype=np.float32)
            pieces: list[np.ndarray] = []
            for i in range(spec.staccato):
                pieces.append(hit)
                if i < spec.staccato - 1:
                    pieces.append(gap)
            out = np.concatenate(pieces)
        else:
            out = hit
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1e-9:
            out = out * (spec.gain / peak)
        return out.astype(np.float32)

    def names(self) -> list[str]:
        return sorted(STINGER_BANK.keys())
