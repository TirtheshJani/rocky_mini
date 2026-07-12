"""The Mixer: the single owner of push_audio_sample.

Two buses feed it:
  - VoiceBus: generation-tagged TTS PCM. On barge-in the conversation loop bumps the
    generation counter; the Mixer drops any voice chunk tagged with a stale generation,
    so interrupted speech stops immediately.
  - ChordBus: Eridian stingers and the underlay bed (already leveled by the caller).

Chain per output frame: ring-modulate voice (phase carried across chunks) + chord,
sum, tanh soft-clip, push. Voice is optionally ring-modulated for the alien timbre.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .dsp import ring_mod, soft_clip
from .io import AudioIO


@dataclass
class VoiceItem:
    pcm: np.ndarray
    generation: int


class Mixer:
    """Synchronous, deterministic mixer.

    render() drains both buses and pushes fixed-size frames to the AudioIO. It is
    called on demand in sim/tests; a thread wrapper (pump loop) drives it live. Keeping
    the core synchronous makes the generation-flush and DSP behavior unit-testable.
    """

    def __init__(
        self,
        io: AudioIO,
        *,
        frame: int = 320,
        ring_carrier_hz: float = 140.0,
        sample_rate: int = 22050,
        ring_mod_enabled: bool = True,
    ) -> None:
        self.io = io
        self.frame = frame
        self.ring_carrier_hz = ring_carrier_hz
        self.sample_rate = sample_rate
        self.ring_mod_enabled = ring_mod_enabled
        self._voice: deque[VoiceItem] = deque()
        self._chord: deque[np.ndarray] = deque()
        self._generation = 0
        self._carrier_phase = 0.0
        self.dropped_generations = 0  # for tests/metrics

    # -- producers ---------------------------------------------------------
    def push_voice(self, pcm: np.ndarray, generation: int) -> None:
        self._voice.append(VoiceItem(np.asarray(pcm, dtype=np.float32), generation))

    def push_chord(self, pcm: np.ndarray) -> None:
        self._chord.append(np.asarray(pcm, dtype=np.float32))

    def set_generation(self, generation: int) -> None:
        """Barge-in: any queued voice older than this is discarded on next render."""
        self._generation = generation

    @property
    def generation(self) -> int:
        return self._generation

    def pending(self) -> bool:
        return bool(self._voice) or bool(self._chord)

    # -- consumer ----------------------------------------------------------
    def _drain_voice(self) -> np.ndarray:
        pieces: list[np.ndarray] = []
        while self._voice:
            item = self._voice.popleft()
            if item.generation < self._generation:
                self.dropped_generations += 1
                continue
            if self.ring_mod_enabled:
                modded, self._carrier_phase = ring_mod(
                    item.pcm, self.ring_carrier_hz, self.sample_rate, self._carrier_phase
                )
                pieces.append(modded)
            else:
                pieces.append(item.pcm)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def _drain_chord(self) -> np.ndarray:
        pieces = list(self._chord)
        self._chord.clear()
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def render(self) -> np.ndarray:
        """Drain both buses, mix, push frames, and return the full mixed buffer."""
        voice = self._drain_voice()
        chord = self._drain_chord()
        n = max(len(voice), len(chord))
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        voice = _pad(voice, n)
        chord = _pad(chord, n)
        mixed = soft_clip(voice + chord)
        self._push_frames(mixed)
        return mixed

    def _push_frames(self, mixed: np.ndarray) -> None:
        for start in range(0, len(mixed), self.frame):
            self.io.push_audio_sample(mixed[start : start + self.frame])


def _pad(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x
    out = np.zeros(n, dtype=np.float32)
    out[: len(x)] = x
    return out
