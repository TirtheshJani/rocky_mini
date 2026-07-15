"""AudioInThread: mic ring buffer -> VAD gate -> speech events; DoA poll -> motion.

The thread owns the voice-input latency domain (plan.md thread 3). All logic here is
pure and clock-injected: tests drive step() against FakeAudioIO with no threads and
no robot. Only AudioIO.get_audio_sample() ever touches hardware.

Latency honesty (footgun 7): a SpeechSegment carries t_last_voiced, the time the
user's last voiced frame was read. Turn latency is measured from there, so the 0.6 s
hangover the gate waits before closing the utterance counts against the budget.

Half-duplex (decisions.md #6/#9): while Rocky speaks, mic frames are dropped and DoA
updates are suppressed, so his own voice never becomes a turn and never drags his
gaze. With half_duplex off (open mic, post-bring-up), speech during playback fires
on_barge_in instead.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .io import AudioIO
from .vad import VAD

logger = logging.getLogger("rocky_mini.audio_in")


@dataclass
class SpeechStart:
    t: float


@dataclass
class SpeechSegment:
    """One finished utterance, preroll included, hangover silence trimmed."""

    pcm: np.ndarray
    sample_rate: int
    t_start: float
    t_last_voiced: float  # the latency anchor (footgun 7)
    t_end: float  # t_last_voiced + hangover: when the gate actually closed


class RingBuffer:
    """Fixed 30 s style ring of mono float32; last(n) returns the newest samples."""

    def __init__(self, seconds: float, rate: int) -> None:
        self._cap = max(1, int(seconds * rate))
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._write = 0
        self._filled = 0

    def append(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size >= self._cap:
            self._buf[:] = block[-self._cap:]
            self._write = 0
            self._filled = self._cap
            return
        end = self._write + block.size
        if end <= self._cap:
            self._buf[self._write:end] = block
        else:
            split = self._cap - self._write
            self._buf[self._write:] = block[:split]
            self._buf[: end - self._cap] = block[split:]
        self._write = end % self._cap
        self._filled = min(self._cap, self._filled + block.size)

    def last(self, n: int) -> np.ndarray:
        n = min(n, self._filled)
        start = (self._write - n) % self._cap
        if start + n <= self._cap:
            return self._buf[start : start + n].copy()
        return np.concatenate([self._buf[start:], self._buf[: (start + n) % self._cap]])


class VadGate:
    """Speech/hangover state machine. feed() is pure given (frame, prob, t)."""

    def __init__(
        self,
        rate: int,
        hangover_s: float = 0.6,
        preroll_s: float = 0.3,
        min_speech_s: float = 0.2,
        enter_prob: float = 0.6,
        exit_prob: float = 0.35,
    ) -> None:
        self.rate = rate
        self.hangover_s = hangover_s
        self.preroll_samples = int(preroll_s * rate)
        self.min_speech_s = min_speech_s
        self.enter_prob = enter_prob
        self.exit_prob = exit_prob
        self.in_speech = False
        self._preroll: list[np.ndarray] = []
        self._utter: list[np.ndarray] = []
        self._voiced_frames = 0  # utterance frames up to and incl. the last voiced one
        self._t_start = 0.0
        self._t_last_voiced = 0.0

    def reset(self) -> None:
        self.in_speech = False
        self._preroll.clear()
        self._utter.clear()

    def feed(self, frame: np.ndarray, prob: float, t: float) -> list:
        events: list = []
        if not self.in_speech:
            if prob >= self.enter_prob:
                self.in_speech = True
                self._utter = list(self._preroll) + [frame]
                self._voiced_frames = len(self._utter)
                self._t_start = t
                self._t_last_voiced = t
                self._preroll = []
                events.append(SpeechStart(t))
            else:
                self._push_preroll(frame)
            return events

        self._utter.append(frame)
        if prob >= self.exit_prob:
            self._t_last_voiced = t
            self._voiced_frames = len(self._utter)
        elif t - self._t_last_voiced >= self.hangover_s:
            frame_s = len(frame) / self.rate
            voiced_duration = self._t_last_voiced + frame_s - self._t_start
            if voiced_duration >= self.min_speech_s:
                events.append(
                    SpeechSegment(
                        pcm=np.concatenate(self._utter[: self._voiced_frames]),
                        sample_rate=self.rate,
                        t_start=self._t_start,
                        t_last_voiced=self._t_last_voiced,
                        t_end=self._t_last_voiced + self.hangover_s,
                    )
                )
            self.reset()
        return events

    def _push_preroll(self, frame: np.ndarray) -> None:
        self._preroll.append(frame)
        total = sum(len(f) for f in self._preroll)
        while self._preroll and total - len(self._preroll[0]) >= self.preroll_samples:
            total -= len(self._preroll[0])
            self._preroll.pop(0)


class AudioInThread:
    """Owns voice input. step() is the unit of work; run() loops it on hardware."""

    def __init__(
        self,
        io: AudioIO,
        vad: VAD,
        motion,  # MovementManager; only set_doa is used
        on_speech: Callable[[SpeechSegment], None],
        is_speaking: Callable[[], bool] = lambda: False,
        on_barge_in: Callable[[], None] | None = None,
        half_duplex: bool = True,
        rate: int = 16000,
        ring_seconds: float = 30.0,
        hangover_s: float = 0.6,
        preroll_s: float = 0.3,
        min_speech_s: float = 0.2,
        doa_poll_s: float = 0.1,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.io = io
        self.vad = vad
        self.motion = motion
        self.on_speech = on_speech
        self.is_speaking = is_speaking
        self.on_barge_in = on_barge_in
        self.half_duplex = half_duplex
        self.rate = rate
        self.clock = clock
        self.ring = RingBuffer(ring_seconds, rate)
        self.gate = VadGate(
            rate=rate, hangover_s=hangover_s, preroll_s=preroll_s, min_speech_s=min_speech_s
        )
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_doa_poll = float("-inf")
        self._doa_poll_s = doa_poll_s

    def step(self) -> bool:
        """One unit of work. Returns True if input data was consumed."""
        block = self.io.get_audio_sample()
        t = self.clock()
        had_data = block.size > 0
        if had_data:
            self.ring.append(block)
            self._pending = np.concatenate([self._pending, np.asarray(block, dtype=np.float32)])
            n = self.vad.frame_samples
            while len(self._pending) >= n:
                frame = self._pending[:n]
                self._pending = self._pending[n:]
                self._process_frame(frame, t)
        self._poll_doa(t)
        return had_data

    def run(self, stop_event) -> None:  # pragma: no cover - thread loop, logic is step()
        logger.info("AudioInThread up (half_duplex=%s)", self.half_duplex)
        while not stop_event.is_set():
            if not self.step():
                time.sleep(0.005)

    # -- internals -----------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, t: float) -> None:
        speaking = self.is_speaking()
        if speaking and self.half_duplex:
            # Rocky is talking and there is no trustworthy AEC guarantee yet:
            # drop the mic entirely so his own voice can never become a turn.
            self.gate.reset()
            self.vad.reset()
            return
        for event in self.gate.feed(frame, self.vad.probability(frame), t):
            if isinstance(event, SpeechStart):
                if speaking and not self.half_duplex and self.on_barge_in is not None:
                    self.on_barge_in()
            elif isinstance(event, SpeechSegment):
                self.on_speech(event)

    def _poll_doa(self, t: float) -> None:
        if t - self._last_doa_poll < self._doa_poll_s:
            return
        self._last_doa_poll = t
        if self.half_duplex and self.is_speaking():
            return
        reading = self.io.get_DoA()
        if reading is None:
            return
        deg, hardware_speech = reading
        # AND-gate (plan.md M8): the XVF3800's own speech flag AND our VAD state,
        # so background noise never drags Rocky's gaze around.
        if hardware_speech and self.gate.in_speech:
            self.motion.set_doa(deg)
