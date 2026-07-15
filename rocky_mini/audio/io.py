"""AudioIO seam: one Protocol, three implementations.

The Mixer and AudioInThread talk only to this Protocol, so Windows dev (sounddevice),
the physical robot (mini.media), and tests (Fake) all share one code path. Real
backends are imported lazily so importing this module never requires the SDK,
sounddevice, or a working GStreamer.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AudioIO(Protocol):
    """Mono float32 audio in/out plus optional direction-of-arrival."""

    def get_audio_sample(self) -> np.ndarray:
        """Return the next input block (16 kHz mono float32). Empty array if none."""
        ...

    def push_audio_sample(self, frame: np.ndarray) -> None:
        """Play one output frame. Only the Mixer calls this."""
        ...

    def flush(self) -> None:
        """Drop any audio already queued for playout (barge-in). No-op off-robot."""
        ...

    def get_DoA(self) -> tuple[float, bool] | None:
        """(signed degrees off straight-ahead, speech_detected), or None if unavailable.

        Positive degrees = to the robot's left (matches Pose.head_yaw sign).
        """
        ...

    def close(self) -> None:
        ...


class FakeAudioIO:
    """In-memory AudioIO for tests and sim. Records everything pushed."""

    def __init__(self, doa: float | None = None, speech: bool = True) -> None:
        self._in: deque[np.ndarray] = deque()
        self.pushed: list[np.ndarray] = []
        self.flushed = 0
        self._doa = doa
        self._speech = speech

    def feed_input(self, block: np.ndarray) -> None:
        self._in.append(np.asarray(block, dtype=np.float32))

    def get_audio_sample(self) -> np.ndarray:
        if self._in:
            return self._in.popleft()
        return np.zeros(0, dtype=np.float32)

    def push_audio_sample(self, frame: np.ndarray) -> None:
        self.pushed.append(np.asarray(frame, dtype=np.float32).copy())

    def pushed_concat(self) -> np.ndarray:
        if not self.pushed:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.pushed)

    def flush(self) -> None:
        self.flushed += 1

    def get_DoA(self) -> tuple[float, bool] | None:
        if self._doa is None:
            return None
        return (self._doa, self._speech)

    def set_DoA(self, doa: float | None, speech: bool = True) -> None:
        self._doa = doa
        self._speech = speech

    def close(self) -> None:
        self._in.clear()


class SoundDeviceIO:
    """PC dev backend using the sounddevice library. Lazy import.

    Used with the SDK's media_backend="no_media" so Windows dev does not depend on
    GStreamer. Half-duplex friendly (separate in/out streams).
    """

    def __init__(self, sample_rate_in: int = 16000, sample_rate_out: int = 16000,
                 block: int = 320) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "sounddevice is not installed. Install with: pip install 'rocky_mini[dev]'"
            ) from exc
        self._sd = sd
        self._rate_in = sample_rate_in
        self._rate_out = sample_rate_out
        self._block = block
        self._in = sd.InputStream(samplerate=sample_rate_in, channels=1, blocksize=block)
        self._out = sd.OutputStream(samplerate=sample_rate_out, channels=1, blocksize=block)
        self._in.start()
        self._out.start()

    def get_audio_sample(self) -> np.ndarray:  # pragma: no cover - hardware path
        data, _ = self._in.read(self._block)
        return data.reshape(-1).astype(np.float32)

    def push_audio_sample(self, frame: np.ndarray) -> None:  # pragma: no cover
        self._out.write(np.asarray(frame, dtype=np.float32).reshape(-1, 1))

    def flush(self) -> None:
        pass  # sounddevice writes are near-synchronous; nothing meaningful to drop.

    def get_DoA(self) -> tuple[float, bool] | None:
        return None  # no mic array on a laptop.

    def close(self) -> None:  # pragma: no cover - hardware path
        for stream in (self._in, self._out):
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


class ReachyMediaIO:
    """Physical-robot backend wrapping mini.media (reachy-mini 1.9.0 contract).

    Fails loudly at construction if the SDK surface it needs is missing: a robot
    that cannot hear or speak must error at launch, not shrug (audit F6/F7).
    """

    # SDK DoA convention (audio_doa.py docstring): 0 rad = left, pi/2 = front/back,
    # pi = right. Mapped to signed degrees where positive = robot's left, matching
    # Pose.head_yaw. The front/back ambiguity and the sign are a bring-up.md check.
    _DOA_FRONT_RAD = 1.5707963267948966  # pi / 2

    def __init__(self, mini: object) -> None:
        media = getattr(mini, "media", None)
        if media is None:
            raise RuntimeError(
                "ReachyMini handle has no .media surface; was the app started with "
                "a media backend? (request_media_backend must not be 'no_media')"
            )
        for name in ("get_audio_sample", "push_audio_sample", "get_DoA",
                     "start_recording", "start_playing",
                     "get_input_audio_samplerate", "get_output_audio_samplerate"):
            if not callable(getattr(media, name, None)):
                raise RuntimeError(f"mini.media lacks {name}(); SDK contract changed, re-audit")
        self._mini = mini
        self._media = media
        # SDK pipelines are lazy: without these calls, reads return None forever
        # and pushed frames are dropped with only a log warning (audit F5/F6).
        media.start_recording()
        media.start_playing()
        self.output_sample_rate = int(media.get_output_audio_samplerate())
        self.input_sample_rate = int(media.get_input_audio_samplerate())

    def get_audio_sample(self) -> np.ndarray:
        sample = self._media.get_audio_sample()
        if sample is None:  # no data ready yet; the Protocol promises an empty frame
            return np.zeros(0, dtype=np.float32)
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr

    def push_audio_sample(self, frame: np.ndarray) -> None:
        # Mono float32 at output_sample_rate; the SDK MediaManager expands channels.
        self._media.push_audio_sample(np.asarray(frame, dtype=np.float32))

    def flush(self) -> None:
        audio = getattr(self._media, "audio", None)
        clear = getattr(audio, "clear_player", None)
        if callable(clear):
            clear()

    def get_DoA(self) -> tuple[float, bool] | None:
        reading = self._media.get_DoA()
        if reading is None:
            return None
        angle_rad, speech = reading
        deg = np.degrees(self._DOA_FRONT_RAD - float(angle_rad))
        return (float(deg), bool(speech))

    def close(self) -> None:  # pragma: no cover - hardware path
        for name in ("stop_playing", "stop_recording"):
            stop = getattr(self._media, name, None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass


def make_audio_io(backend: str, **kwargs: object) -> AudioIO:
    """Factory. backend in {"fake", "sounddevice", "reachy"}."""
    if backend == "fake":
        return FakeAudioIO()
    if backend == "sounddevice":
        return SoundDeviceIO(**kwargs)  # type: ignore[arg-type]
    if backend == "reachy":
        return ReachyMediaIO(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown audio backend: {backend!r}")
