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

    def get_DoA(self) -> float | None:
        """Direction of arrival in degrees, or None if unavailable (e.g. in sim)."""
        ...

    def close(self) -> None:
        ...


class FakeAudioIO:
    """In-memory AudioIO for tests and sim. Records everything pushed."""

    def __init__(self, doa: float | None = None) -> None:
        self._in: deque[np.ndarray] = deque()
        self.pushed: list[np.ndarray] = []
        self._doa = doa

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

    def get_DoA(self) -> float | None:
        return self._doa

    def set_DoA(self, doa: float | None) -> None:
        self._doa = doa

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

    def get_DoA(self) -> float | None:
        return None  # no mic array on a laptop.

    def close(self) -> None:  # pragma: no cover - hardware path
        for stream in (self._in, self._out):
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


class ReachyMediaIO:
    """Physical-robot backend wrapping mini.media. Lazy: needs the reachy SDK."""

    def __init__(self, mini: object) -> None:
        # mini is a live ReachyMini handle; media is its audio surface.
        self._mini = mini
        self._media = getattr(mini, "media")

    def get_audio_sample(self) -> np.ndarray:  # pragma: no cover - hardware path
        sample = self._media.get_audio_sample()
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr

    def push_audio_sample(self, frame: np.ndarray) -> None:  # pragma: no cover
        self._media.push_audio_sample(np.asarray(frame, dtype=np.float32))

    def get_DoA(self) -> float | None:  # pragma: no cover - hardware path
        getter = getattr(self._mini, "get_DoA", None)
        return float(getter()) if getter else None

    def close(self) -> None:  # pragma: no cover - hardware path
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
