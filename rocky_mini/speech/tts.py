"""Text-to-speech seam.

Real synthesis is Piper (en_US-lessac-medium, 22.05 kHz) on the PC, reached over HTTP
at the brain server's /tts endpoint. The chunker feeds one sentence at a time; the
returned PCM is ring-modulated and mixed downstream. FakeTTS renders a deterministic
placeholder tone so the whole pipeline can be exercised with no model. A canned-WAV
fallback covers "brain server down".
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Protocol

import numpy as np


class TTS(Protocol):
    async def synthesize(self, text: str) -> np.ndarray:
        """Return mono float32 PCM at self.sample_rate."""
        ...


class FakeTTS:
    """Deterministic placeholder voice: a soft tone whose length tracks the text.

    Not meant to sound like speech; meant to give the Mixer real, testable audio
    so the pipeline (chunker -> TTS -> ring-mod -> Mixer -> push) can be verified.
    """

    def __init__(self, sample_rate: int = 22050, per_char_s: float = 0.04) -> None:
        self.sample_rate = sample_rate
        self.per_char_s = per_char_s

    async def synthesize(self, text: str) -> np.ndarray:
        n_chars = max(1, len(text.strip()))
        n = int(n_chars * self.per_char_s * self.sample_rate)
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        # A gentle two-tone so it is clearly non-silent but not harsh.
        wave_ = 0.3 * np.sin(2 * np.pi * 180 * t) + 0.15 * np.sin(2 * np.pi * 260 * t)
        # Short fade in/out to avoid clicks.
        fade = min(n, int(0.01 * self.sample_rate))
        if fade > 0:
            wave_[:fade] *= np.linspace(0, 1, fade)
            wave_[-fade:] *= np.linspace(1, 0, fade)
        return wave_.astype(np.float32)


class CannedTTS:
    """Plays in-character WAVs from assets/canned when the brain server is down."""

    def __init__(self, canned_dir: Path, sample_rate: int = 22050) -> None:
        self.canned_dir = Path(canned_dir)
        self.sample_rate = sample_rate

    def _load_wav(self, name: str) -> np.ndarray:
        path = self.canned_dir / name
        if not path.exists():
            return np.zeros(0, dtype=np.float32)
        with wave.open(str(path), "rb") as w:
            frames = w.readframes(w.getnframes())
            arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return arr

    async def synthesize(self, text: str) -> np.ndarray:
        # Text is ignored; the caller picks the canned clip by name via `text`.
        return self._load_wav(text if text.endswith(".wav") else "signal_bad.wav")


class RemoteTTS:
    """POSTs text to the brain server /tts (Piper). Lazy httpx import."""

    def __init__(
        self, base_url: str, sample_rate: int = 22050, timeout_s: float = 15.0, token: str = ""
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "httpx is not installed. Install with: pip install 'rocky_mini[llm]'"
            ) from exc
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.sample_rate = sample_rate
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, headers=headers)

    async def synthesize(self, text: str) -> np.ndarray:  # pragma: no cover - live server
        resp = await self._client.post("/tts", json={"text": text})
        resp.raise_for_status()
        pcm16 = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768.0
        return pcm16

    async def aclose(self) -> None:  # pragma: no cover
        await self._client.aclose()
