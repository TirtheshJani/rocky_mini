"""Speech-to-text seam.

Real transcription runs faster-whisper on the PC GPU, reached over HTTP at the brain
server's /stt endpoint. The sim/test path uses FakeSTT (scripted). No model or GPU is
needed to import this module.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class STT(Protocol):
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        ...


class FakeSTT:
    """Returns scripted transcripts. Used for typed-turn sim and tests."""

    def __init__(self, transcripts: list[str] | str | None = None) -> None:
        if isinstance(transcripts, str):
            transcripts = [transcripts]
        self._queue = list(transcripts or [])
        self.default = ""

    def push(self, text: str) -> None:
        self._queue.append(text)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self._queue:
            return self._queue.pop(0)
        return self.default


class RemoteSTT:
    """POSTs audio to the brain server /stt (faster-whisper). Lazy httpx import."""

    def __init__(self, base_url: str, timeout_s: float = 15.0, token: str = "") -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "httpx is not installed. Install with: pip install 'rocky_mini[llm]'"
            ) from exc
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, headers=headers)

    async def transcribe(  # pragma: no cover - requires a live brain server
        self, audio: np.ndarray, sample_rate: int
    ) -> str:
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        resp = await self._client.post(
            "/stt",
            content=pcm16,
            params={"sample_rate": sample_rate},
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    async def aclose(self) -> None:  # pragma: no cover
        await self._client.aclose()
