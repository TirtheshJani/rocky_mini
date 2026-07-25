"""Image embedder: turn a camera JPEG into a vector Rocky can match by sight.

Visual object memory (decision 21) needs a frozen embedding of what the camera sees, so
a taught object can be re-identified later by cosine similarity. Ollama has no
image-embedding API, so the real embedder is a thin HTTP client to the speech server's
/embed_image endpoint (DINOv2 on the PC); nothing heavy lives in the robot app path.

Sim-first: FakeImageEmbedder carries the whole path with no I/O, so the sim and test
path never touches the network (mirroring vision/observe.py and speech/tts.py).
HttpImageEmbedder lazily imports httpx inside embed(), so importing this module pulls no
network stack. Any error, timeout, or missing frame degrades to None rather than raising,
so a look-around or a teach never breaks a turn.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

# Fake embedding dimension. Kept small and independent of the server's DINOv2 width
# (384): the Fake only needs to be deterministic and separable, not compatible.
_FAKE_DIM = 64


class ImageEmbedder(Protocol):
    def embed(self, jpeg: bytes | None) -> list[float] | None:
        """Return an embedding vector for a JPEG frame, or None if unavailable."""
        ...


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity of two vectors. Pure python (no numpy in the robot app path).

    Returns 0.0 on a length mismatch, an empty vector, or a zero-norm vector, so a
    malformed embedding never raises and never spuriously matches.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class FakeImageEmbedder:
    """Deterministic embedder for sim and tests. No I/O, never raises.

    Maps sha256(jpeg) to a fixed-dim unit vector, so the same frame always embeds to the
    same vector (identical -> cosine 1.0) and unrelated frames land near cosine 0. None
    for a missing frame, mirroring the AudioIO and observer Fakes.
    """

    def __init__(self, dim: int = _FAKE_DIM) -> None:
        self.dim = dim

    def embed(self, jpeg: bytes | None) -> list[float] | None:
        if not jpeg:
            return None
        raw = bytearray()
        counter = 0
        while len(raw) < self.dim:
            raw += hashlib.sha256(jpeg + counter.to_bytes(4, "big")).digest()
            counter += 1
        vals = [(raw[i] - 127.5) / 127.5 for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in vals))
        if norm == 0.0:
            return vals
        return [v / norm for v in vals]


class HttpImageEmbedder:
    """POSTs a JPEG to the speech server's /embed_image endpoint and returns its vector.

    Local only (footgun 8): base_url points at the PC speech server. httpx is imported
    lazily inside embed() so the sim path never pulls it. embed() never raises: a missing
    frame, a missing httpx, a timeout, or any HTTP/parse error all degrade to None, so a
    glance or a teach is fact-only rather than broken.
    """

    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s

    def embed(self, jpeg: bytes | None) -> list[float] | None:
        if not jpeg:
            return None
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "httpx is not installed. It is a core dependency; reinstall with: "
                "pip install -e ."
            ) from exc
        try:
            url = f"{self.base_url.rstrip('/')}/embed_image"
            resp = httpx.post(
                url,
                content=jpeg,
                headers={"Content-Type": "image/jpeg"},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            vector = resp.json().get("vector")
            if not isinstance(vector, list) or not vector:
                return None
            return [float(x) for x in vector]
        except Exception:  # noqa: BLE001 - any fault degrades to None, never raises
            return None
