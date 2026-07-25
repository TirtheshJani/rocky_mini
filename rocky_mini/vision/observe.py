"""Look-around observer: Rocky reads the room through his camera instrument.

In the film Rocky has no eyes; he echolocates. Here the camera is framed as an
instrument Rocky engineered (sonar-adjacent lore, decision 18), not organic sight, so
the look-around tool keeps the alien register. The observation is produced by a LOCAL
Ollama vision model (footgun 8: no paid APIs, no cloud keys); the client here talks to
the same PC Ollama that serves the brain, over its OpenAI-compatible endpoint.

Sim-first: FakeVisionObserver carries the whole path with no I/O, so the sim and test
path never touches the network. OllamaVisionObserver lazily imports httpx inside
observe() (mirroring speech/tts.py RemoteTTS), so importing this module pulls no
network stack. Any error or timeout on the live path degrades to a short in-character
"no reading" line rather than raising, so a look-around never breaks a turn.
"""

from __future__ import annotations

import base64
from typing import Protocol

# In-character fallback when the camera returns nothing or the model call fails.
_NO_READING = "Camera instrument returns no reading."

# Fixed terse instruction for the vision model. Kept plain: the persona layer supplies
# Rocky's voice; this only needs a faithful, short description to hand back.
_OBSERVER_SYSTEM = (
    "You are a camera describing a scene. In one or two plain sentences, state only "
    "what is visibly in front of the camera: people, objects, and lighting. Be concise "
    "and literal. Do not guess, do not add commentary, do not use lists."
)


class VisionObserver(Protocol):
    def observe(self, jpeg: bytes | None, question: str) -> str:
        """Return a short observation of what the camera sees, in plain sentences."""
        ...


class FakeVisionObserver:
    """Deterministic observer for sim and tests. No I/O, never raises.

    Returns a fixed in-character reading so the look-around plumbing has something real
    to carry without a model, mirroring the AudioIO and TTS Fakes.
    """

    def observe(self, jpeg: bytes | None, question: str) -> str:
        if not jpeg:
            return _NO_READING
        return "Camera instrument reads: a room, one human shape, warm light."


class OllamaVisionObserver:
    """Talks to a local Ollama vision model over its OpenAI-compatible endpoint.

    Local only (footgun 8): base_url points at the PC Ollama. httpx is imported lazily
    inside observe() so the sim path never pulls it. observe() never raises: a missing
    frame, a missing httpx, a timeout, or any HTTP/parse error all degrade to the
    in-character "no reading" line.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        timeout_s: float = 20.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def observe(self, jpeg: bytes | None, question: str) -> str:
        if not jpeg:
            return _NO_READING
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "httpx is not installed. It is a core dependency; reinstall with: "
                "pip install -e ."
            ) from exc
        try:
            b64 = base64.b64encode(jpeg).decode("ascii")
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            body = {
                "model": self.model,
                "stream": False,
                "max_tokens": 120,
                "messages": [
                    {"role": "system", "content": _OBSERVER_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question or "What is in front of you?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    },
                ],
            }
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            resp = httpx.post(url, json=body, headers=headers, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            return content or _NO_READING
        except Exception:  # noqa: BLE001 - any fault degrades to an in-character no-op
            return _NO_READING
