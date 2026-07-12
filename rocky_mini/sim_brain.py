"""SimResponder: a deterministic rule-based Rocky for sim and the UI chat box.

This is NOT the real brain. The real brain is Qwen2.5 (+ Rocky LoRA) served by Ollama
via OllamaLLM. SimResponder lets the whole app run and demo in character with no model,
no GPU, and no network: it recognizes teaching, questions, greetings, and the sleep
trigger, and emits in-voice replies plus the matching tool calls, so the fact table,
naivety deflection, jazz-hands, and sleep-watch all light up in sim.
"""

from __future__ import annotations

import json
import re

from .brain.llm import FakeReply, ToolCall

_QUESTION_WORDS = ("what", "who", "where", "when", "why", "how", "which", "is ", "are ", "do ")
_ARTICLE_PREFIX = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_TEACH_RE = re.compile(r"^(.*?\b(?:is|are|means|called)\b.*)$", re.IGNORECASE)


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


class SimResponder:
    def __call__(self, messages: list[dict]) -> FakeReply:
        return self.respond(_last_user(messages))

    def respond(self, user: str) -> FakeReply:
        text = user.strip()
        low = text.lower()

        if not text:
            return FakeReply(text="Rocky here. You teach, question?")

        if any(g in low for g in ("hello", "hi ", "hey", "hi rocky")) or low in ("hi", "hey"):
            return FakeReply(text="TJ! Good. Good. Good. You teach Rocky, question?")

        if "tired" in low or "sleep" in low or "rest" in low:
            tc = ToolCall(name="set_sleep_watch", arguments=json.dumps({"on": True}))
            return FakeReply(text="[emote:sleepy] Rocky watch. Watch. Watch. Sleep now.", tool_calls=[tc])

        if low.endswith("?") or low.startswith(_QUESTION_WORDS):
            # Treat as an Earth-knowledge probe: stay naive and deflect.
            return FakeReply(text="New word, question? Rocky not know. You teach Rocky, question?")

        m = _TEACH_RE.match(text)
        if m:
            fact = _ARTICLE_PREFIX.sub("", text).rstrip(".").strip()
            tc = ToolCall(
                name="remember_fact",
                arguments=json.dumps(
                    {"category": "general", "fact": fact, "source_quote": text}
                ),
            )
            return FakeReply(
                text=f"[emote:understand] Understand. Rocky learn: {fact}. Thank.",
                tool_calls=[tc],
            )

        return FakeReply(text="Understand. Tell Rocky more, question?")
