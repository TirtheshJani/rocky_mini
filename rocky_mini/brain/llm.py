"""LLM seam: streaming chat behind a Protocol.

The sim/test path uses FakeLLM (scripted, no network). The hardware path uses
OllamaLLM (the openai client pointed at Ollama). FakeLLM also simulates Ollama's
KV-cache prefix reuse: when the byte-stable system prefix is unchanged from the
previous call, it reports a small prompt_eval_count, so the integration test can
assert that steady-state turns do not re-ingest the whole persona.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Protocol


@dataclass
class ToolCall:
    name: str
    arguments: str  # JSON string, as the OpenAI/Ollama API returns it.


@dataclass
class StreamEvent:
    delta: str = ""
    done: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_eval_count: int = 0  # prompt tokens actually ingested (small => cache reuse).
    eval_count: int = 0  # generated tokens.


class LLM(Protocol):
    def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[StreamEvent]:
        ...


@dataclass
class FakeReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def _tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class FakeLLM:
    """Scripted LLM for tests and sim.

    responder: either a fixed FakeReply, a list consumed in order, or a callable
    mapping the message list to a FakeReply.
    """

    def __init__(
        self,
        responder: FakeReply | list[FakeReply] | Callable[[list[dict]], FakeReply],
    ) -> None:
        self._responder = responder
        self._queue: list[FakeReply] = list(responder) if isinstance(responder, list) else []
        self._prev_prefix: str | None = None
        self.calls = 0

    def _next_reply(self, messages: list[dict]) -> FakeReply:
        if isinstance(self._responder, FakeReply):
            return self._responder
        if isinstance(self._responder, list):
            if self._queue:
                return self._queue.pop(0)
            return FakeReply(text="Understand.")
        return self._responder(messages)

    async def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        reply = self._next_reply(messages)

        prefix = messages[0]["content"] if messages else ""
        full_prompt = "".join(m.get("content", "") for m in messages)
        if prefix == self._prev_prefix:
            # Prefix cached: only the non-prefix (new) tokens are ingested.
            new_part = "".join(m.get("content", "") for m in messages[1:])
            prompt_eval = _tokens(new_part)
        else:
            prompt_eval = _tokens(full_prompt)
        self._prev_prefix = prefix

        # Stream the text word by word.
        words = reply.text.split(" ")
        for i, w in enumerate(words):
            piece = w if i == len(words) - 1 else w + " "
            if piece:
                yield StreamEvent(delta=piece)
        yield StreamEvent(
            done=True,
            tool_calls=list(reply.tool_calls),
            prompt_eval_count=prompt_eval,
            eval_count=_tokens(reply.text),
        )


class OllamaLLM:
    """Real streaming client: the openai library pointed at Ollama. Lazy import."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        keep_alive: int = -1,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "openai is not installed. Install with: pip install 'rocky_mini[llm]'"
            ) from exc
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._keep_alive = keep_alive
        # None keeps the served model's own sampling settings (runtime path).
        # The eval gate pins temperature=0 and a seed so PASS/FAIL is reproducible.
        self._temperature = temperature
        self._seed = seed

    async def stream(  # pragma: no cover - requires a live Ollama server
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"keep_alive": self._keep_alive},
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._seed is not None:
            kwargs["seed"] = self._seed
        if tools:
            kwargs["tools"] = tools
        tool_acc: dict[int, dict] = {}
        prompt_eval = 0
        eval_count = 0
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.usage is not None:
                prompt_eval = chunk.usage.prompt_tokens or prompt_eval
                eval_count = chunk.usage.completion_tokens or eval_count
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield StreamEvent(delta=delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = tool_acc.setdefault(tc.index, {"name": "", "arguments": ""})
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        tool_calls = [
            ToolCall(name=v["name"], arguments=v["arguments"])
            for v in tool_acc.values()
            if v["name"]
        ]
        yield StreamEvent(
            done=True,
            tool_calls=tool_calls,
            prompt_eval_count=prompt_eval,
            eval_count=eval_count,
        )
