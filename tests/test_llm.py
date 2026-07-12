"""FakeLLM tests: streaming, KV-prefix reuse simulation, tool calls."""

import pytest

from rocky_mini.brain.llm import FakeLLM, FakeReply, ToolCall


async def _collect(stream):
    deltas, done = [], None
    async for ev in stream:
        if ev.delta:
            deltas.append(ev.delta)
        if ev.done:
            done = ev
    return "".join(deltas), done


@pytest.mark.asyncio
async def test_stream_reassembles_text():
    llm = FakeLLM(FakeReply(text="Rocky fix now"))
    text, done = await _collect(llm.stream([{"role": "system", "content": "sys"}]))
    assert text == "Rocky fix now"
    assert done is not None and done.eval_count > 0


@pytest.mark.asyncio
async def test_prefix_reuse_lowers_prompt_eval_count():
    llm = FakeLLM([FakeReply(text="one"), FakeReply(text="two")])
    prefix = "S" * 4000  # a big, byte-stable persona prefix.
    msgs1 = [{"role": "system", "content": prefix}, {"role": "user", "content": "hi"}]
    _, d1 = await _collect(llm.stream(msgs1))
    msgs2 = [{"role": "system", "content": prefix}, {"role": "user", "content": "hi again"}]
    _, d2 = await _collect(llm.stream(msgs2))
    # First call ingests the whole prefix; second reuses it and ingests far fewer tokens.
    assert d1.prompt_eval_count > 500
    assert d2.prompt_eval_count < 50


@pytest.mark.asyncio
async def test_tool_calls_surface_on_done():
    tc = ToolCall(name="remember_fact", arguments='{"category":"food","fact":"x","source_quote":"y"}')
    llm = FakeLLM(FakeReply(text="Understand.", tool_calls=[tc]))
    _, done = await _collect(llm.stream([{"role": "system", "content": "s"}]))
    assert len(done.tool_calls) == 1
    assert done.tool_calls[0].name == "remember_fact"


@pytest.mark.asyncio
async def test_callable_responder_sees_messages():
    seen = {}

    def responder(messages):
        seen["n"] = len(messages)
        return FakeReply(text="ok")

    llm = FakeLLM(responder)
    await _collect(llm.stream([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]))
    assert seen["n"] == 2
