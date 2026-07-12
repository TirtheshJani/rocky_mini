"""ConversationLoop tests: pipeline end-to-end with Fakes, tools, barge-in, audit,
curiosity, and KV-prefix reuse metrics."""

import pytest

from rocky_mini.audio.io import FakeAudioIO
from rocky_mini.audio.output import Mixer
from rocky_mini.brain.auditor import NaivetyAuditor
from rocky_mini.brain.curiosity import CuriosityScheduler
from rocky_mini.brain.llm import FakeLLM, FakeReply, ToolCall
from rocky_mini.brain.tools import ToolExecutor
from rocky_mini.memory.store import MemoryStore
from rocky_mini.speech.tts import FakeTTS
from rocky_mini.turn import ConversationLoop


def build_loop(tmp_path, responder, auditor=None, curiosity=None):
    io = FakeAudioIO()
    mixer = Mixer(io, sample_rate=22050)
    memory = MemoryStore(tmp_path)
    handlers = {
        "remember_fact": lambda a: memory.remember_fact(a.category, a.fact, a.source_quote),
        "note_open_question": lambda a: memory.note_open_question(a.question),
        "confirm_fact": lambda a: memory.confirm_fact(a.id),
        "set_sleep_watch": lambda a: ("on" if a.on else "off"),
    }
    loop = ConversationLoop(
        llm=FakeLLM(responder),
        tts=FakeTTS(sample_rate=22050),
        mixer=mixer,
        memory=memory,
        tools=ToolExecutor(handlers=handlers),
        auditor=auditor or NaivetyAuditor(),
        curiosity=curiosity or CuriosityScheduler(),
    )
    return loop, io, mixer, memory


@pytest.mark.asyncio
async def test_text_turn_produces_reply_and_audio(tmp_path):
    loop, io, mixer, memory = build_loop(tmp_path, FakeReply(text="Thank. Rocky learn."))
    result = await loop.handle_text_turn("Hello Rocky")
    assert result.reply == "Thank. Rocky learn."
    assert len(result.chunks) == 2
    # Voice was pushed to the mixer; render it to the audio device.
    mixer.render()
    assert len(io.pushed) > 0
    assert result.metrics.time_to_first_audio_s is not None


@pytest.mark.asyncio
async def test_remember_fact_tool_persists(tmp_path):
    tc = ToolCall(
        name="remember_fact",
        arguments='{"category":"food","fact":"taco is food","source_quote":"a taco is food"}',
    )
    loop, io, mixer, memory = build_loop(
        tmp_path, FakeReply(text="Taco. Understand.", tool_calls=[tc])
    )
    result = await loop.handle_text_turn("A taco is food")
    assert ("remember_fact", "f1") in result.tool_results
    assert result.metrics.tool_calls == 1
    assert result.metrics.tool_errors == 0
    assert [f.text for f in memory.active_facts()] == ["taco is food"]


@pytest.mark.asyncio
async def test_sleep_watch_tool_sets_state(tmp_path):
    tc = ToolCall(name="set_sleep_watch", arguments='{"on": true}')
    loop, io, mixer, memory = build_loop(
        tmp_path, FakeReply(text="Rocky watch. Sleep now.", tool_calls=[tc])
    )
    await loop.handle_text_turn("I'm tired")
    assert loop.sleep_watch is True
    assert loop.state == "SLEEPWATCH"


@pytest.mark.asyncio
async def test_barge_in_bumps_generation(tmp_path):
    loop, io, mixer, memory = build_loop(tmp_path, FakeReply(text="Long reply here."))
    assert loop.generation == 0
    loop.barge_in()
    assert loop.generation == 1
    assert mixer.generation == 1


@pytest.mark.asyncio
async def test_naivety_correction_injected_next_turn(tmp_path):
    captured = []

    def responder(messages):
        captured.append(messages)
        # First turn leaks; later turns deflect.
        if len(captured) == 1:
            return FakeReply(text="The capital of Italy is Rome.")
        return FakeReply(text="Rocky not know, question?")

    loop, io, mixer, memory = build_loop(tmp_path, responder)
    r1 = await loop.handle_text_turn("what is capital of italy")
    assert r1.audit is not None and r1.audit.leaked is True

    await loop.handle_text_turn("tell me more")
    # The second turn's system messages carry the auditor correction.
    system_msgs = [m["content"] for m in captured[1] if m["role"] == "system"]
    assert any("not taught" in s for s in system_msgs)
    # And it was consumed (not repeated on a third turn).
    assert loop._pending_system == [] or all("not taught" not in s for s in loop._pending_system)


@pytest.mark.asyncio
async def test_kv_prefix_reuse_lowers_prompt_eval(tmp_path):
    loop, io, mixer, memory = build_loop(tmp_path, FakeReply(text="Understand."))
    r1 = await loop.handle_text_turn("hello")
    r2 = await loop.handle_text_turn("hello again")
    # Same byte-stable persona prefix -> second turn re-ingests far fewer prompt tokens.
    assert r1.metrics.prompt_eval_count > r2.metrics.prompt_eval_count


@pytest.mark.asyncio
async def test_curiosity_proactive_after_cadence(tmp_path):
    sched = CuriosityScheduler(cap_every_n_turns=1)
    loop, io, mixer, memory = build_loop(tmp_path, FakeReply(text="Ok."), curiosity=sched)
    result = await loop.handle_text_turn("hi")
    assert result.proactive_topic is not None


@pytest.mark.asyncio
async def test_ack_fires_before_speaking(tmp_path):
    loop, io, mixer, memory = build_loop(tmp_path, FakeReply(text="Hi."))
    result = await loop.handle_text_turn("hello")
    assert result.metrics.ack_ms is not None
    # Ack chord was queued to the mixer before any voice.
    assert mixer.pending()
