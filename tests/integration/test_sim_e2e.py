"""Sim end-to-end integration: a typed turn drives the whole pipeline with Fakes.

Asserts the plan's honest checks: a typed turn produces pushed audio; steady-state
turns show KV-cache prefix reuse (small prompt_eval_count); honest p50 latency is under
the 2.5 s budget; and a taught fact survives a simulated restart.
"""

import statistics

import pytest

from rocky_mini.audio.io import FakeAudioIO
from rocky_mini.audio.output import Mixer
from rocky_mini.brain.auditor import NaivetyAuditor
from rocky_mini.brain.curiosity import CuriosityScheduler
from rocky_mini.brain.llm import FakeLLM
from rocky_mini.brain.tools import ToolExecutor
from rocky_mini.config import load_settings
from rocky_mini.memory.store import MemoryStore
from rocky_mini.sim_brain import SimResponder
from rocky_mini.speech.tts import FakeTTS
from rocky_mini.turn import ConversationLoop


def build_loop(home):
    io = FakeAudioIO()
    mixer = Mixer(io, sample_rate=22050)
    memory = MemoryStore(home)
    handlers = {
        "remember_fact": lambda a: memory.remember_fact(a.category, a.fact, a.source_quote),
        "note_open_question": lambda a: memory.note_open_question(a.question),
        "confirm_fact": lambda a: memory.confirm_fact(a.id),
        "set_sleep_watch": lambda a: ("on" if a.on else "off"),
    }
    loop = ConversationLoop(
        llm=FakeLLM(SimResponder()),
        tts=FakeTTS(sample_rate=22050),
        mixer=mixer,
        memory=memory,
        tools=ToolExecutor(handlers=handlers),
        auditor=NaivetyAuditor(),
        curiosity=CuriosityScheduler(),
    )
    return loop, io, mixer, memory


@pytest.mark.asyncio
async def test_typed_turn_produces_pushed_audio(tmp_path):
    loop, io, mixer, memory = build_loop(tmp_path)
    result = await loop.handle_text_turn("a taco is food")
    mixer.render()
    assert len(io.pushed) > 0
    assert result.metrics.time_to_first_audio_s is not None


@pytest.mark.asyncio
async def test_steady_state_prefix_reuse(tmp_path):
    loop, io, mixer, memory = build_loop(tmp_path)
    r1 = await loop.handle_text_turn("hello")
    r2 = await loop.handle_text_turn("tell me more")
    r3 = await loop.handle_text_turn("understand")
    # After the first turn, the byte-stable persona prefix is cached: later turns
    # re-ingest far fewer prompt tokens than the first.
    assert r1.metrics.prompt_eval_count > r2.metrics.prompt_eval_count
    assert r3.metrics.prompt_eval_count < r1.metrics.prompt_eval_count


@pytest.mark.asyncio
async def test_honest_p50_latency_under_budget(tmp_path):
    loop, io, mixer, memory = build_loop(tmp_path)
    budget = load_settings(home_dir=tmp_path).latency_p50_budget_s
    ttfas = []
    for msg in ["hello", "a taco is food", "what is rain?", "understand", "a dog is animal"]:
        r = await loop.handle_text_turn(msg)
        ttfas.append(r.metrics.time_to_first_audio_s)
    p50 = statistics.median(ttfas)
    assert p50 <= budget


@pytest.mark.asyncio
async def test_taught_fact_survives_restart(tmp_path):
    loop, io, mixer, memory = build_loop(tmp_path)
    await loop.handle_text_turn("a taco is food")
    # Simulate an app restart: a brand new loop over the same home dir.
    loop2, io2, mixer2, memory2 = build_loop(tmp_path)
    assert any(f.text == "taco is food" for f in memory2.active_facts())
