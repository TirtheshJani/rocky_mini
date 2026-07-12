"""Tests for the auditor, curiosity scheduler, and reflection worker."""

import pytest

from rocky_mini.brain.auditor import NaivetyAuditor
from rocky_mini.brain.curiosity import CANON_TARGETS, CuriosityScheduler
from rocky_mini.brain.reflection import ReflectionWorker


# -- auditor ---------------------------------------------------------------
def test_auditor_flags_leak_and_corrects():
    auditor = NaivetyAuditor()
    result = auditor.audit_local("The capital of Italy is Rome.")
    assert result.leaked is True
    assert result.correction is not None
    assert "not taught" in result.correction


def test_auditor_passes_deflection():
    auditor = NaivetyAuditor()
    result = auditor.audit_local("Rome? Rocky not know, question?")
    assert result.leaked is False
    assert result.correction is None


@pytest.mark.asyncio
async def test_auditor_async_local_path():
    auditor = NaivetyAuditor(llm=None)
    result = await auditor.audit("what is capital of italy", "It is Rome.")
    assert result.leaked is True


# -- curiosity -------------------------------------------------------------
def test_scheduler_seeded_with_canon():
    sched = CuriosityScheduler()
    topics = {it.topic for it in sched.items}
    assert set(CANON_TARGETS).issubset(topics)


def test_cadence_cap_one_per_n_turns():
    sched = CuriosityScheduler(cap_every_n_turns=3)
    first = sched.pick()
    assert first is not None  # starts ready
    # Immediately after asking, not eligible.
    assert sched.pick() is None
    sched.note_turn()
    assert sched.pick() is None
    sched.note_turn()
    sched.note_turn()
    assert sched.pick() is not None  # eligible again after 3 turns


def test_satisfy_removes_topic_from_selection():
    sched = CuriosityScheduler(cap_every_n_turns=1)
    sched.satisfy("sleep")
    picked = set()
    for _ in range(len(CANON_TARGETS)):
        t = sched.pick(force=True)
        if t:
            picked.add(t)
    assert "sleep" not in picked


def test_boost_raises_priority():
    sched = CuriosityScheduler()
    sched.boost("lying", amount=10.0)
    assert sched.pick(force=True) == "lying"


# -- reflection ------------------------------------------------------------
@pytest.mark.asyncio
async def test_reflection_local_summary():
    worker = ReflectionWorker()
    refl = await worker.reflect(["taco is food", "sun is hot"])
    assert "taco is food" in refl.summary
    assert refl.follow_ups


@pytest.mark.asyncio
async def test_reflection_empty_session():
    worker = ReflectionWorker()
    refl = await worker.reflect([])
    assert refl.summary
    assert refl.follow_ups == []


def test_greeting_ritual():
    worker = ReflectionWorker()
    greeting = worker.greeting("Rocky learn: taco is food.", ["Is burrito also sandwich, question?"])
    assert greeting.startswith("TJ! You return.")
    assert "taco" in greeting
    assert "burrito" in greeting


def test_greeting_first_time():
    worker = ReflectionWorker()
    greeting = worker.greeting(None)
    assert "Rocky" in greeting
