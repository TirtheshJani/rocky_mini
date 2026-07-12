"""Memory tests: durability across restart, confidence lifecycle, dedup, digest,
fuzzy facts, export/import round-trip."""

from itertools import count

import pytest

from rocky_mini.memory.models import CONFIRMED, HEARD_ONCE, MASTERED
from rocky_mini.memory.store import MemoryStore


def make_clock():
    counter = count(1)
    # Zero-padded so lexical order equals chronological order.
    return lambda: f"{next(counter):06d}"


def test_teach_then_restart_still_knows(tmp_path):
    store = MemoryStore(tmp_path, now=make_clock())
    store.remember_fact("food", "taco is food", "a taco is food")

    # Simulate an app restart: a fresh store over the same dir.
    reloaded = MemoryStore(tmp_path, now=make_clock())
    facts = reloaded.active_facts()
    assert len(facts) == 1
    assert facts[0].text == "taco is food"


def test_dedup_bumps_confidence(tmp_path):
    store = MemoryStore(tmp_path, now=make_clock())
    id1 = store.remember_fact("food", "Taco is food", "q1")
    id2 = store.remember_fact("food", "  taco   is food  ", "q2")  # same fact, noisy text
    assert id1 == id2
    fact = store.get_fact(id1)
    assert fact.heard_count == 2
    assert fact.confidence == CONFIRMED


def test_confirm_lifecycle(tmp_path):
    store = MemoryStore(tmp_path, now=make_clock())
    fid = store.remember_fact("sky", "sun is hot")
    assert store.get_fact(fid).confidence == HEARD_ONCE
    assert store.confirm_fact(fid) == CONFIRMED
    assert store.confirm_fact(fid) == MASTERED
    assert store.confirm_fact(fid) == MASTERED  # saturates


def test_delete_tombstone_survives_reload(tmp_path):
    clock = make_clock()
    store = MemoryStore(tmp_path, now=clock)
    fid = store.remember_fact("x", "temporary fact")
    store.delete_fact(fid)
    assert store.active_facts() == []
    reloaded = MemoryStore(tmp_path, now=make_clock())
    assert reloaded.active_facts() == []


def test_open_question_dedup(tmp_path):
    store = MemoryStore(tmp_path, now=make_clock())
    a = store.note_open_question("What is rain?")
    b = store.note_open_question("what is rain?")
    assert a == b
    assert len(store.active_questions()) == 1


def test_digest_orders_by_confidence(tmp_path):
    store = MemoryStore(tmp_path, now=make_clock())
    store.remember_fact("a", "weak fact")
    strong = store.remember_fact("b", "strong fact")
    store.confirm_fact(strong)
    store.confirm_fact(strong)  # mastered
    digest = store.digest_facts()
    assert digest[0] == "strong fact"  # mastered ranks first


def test_fuzzy_facts_are_heard_once(tmp_path):
    store = MemoryStore(tmp_path, now=make_clock())
    store.remember_fact("a", "fuzzy one")
    store.remember_fact("b", "fuzzy two")
    solid = store.remember_fact("c", "solid")
    store.confirm_fact(solid)
    fuzzy = store.fuzzy_facts(limit=2)
    assert "solid" not in fuzzy
    assert set(fuzzy) == {"fuzzy one", "fuzzy two"}


def test_known_words_counts_active(tmp_path):
    store = MemoryStore(tmp_path, now=make_clock())
    store.remember_fact("a", "one")
    store.remember_fact("b", "two")
    assert store.known_words() == 2


def test_export_import_round_trip(tmp_path):
    src = MemoryStore(tmp_path / "src", now=make_clock())
    src.remember_fact("food", "taco is food")
    src.note_open_question("what is rain?")
    zpath = src.export_zip(tmp_path / "rocky_memory.zip")

    dst = MemoryStore(tmp_path / "dst", now=make_clock())
    assert dst.active_facts() == []
    dst.import_zip(zpath)
    assert [f.text for f in dst.active_facts()] == ["taco is food"]
    assert len(dst.active_questions()) == 1


def test_add_and_get_session(tmp_path):
    store = MemoryStore(tmp_path, now=make_clock())
    store.add_session("taught taco", ["is burrito also sandwich?"])
    last = store.last_session()
    assert last.summary == "taught taco"
    assert last.follow_ups == ["is burrito also sandwich?"]


def test_import_clears_stale_logs(tmp_path):
    # src has facts but no sessions; dst has a session that must be wiped on import.
    src = MemoryStore(tmp_path / "src", now=make_clock())
    src.remember_fact("food", "taco is food")
    zpath = src.export_zip(tmp_path / "snap.zip")

    dst = MemoryStore(tmp_path / "dst", now=make_clock())
    dst.add_session("old session that should vanish")
    dst.import_zip(zpath)
    assert dst.last_session() is None  # stale sessions.jsonl removed
    assert [f.text for f in dst.active_facts()] == ["taco is food"]


def test_active_facts_are_chronological_beyond_ten(tmp_path):
    store = MemoryStore(tmp_path, now=make_clock())
    for i in range(12):
        store.remember_fact("n", f"fact {i}")
    ids = [f.id for f in store.active_facts()]
    # Insertion order, not lexical id order (f10 must not sort before f2).
    assert ids == [f"f{i}" for i in range(1, 13)]
