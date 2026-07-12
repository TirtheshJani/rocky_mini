"""Persona tests: byte-stability of the prefix, token budget, digest, growth stages."""

from rocky_mini.brain import persona
from rocky_mini.brain.persona import (
    PersonaProfile,
    STAGE_FLUENT,
    STAGE_LEARNING,
    STAGE_TELEGRAPHIC,
)


def test_system_prompt_is_byte_stable_for_same_stage():
    a = persona.build_system_prompt(STAGE_TELEGRAPHIC)
    b = persona.build_system_prompt(STAGE_TELEGRAPHIC)
    assert a == b
    assert isinstance(a, str) and len(a) > 200


def test_system_prompt_differs_by_stage():
    assert persona.build_system_prompt(STAGE_TELEGRAPHIC) != persona.build_system_prompt(STAGE_FLUENT)


def test_build_messages_prefix_is_stable_regardless_of_digest():
    p1 = PersonaProfile(stage=STAGE_TELEGRAPHIC, learned_facts=["Sun is hot"])
    p2 = PersonaProfile(stage=STAGE_TELEGRAPHIC, learned_facts=["Sun is hot", "Water is wet"])
    m1 = persona.build_messages(p1)
    m2 = persona.build_messages(p2)
    # The stable prefix (message 0) is byte-identical; only the digest (message 1) grows.
    assert m1[0]["content"] == m2[0]["content"]
    assert m1[1]["content"] != m2[1]["content"]


def test_digest_reflects_facts_and_fuzzy_and_questions():
    p = PersonaProfile(
        learned_facts=["Dogs bark"],
        fuzzy_facts=["Cats meow"],
        open_questions=["What is rain"],
        known_words=12,
    )
    digest = persona.build_digest(p)
    assert "Dogs bark" in digest
    assert "memory fuzzy" in digest
    assert "What is rain" in digest
    assert "12" in digest


def test_token_budget_under_2000():
    p = PersonaProfile(
        stage=STAGE_FLUENT,
        learned_facts=[f"fact number {i} about the world" for i in range(40)],
        known_words=350,
    )
    msgs = persona.build_messages(p)
    total = sum(persona.estimate_tokens(m["content"]) for m in msgs)
    assert total <= 2000


def test_stage_for_words_boundaries():
    assert persona.stage_for_words(0) == STAGE_TELEGRAPHIC
    assert persona.stage_for_words(59) == STAGE_TELEGRAPHIC
    assert persona.stage_for_words(60) == STAGE_LEARNING
    assert persona.stage_for_words(299) == STAGE_LEARNING
    assert persona.stage_for_words(300) == STAGE_FLUENT


def test_prompt_contains_no_em_dash():
    for stage in (STAGE_TELEGRAPHIC, STAGE_LEARNING, STAGE_FLUENT):
        prompt = persona.build_system_prompt(stage)
        assert "—" not in prompt
        assert "–" not in prompt


def test_epistemic_ledger_is_present():
    prompt = persona.build_system_prompt(STAGE_TELEGRAPHIC)
    assert "EPISTEMIC LEDGER" in prompt
    assert "question?" in prompt
