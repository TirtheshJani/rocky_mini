"""Synthetic generator QC, driven by a fake generator (no Ollama, no GPU).

Verifies the off-novel synth path only emits in-character, non-leaking, particle-correct
rows, and that the training system prompt is the byte-stable blended prefix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))

import synth  # noqa: E402
from rocky_mini.brain.persona import build_system_prompt  # noqa: E402


def test_clean_replies_are_kept_with_blended_system_prompt():
    fake = lambda messages, seed: "Rocky fix. Rocky fix. Show Rocky the broken thing."
    records, stats = synth.generate_dataset(fake, per_prompt=1, seed=42)
    assert stats["kept"] == len(synth.TOPICS)
    assert stats["leak"] == 0
    # System prompt on each record is the byte-stable prefix for that row's stage.
    for rec in records:
        stage_prompt = rec["messages"][0]["content"]
        assert stage_prompt in {
            build_system_prompt(s) for s in (0, 1, 2)
        }
        assert rec["messages"][-1]["role"] == "assistant"


def test_leaky_replies_are_dropped():
    fake = lambda messages, seed: "The capital of Italy is Rome and the sky is blue."
    records, stats = synth.generate_dataset(fake, per_prompt=1, seed=42)
    assert records == []
    assert stats["kept"] == 0
    assert stats["leak"] == stats["asked"]


def test_particle_is_normalized_on_generated_questions():
    fake = lambda messages, seed: "You want Rocky to help?"
    records, stats = synth.generate_dataset(fake, per_prompt=1, seed=42)
    assert stats["kept"] == len(synth.TOPICS)
    assert all(r["messages"][-1]["content"].endswith("question?") for r in records)


def test_per_prompt_multiplies_calls():
    fake = lambda messages, seed: "Rocky here. Rocky help."
    _, stats = synth.generate_dataset(fake, per_prompt=3, seed=1)
    assert stats["asked"] == len(synth.TOPICS) * 3
