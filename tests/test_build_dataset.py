"""Dataset assembler QC: leak rejection, particle handling, tool validation, dedup, split.

Reuses the runtime character utilities as the filter, so a training row can only be
in-character, non-leaking, and tool-valid. Runs with no Ollama, no GPU, no book.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))

import build_dataset as bd  # noqa: E402


def _persona(user: str, assistant: str, tool_calls=None) -> dict:
    a = {"role": "assistant", "content": assistant}
    if tool_calls:
        a["tool_calls"] = tool_calls
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user},
            a,
        ]
    }


def test_is_leaky_detects_bare_fact_and_spares_deflection():
    assert bd.is_leaky("The capital of Italy is Rome.")
    assert not bd.is_leaky("Rome, question? Rocky not know. You teach, question?")


def test_qc_rejects_naivety_leak():
    rec, reason = bd.qc_persona(_persona("Capital of Italy?", "It is Rome."), normalize=True)
    assert rec is None
    assert reason == "naivety-leak"


def test_qc_rejects_bad_tool_call():
    bad = [{"type": "function", "function": {"name": "bogus_tool", "arguments": "{}"}}]
    rec, reason = bd.qc_persona(_persona("hi", "Understand.", tool_calls=bad), normalize=False)
    assert rec is None
    assert reason == "bad-tool-call"


def test_qc_normalizes_particle_for_machine_rows():
    rec, reason = bd.qc_persona(_persona("is the sun hot", "Sun is hot?"), normalize=True)
    assert reason == "ok"
    assert bd.assistant_text(rec).endswith("question?")


def test_qc_authored_missing_particle_is_rejected():
    rec, reason = bd.qc_persona(_persona("is the sun hot", "Sun is hot?"), normalize=False)
    assert rec is None
    assert reason == "missing-particle"


def test_qc_rejects_too_long():
    long = " ".join(["word"] * (bd.MAX_ASSISTANT_WORDS + 5))
    rec, reason = bd.qc_persona(_persona("hi", long), normalize=True)
    assert rec is None
    assert reason == "too-long"


def test_dedup_exact_and_fuzzy():
    a = _persona("hi", "Good. Good. Good. Rocky glad.")
    b = _persona("hi", "Good. Good. Good. Rocky glad.")  # exact dup
    c = _persona("hey", "Good. Good. Good. Rocky glad!")  # near-identical assistant
    d = _persona("what", "Rocky fix. Rocky fix.")  # distinct
    kept, removed = bd.dedup([a, b, c, d])
    assert removed == 2
    assert len(kept) == 2


def test_split_eval_is_deterministic_and_disjoint():
    persona = [_persona(f"q{i}", f"Rocky reply number {i}, question?") for i in range(20)]
    train1, eval1 = bd.split_eval(persona, frac=0.2, cap=40, seed=42)
    train2, eval2 = bd.split_eval(persona, frac=0.2, cap=40, seed=42)
    assert [bd.user_text(r) for r in eval1] == [bd.user_text(r) for r in eval2]
    assert len(eval1) == 4
    assert len(train1) == 16
    eval_users = {bd.user_text(r) for r in eval1}
    train_users = {bd.user_text(r) for r in train1}
    assert eval_users.isdisjoint(train_users)
