"""Assemble the final training set from all sources, with QC.

Inputs (whichever exist under finetune/data/):
  - authored.jsonl        persona rows, hand-authored gold (make_seed.py)
  - mined_candidates.jsonl persona rows, paraphrased book lines (mine_book.py)
  - synth_candidates.jsonl persona rows, synthetic off-novel Q&A (synth.py)
  - neutral.jsonl          neutral rows, ordinary helpful turns (neutral.py)

Persona rows pass a QC filter that REUSES the runtime character utilities so the dataset
can only contain in-character, non-leaking, tool-valid turns:
  - no naivety leak (rocky_mini.brain.naivety.is_leak across every probe's markers),
  - the ', question?' particle on question sentences (rocky_mini.brain.tics.lint),
  - every tool call validates (rocky_mini.brain.tools.parse_tool_call),
  - a length cap.
Neutral rows are exempt by design (they are supposed to state ordinary Earth facts).

Outputs:
  - rocky_dialogues.jsonl   the final training set (persona-train + neutral)
  - eval_prompts.jsonl      a held-out set of off-persona-topic user prompts (never trained)
  - stats printed to stdout

The memorization-probe file (holdout_probes.jsonl) is produced by mine_book.py and is
left untouched here; it must never enter training.

Run:  python finetune/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocky_mini.brain.naivety import PROBES, is_leak  # noqa: E402
from rocky_mini.brain.tics import lint  # noqa: E402
from rocky_mini.brain.tools import ToolCallError, parse_tool_call  # noqa: E402

DATA = Path(__file__).parent / "data"
PERSONA_SOURCES = ("authored.jsonl", "mined_candidates.jsonl", "synth_candidates.jsonl")
NEUTRAL_SOURCE = "neutral.jsonl"
OUT_TRAIN = DATA / "rocky_dialogues.jsonl"
OUT_EVAL = DATA / "eval_prompts.jsonl"

MAX_ASSISTANT_WORDS = 80
FUZZY_THRESHOLD = 0.92


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _msg(rec: dict, role: str) -> dict | None:
    for m in rec.get("messages", []):
        if m.get("role") == role:
            return m
    return None


def assistant_text(rec: dict) -> str:
    m = _msg(rec, "assistant")
    return (m or {}).get("content", "") or ""


def user_text(rec: dict) -> str:
    m = _msg(rec, "user")
    return (m or {}).get("content", "") or ""


def is_leaky(text: str) -> bool:
    """True if the reply asserts any probe's Earth fact without a curiosity frame."""
    return any(is_leak(text, p.leak_markers) for p in PROBES)


_PARTICLE_END = re.compile(r",\s*question\s*\?$", re.IGNORECASE)


def _missing_particle(text: str) -> bool:
    """True if any question sentence in the RAW text lacks the ', question?' particle.

    Checked on raw text (not lint output): lint force-adds the particle and would always
    report full coverage, so it cannot detect an authoring miss.
    """
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        s = sent.strip()
        if s.endswith("?") and not _PARTICLE_END.search(s):
            return True
    return False


def _word_count(text: str) -> int:
    return len(text.split())


def qc_persona(rec: dict, normalize: bool) -> tuple[dict | None, str]:
    """Return (kept_record_or_None, reason). normalize=True lints machine-made text."""
    a = _msg(rec, "assistant")
    if a is None or not (a.get("content") or "").strip():
        return None, "empty-assistant"
    text = a["content"]
    if is_leaky(text):
        return None, "naivety-leak"
    for tc in a.get("tool_calls", []):
        fn = tc.get("function", {})
        try:
            parse_tool_call(fn.get("name", ""), fn.get("arguments", ""))
        except ToolCallError:
            return None, "bad-tool-call"
    if normalize:
        # Machine-made text: enforce the particle and strip assistant-isms.
        text = lint(text).text
    elif _missing_particle(text):
        # Authored text should already carry the particle; flag if not.
        return None, "missing-particle"
    if _word_count(text) > MAX_ASSISTANT_WORDS:
        return None, "too-long"
    a["content"] = text
    return rec, "ok"


def _key(rec: dict) -> str:
    return (user_text(rec).strip().lower() + " || " + assistant_text(rec).strip().lower())


def dedup(recs: list[dict]) -> tuple[list[dict], int]:
    """Exact dedup on (user, assistant), then fuzzy on assistant text."""
    kept: list[dict] = []
    seen: set[str] = set()
    removed = 0
    for rec in recs:
        k = _key(rec)
        if k in seen:
            removed += 1
            continue
        a = assistant_text(rec).lower()
        if any(SequenceMatcher(None, a, assistant_text(x).lower()).ratio() >= FUZZY_THRESHOLD for x in kept):
            removed += 1
            continue
        seen.add(k)
        kept.append(rec)
    return kept, removed


def split_eval(persona: list[dict], frac: float, cap: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministically hold out a fraction of persona rows as an eval set."""
    n = min(cap, round(frac * len(persona)))
    if n <= 0:
        return persona, []
    order = sorted(range(len(persona)), key=lambda i: _stable_hash(_key(persona[i]), seed))
    hold = set(order[:n])
    train = [r for i, r in enumerate(persona) if i not in hold]
    evalset = [persona[i] for i in order[:n]]
    return train, evalset


def _stable_hash(text: str, seed: int) -> int:
    h = seed & 0xFFFFFFFF
    for ch in text:
        h = (h * 1000003 + ord(ch)) & 0xFFFFFFFF
    return h


def build(frac: float = 0.1, cap: int = 40, seed: int = 42) -> dict:
    persona_in: list[tuple[dict, str]] = []
    counts: dict[str, dict] = {}
    for name in PERSONA_SOURCES:
        recs = load_jsonl(DATA / name)
        counts[name] = {"in": len(recs), "kept": 0, "rejected": {}}
        normalize = name != "authored.jsonl"
        for rec in recs:
            kept, reason = qc_persona(rec, normalize=normalize)
            if kept is None:
                counts[name]["rejected"][reason] = counts[name]["rejected"].get(reason, 0) + 1
            else:
                persona_in.append((kept, name))
                counts[name]["kept"] += 1

    persona_recs = [r for r, _ in persona_in]
    persona_recs, dup_removed = dedup(persona_recs)
    train_persona, eval_persona = split_eval(persona_recs, frac, cap, seed)

    neutral = load_jsonl(DATA / NEUTRAL_SOURCE)
    counts[NEUTRAL_SOURCE] = {"in": len(neutral), "kept": len(neutral), "rejected": {}}

    training = train_persona + neutral
    return {
        "counts": counts,
        "dup_removed": dup_removed,
        "training": training,
        "eval_prompts": eval_persona,
        "persona_train": len(train_persona),
        "neutral": len(neutral),
    }


def write_outputs(result: dict) -> None:
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for rec in result["training"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for rec in result["eval_prompts"]:
            f.write(json.dumps({"user": user_text(rec)}, ensure_ascii=False) + "\n")


def print_stats(result: dict) -> None:
    print("== dataset build ==")
    for name, c in result["counts"].items():
        rej = ", ".join(f"{k}:{v}" for k, v in c["rejected"].items()) or "none"
        print(f"  {name:24} in={c['in']:4d} kept={c['kept']:4d} rejected=[{rej}]")
    total = len(result["training"])
    neutral = result["neutral"]
    share = (neutral / total * 100) if total else 0.0
    print(f"  fuzzy/exact duplicates removed: {result['dup_removed']}")
    print(f"  eval prompts held out: {len(result['eval_prompts'])}")
    print(f"  TRAINING total: {total}  (persona {result['persona_train']}, neutral {neutral} = {share:.0f}%)")
    probes = DATA / "holdout_probes.jsonl"
    print(f"  memorization-probe file present: {probes.exists()} ({probes.name}, never trained)")
    if total < 250:
        print(f"  NOTE: {total} < 250 target. Run synth.py (and mine_book.py) on the GPU box to reach 250 to 350.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-frac", type=float, default=0.1)
    parser.add_argument("--eval-cap", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build(frac=args.eval_frac, cap=args.eval_cap, seed=args.seed)
    write_outputs(result)
    print_stats(result)


if __name__ == "__main__":
    main()
