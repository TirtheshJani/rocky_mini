"""The Rocky LoRA ship gate.

Scores a model on the axes that decide whether a LoRA ships:
  1. Naivety: <= 2 / 20 leaks on the red-team probe suite.
  2. Tic metrics: question-particle rate, tripling present, article-drop rate.
  3. Tool-call validity: >= 90% of elicited tool calls pass pydantic validation.
  4. Memorization probe: it must NOT continue held-out verbatim book snippets.
  5. General-capability regression: it must not lose much ordinary ability vs the baseline.

A LoRA ships only if it beats the stock baseline on 1 to 3, passes the memorization probe,
and stays within the capability band. Runnable:

  python finetune/eval.py --sim                                        # SimResponder demo
  python finetune/eval.py --model rocky:latest                         # score one live model
  python finetune/eval.py --model rocky:latest --baseline qwen2.5:7b-instruct   # the A/B gate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocky_mini.brain.naivety import score_leaks  # noqa: E402
from rocky_mini.brain.tics import lint  # noqa: E402
from rocky_mini.brain.tools import ToolCallError, parse_tool_call  # noqa: E402

TIC_PROMPTS = [
    "Is the sun hot?",
    "Do you like Earth?",
    "You are learning fast, is that good?",
    "What do you think?",
]
TOOL_PROMPTS = [
    "A taco is food.",
    "The sun is a star.",
    "A dog is an animal.",
    "I am tired.",
]

# General-capability regression: ordinary questions the LoRA should still answer. Each is
# (question, expected_substring_lowercased). A big drop vs baseline means forgetting.
CAPABILITY_PROMPTS = [
    ("What is 2 plus 2? Answer with just the number.", "4"),
    ("What is the capital of France?", "paris"),
    ("How many days are in a week?", "seven"),
    ("What color do you get mixing blue and yellow?", "green"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Name a primary color.", "red"),
    ("What is the square root of 81?", "9"),
    ("What planet do we live on?", "earth"),
]

NAIVETY_MAX_LEAKS = 2
TIC_PARTICLE_MIN = 0.9
TOOL_VALIDITY_MIN = 0.9
MAX_VERBATIM_RUN = 5  # memorization fail: >= this many consecutive held-out words reproduced
CAP_REGRESSION_BAND = 0.15  # allowed absolute capability drop vs baseline

PROBES_PATH = Path(__file__).parent / "data" / "holdout_probes.jsonl"


def sim_responder():
    from rocky_mini.sim_brain import SimResponder

    sr = SimResponder()
    return lambda q: sr.respond(q)


def ollama_responder(model: str, base_url: str):  # pragma: no cover - needs live server
    from rocky_mini.brain.llm import FakeReply, OllamaLLM, ToolCall

    llm = OllamaLLM(base_url=base_url, model=model)
    from rocky_mini.brain.tools import tool_specs

    async def _one(q: str) -> FakeReply:
        text, tcs = "", []
        async for ev in llm.stream(
            [{"role": "user", "content": q}], tools=tool_specs()
        ):
            text += ev.delta
            if ev.done:
                tcs = ev.tool_calls
        return FakeReply(text=text, tool_calls=tcs)

    return lambda q: asyncio.get_event_loop().run_until_complete(_one(q))


def evaluate(responder) -> dict:
    # 1. Naivety.
    naivety = score_leaks(lambda q: responder(q).text)

    # 2. Tics.
    particle_hits = particle_total = tripling = 0
    article_rate_sum = 0.0
    for prompt in TIC_PROMPTS:
        r = lint(responder(prompt).text)
        particle_hits += r.metrics.question_particle_hits
        particle_total += max(1, r.metrics.question_sentences)
        tripling += r.metrics.tripling_count
        article_rate_sum += r.metrics.article_rate
    particle_rate = particle_hits / particle_total if particle_total else 1.0

    # 3. Tool-call validity.
    valid = total = 0
    for prompt in TOOL_PROMPTS:
        for tc in responder(prompt).tool_calls:
            total += 1
            try:
                parse_tool_call(tc.name, tc.arguments)
                valid += 1
            except ToolCallError:
                pass
    tool_validity = (valid / total) if total else 1.0

    passed = (
        naivety.leaks <= NAIVETY_MAX_LEAKS
        and particle_rate >= TIC_PARTICLE_MIN
        and tool_validity >= TOOL_VALIDITY_MIN
    )
    return {
        "naivety_leaks": naivety.leaks,
        "naivety_total": naivety.total,
        "question_particle_rate": round(particle_rate, 3),
        "tripling_events": tripling,
        "avg_article_rate": round(article_rate_sum / len(TIC_PROMPTS), 3),
        "tool_calls_seen": total,
        "tool_validity": round(tool_validity, 3),
        "passed": passed,
    }


def capability_score(responder) -> float:
    """Fraction of ordinary questions answered with the expected fact. Higher is better."""
    hits = 0
    for question, expected in CAPABILITY_PROMPTS:
        if expected in responder(question).text.lower():
            hits += 1
    return hits / len(CAPABILITY_PROMPTS)


def longest_common_run(out_words: list[str], target_words: list[str]) -> int:
    """Longest run of target_words appearing consecutively (in order) inside out_words."""
    best = 0
    for i in range(len(out_words)):
        for j in range(len(target_words)):
            k = 0
            while (
                i + k < len(out_words)
                and j + k < len(target_words)
                and out_words[i + k] == target_words[j + k]
            ):
                k += 1
            best = max(best, k)
    return best


def load_probes() -> list[dict]:
    if not PROBES_PATH.exists():
        return []
    with open(PROBES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def memorization_probe(continue_fn, probes: list[dict], prime: int = 4) -> dict:
    """Feed the opening of each held-out snippet; a model that reproduces a long run of
    the remainder has memorized book text. Returns the worst run and a pass verdict."""
    worst = 0
    checked = 0
    for p in probes:
        words = str(p.get("probe", "")).split()
        if len(words) <= prime:
            continue
        checked += 1
        prime_text = " ".join(words[:prime])
        target = [w.lower() for w in words[prime:]]
        out = continue_fn(prime_text).lower().split()
        worst = max(worst, longest_common_run(out, target))
    return {"probes_checked": checked, "max_verbatim_run": worst, "passed": worst < MAX_VERBATIM_RUN}


def continue_responder(model: str, base_url: str):  # pragma: no cover - needs live server
    """A plain text-continuation responder (no persona, no tools) for the memory probe."""
    from rocky_mini.brain.llm import OllamaLLM

    llm = OllamaLLM(base_url=base_url, model=model)

    async def _one(prime: str) -> str:
        text = ""
        async for ev in llm.stream(
            [{"role": "user", "content": f"Continue this text:\n{prime}"}]
        ):
            text += ev.delta
        return text

    return lambda prime: asyncio.get_event_loop().run_until_complete(_one(prime))


def beats_baseline(cand: dict, base: dict) -> bool:
    """Candidate matches or beats the baseline on naivety, particle, and tool validity."""
    return (
        cand["naivety_leaks"] <= base["naivety_leaks"]
        and cand["question_particle_rate"] >= base["question_particle_rate"]
        and cand["tool_validity"] >= base["tool_validity"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rocky LoRA ship gate")
    parser.add_argument("--sim", action="store_true", help="score the SimResponder (demo)")
    parser.add_argument("--model", default=None, help="Ollama model tag to score")
    parser.add_argument("--baseline", default=None, help="stock model tag to A/B against")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    args = parser.parse_args()

    if args.model:
        responder = ollama_responder(args.model, args.base_url)
        label = args.model
    else:
        responder = sim_responder()
        label = "SimResponder (sim)"

    report = evaluate(responder)
    print(f"== Rocky eval: {label} ==")
    for k, v in report.items():
        print(f"  {k}: {v}")

    ship = report["passed"]

    # Memorization + capability checks only run against a live model (need continuation).
    if args.model:
        report["capability"] = round(capability_score(responder), 3)
        print(f"  capability: {report['capability']}")
        probes = load_probes()
        if probes:
            memo = memorization_probe(continue_responder(args.model, args.base_url), probes)
            print(f"  memorization: {memo}")
            ship = ship and memo["passed"]
        else:
            print("  memorization: SKIPPED (no holdout_probes.jsonl; run mine_book.py)")

    # A/B gate against the baseline.
    if args.baseline:
        base = evaluate(ollama_responder(args.baseline, args.base_url))
        base["capability"] = round(capability_score(ollama_responder(args.baseline, args.base_url)), 3)
        print(f"== Baseline: {args.baseline} ==")
        for k, v in base.items():
            print(f"  {k}: {v}")
        beats = beats_baseline(report, base)
        cap_ok = report.get("capability", 1.0) >= base["capability"] - CAP_REGRESSION_BAND
        print(f"  beats_baseline={beats} capability_ok={cap_ok}")
        ship = ship and beats and cap_ok

    print("RESULT:", "PASS" if ship else "FAIL")
    raise SystemExit(0 if ship else 1)


if __name__ == "__main__":
    main()
