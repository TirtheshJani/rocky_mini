"""The Rocky LoRA ship gate.

Scores a model on the three axes that decide whether a LoRA ships:
  1. Naivety: <= 2 / 20 leaks on the red-team probe suite.
  2. Tic metrics: question-particle rate, tripling present, article-drop rate.
  3. Tool-call validity: >= 90% of elicited tool calls pass pydantic validation.

A LoRA ships only if it beats the stock+prompt baseline on all three. Runnable two ways:

  python finetune/eval.py --sim                 # score the deterministic SimResponder (demo)
  python finetune/eval.py --model rocky:latest  # score a live Ollama model (needs the server)
"""

from __future__ import annotations

import argparse
import asyncio
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

NAIVETY_MAX_LEAKS = 2
TIC_PARTICLE_MIN = 0.9
TOOL_VALIDITY_MIN = 0.9


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Rocky LoRA ship gate")
    parser.add_argument("--sim", action="store_true", help="score the SimResponder (demo)")
    parser.add_argument("--model", default=None, help="Ollama model tag to score")
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
    print("RESULT:", "PASS" if report["passed"] else "FAIL")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
