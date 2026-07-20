"""Generate synthetic off-novel Rocky Q&A -> finetune/data/synth_candidates.jsonl.

This is the bulk of the training set, and it teaches the VOICE, not the plot. Rocky
answers a broad, deliberately OFF-NOVEL prompt bank (fixing things, cooking, feelings,
weather, small talk) in the blended voice. Off-novel topics are the single best guard
against overfitting to book content, and they are what makes the LoRA generalize the
mannerisms instead of memorizing passages.

Every generated reply is pre-filtered through the same character utilities the app uses
(naivety leak check, the ', question?' particle, a length cap); build_dataset.py filters
again. The generator is injectable, so this module is unit-tested with a fake and needs
no Ollama for the tests. On the GPU box the default generator calls local Ollama.

Run (GPU box):  python finetune/synth.py --per-prompt 3
Dry run (plan only, no model):  python finetune/synth.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocky_mini.brain.naivety import PROBES, is_leak  # noqa: E402
from rocky_mini.brain.persona import (  # noqa: E402
    STAGE_FLUENT,
    STAGE_LEARNING,
    STAGE_TELEGRAPHIC,
    build_system_prompt,
)
from rocky_mini.brain.tics import lint  # noqa: E402

OUT = Path(__file__).parent / "data" / "synth_candidates.jsonl"
MAX_ASSISTANT_WORDS = 80

Generator = Callable[[list[dict], int], str]

# A few strong, in-voice priming turns (original, not from any source). Kept short.
FEWSHOT: list[tuple[str, str]] = [
    ("My shelf is wobbly.", "Shelf wobbly, question? Rocky fix. Rocky fix. Show Rocky the loose part."),
    ("I feel a little sad today.", "You are sad. Rocky stay close. What make you sad, question? You tell Rocky."),
    ("How do I boil an egg?", "Rocky not eat like you, but Rocky learn. You put egg in hot water, question? How long, question?"),
]

# Off-novel prompt bank: (stage, user_prompt). Everyday topics, nothing about the plot.
TOPICS: list[tuple[int, str]] = [
    (STAGE_TELEGRAPHIC, "My bike tire is flat."),
    (STAGE_TELEGRAPHIC, "I dropped my cup and it broke."),
    (STAGE_TELEGRAPHIC, "The light in my room stopped working."),
    (STAGE_TELEGRAPHIC, "I am cold."),
    (STAGE_TELEGRAPHIC, "I made you a small gift."),
    (STAGE_TELEGRAPHIC, "My chair squeaks when I sit."),
    (STAGE_LEARNING, "I burned my toast this morning."),
    (STAGE_LEARNING, "How do I fix a squeaky door?"),
    (STAGE_LEARNING, "I had a long day at work."),
    (STAGE_LEARNING, "My phone screen cracked."),
    (STAGE_LEARNING, "Can you help me tidy my desk?"),
    (STAGE_LEARNING, "I want to build a small wooden box."),
    (STAGE_LEARNING, "It is raining and I forgot my umbrella."),
    (STAGE_LEARNING, "I planted some seeds today."),
    (STAGE_LEARNING, "My headphones only work on one side."),
    (STAGE_FLUENT, "I am nervous about a big meeting tomorrow."),
    (STAGE_FLUENT, "The kitchen faucet keeps dripping."),
    (STAGE_FLUENT, "I want to learn to cook something new."),
    (STAGE_FLUENT, "My friend and I had an argument."),
    (STAGE_FLUENT, "I finally finished a project I was stuck on."),
    (STAGE_FLUENT, "How would you fix a wobbly table leg?"),
    (STAGE_FLUENT, "I feel lonely tonight."),
    (STAGE_FLUENT, "My computer is running very slow."),
    (STAGE_FLUENT, "I want to make my room feel cozier."),
    (STAGE_FLUENT, "The zipper on my jacket is stuck."),
    (STAGE_FLUENT, "I am proud of something I did today."),
    (STAGE_FLUENT, "My garden hose has a small leak."),
    (STAGE_FLUENT, "I want to organize my tools better."),
    (STAGE_FLUENT, "The wind is really loud outside tonight."),
    (STAGE_FLUENT, "I keep losing my keys."),
]


def build_messages(stage: int, user_prompt: str) -> list[dict]:
    """System = the byte-stable blended prefix; then a few in-voice shots; then the prompt."""
    msgs: list[dict] = [{"role": "system", "content": build_system_prompt(stage)}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def _is_leaky(text: str) -> bool:
    return any(is_leak(text, p.leak_markers) for p in PROBES)


def qc_reply(text: str) -> str | None:
    """QC a generated reply: reject leaks and over-long text; normalize the particle.

    Shared by synth.py and mine_book.py so both candidate sources apply one filter.
    """
    text = (text or "").strip()
    if not text or _is_leaky(text):
        return None
    text = lint(text).text  # enforce the particle, strip assistant-isms
    if not text or len(text.split()) > MAX_ASSISTANT_WORDS:
        return None
    return text


def generate_dataset(generator: Generator, per_prompt: int, seed: int) -> tuple[list[dict], dict]:
    """Produce records via the injectable generator. Returns (records, stats)."""
    records: list[dict] = []
    stats = {"asked": 0, "kept": 0, "leak": 0, "empty_or_long": 0}
    for idx, (stage, prompt) in enumerate(TOPICS):
        for variant in range(per_prompt):
            stats["asked"] += 1
            call_seed = seed + idx * 100 + variant
            raw = generator(build_messages(stage, prompt), call_seed)
            accepted = qc_reply(raw)
            if accepted is None:
                if _is_leaky(raw or ""):
                    stats["leak"] += 1
                else:
                    stats["empty_or_long"] += 1
                continue
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": build_system_prompt(stage)},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": accepted},
                    ]
                }
            )
            stats["kept"] += 1
    return records, stats


def _ollama_generator(model: str, base_url: str, temperature: float) -> Generator:
    """Default generator: local Ollama via the OpenAI-compatible endpoint (lazy httpx)."""
    import httpx  # pragma: no cover - needs the local server

    def generate(messages: list[dict], seed: int) -> str:  # pragma: no cover - needs Ollama
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-prompt", type=int, default=3)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; call no model")
    args = parser.parse_args()

    if args.dry_run:
        print(f"prompts={len(TOPICS)} per_prompt={args.per_prompt} -> {len(TOPICS) * args.per_prompt} calls")
        print(f"model={args.model} base_url={args.base_url} out={OUT}")
        return

    generator = _ollama_generator(args.model, args.base_url, args.temperature)
    records, stats = generate_dataset(generator, args.per_prompt, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"synth: asked={stats['asked']} kept={stats['kept']} "
          f"leak={stats['leak']} empty_or_long={stats['empty_or_long']} -> {OUT}")


if __name__ == "__main__":
    main()
