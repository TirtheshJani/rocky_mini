"""Prove Ollama KV-cache prefix reuse for the byte-stable persona (footgun 4).

The hardware audit left this INCONCLUSIVE: the OpenAI-compatible usage field does not
distinguish cached from freshly-ingested prompt tokens. Neither, it turns out, does the
native /api/chat prompt_eval_count on Ollama 0.32.x: it reports the request's total
prompt size whether or not the prefix came from cache, so a count-based check reads as
a false NO CLEAR REUSE. The first version of this script did exactly that and recorded
a wrong negative result; corrected 2026-07-20.

The trustworthy signal is prompt_eval_duration. Prefill on the 4080 runs at thousands
of tokens per second, so re-ingesting the ~750-token persona prefix costs hundreds of
milliseconds, while a cache hit leaves only the handful of new tokens and the duration
collapses. This script sends two turns sharing the identical byte-stable persona prefix
and compares prompt_eval_duration.

Needs Ollama running with the model pulled. keep_alive=-1 keeps the model (and its
cached prefix) resident between the two calls.

  python scripts/check_kv_reuse.py --model qwen2.5:7b-instruct
  python scripts/check_kv_reuse.py --model rocky:latest

Verified 2026-07-20 on Ollama 0.32.1 for Windows: turn 2 prefill ran at 14% (rocky) and
17% (stock qwen2.5) of turn 1's duration while prompt_eval_count claimed 104%. Reuse is
live; the counter was the artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocky_mini.brain.persona import STAGE_FLUENT, build_system_prompt  # noqa: E402

# Prefill faster than this is not physically a fresh ingest on local hardware; it means
# turn 1 itself hit a warm cache (e.g. this script ran twice against a live server).
WARM_CACHE_TOKENS_PER_S = 20_000.0


def _chat(host: str, model: str, messages: list[dict]) -> dict:  # pragma: no cover - needs Ollama
    import httpx

    resp = httpx.post(
        f"{host}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "keep_alive": -1},
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:  # pragma: no cover - needs a live Ollama
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    system = build_system_prompt(STAGE_FLUENT)  # the byte-stable prefix
    turn1 = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Hello Rocky."},
    ]
    r1 = _chat(args.host, args.model, turn1)

    # Second turn: same system prefix, continued conversation. On a cache hit, only the
    # new tokens are prefilled and prompt_eval_duration collapses.
    turn2 = turn1 + [
        {"role": "assistant", "content": r1["message"]["content"]},
        {"role": "user", "content": "Tell me about tools."},
    ]
    r2 = _chat(args.host, args.model, turn2)

    pe1, pe2 = r1.get("prompt_eval_count", 0), r2.get("prompt_eval_count", 0)
    d1 = r1.get("prompt_eval_duration", 0) / 1e6  # ns -> ms
    d2 = r2.get("prompt_eval_duration", 0) / 1e6

    print(f"model: {args.model}")
    print(f"system prefix chars: {len(system)}")
    print(f"turn 1: prompt_eval_count={pe1}  prompt_eval_duration={d1:.1f} ms")
    print(f"turn 2: prompt_eval_count={pe2}  prompt_eval_duration={d2:.1f} ms  (same persona prefix)")
    print("note: prompt_eval_count on Ollama 0.32.x reports total prompt size, cached or")
    print("      not; it cannot show reuse. The verdict below is duration-based.")

    if d1 <= 0 or pe1 <= 0:
        print("VERDICT: INCONCLUSIVE (no prompt_eval_duration returned)")
        raise SystemExit(2)

    t1_tokens_per_s = pe1 / (d1 / 1000.0)
    if t1_tokens_per_s > WARM_CACHE_TOKENS_PER_S:
        print(
            f"VERDICT: REUSE CONFIRMED (turn 1 prefill already ran at {t1_tokens_per_s:,.0f} tok/s;"
            " the prefix was cached before this run)"
        )
        raise SystemExit(0)

    ratio = d2 / d1
    if ratio < 0.5:
        print(f"VERDICT: REUSE CONFIRMED (turn 2 prefill took {ratio:.0%} of turn 1's duration)")
        raise SystemExit(0)
    print(f"VERDICT: NO CLEAR REUSE (turn 2 prefill took {ratio:.0%} of turn 1's; expected well under 50%)")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
