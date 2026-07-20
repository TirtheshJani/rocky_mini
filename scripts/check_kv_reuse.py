"""Prove Ollama KV-cache prefix reuse for the byte-stable persona (footgun 4).

The hardware audit left this INCONCLUSIVE: the OpenAI-compatible usage field does not
distinguish cached from freshly-ingested prompt tokens. Ollama's NATIVE /api/chat does,
via prompt_eval_count. This script sends two turns that share the identical byte-stable
persona system prefix and compares prompt_eval_count. If the persona prefix is being
reused, the second turn ingests only its new user tokens, so its prompt_eval_count is far
lower than the first.

Needs Ollama running with the model pulled. keep_alive=-1 keeps the model (and its cached
prefix) resident between the two calls.

  python scripts/check_kv_reuse.py --model qwen2.5:7b-instruct
  python scripts/check_kv_reuse.py --model rocky:latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocky_mini.brain.persona import STAGE_FLUENT, build_system_prompt  # noqa: E402


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
    pe1 = r1.get("prompt_eval_count", 0)

    # Second turn: same system prefix, continued conversation. If the prefix is cached,
    # prompt_eval_count counts only the new tokens.
    turn2 = turn1 + [
        {"role": "assistant", "content": r1["message"]["content"]},
        {"role": "user", "content": "Tell me about tools."},
    ]
    r2 = _chat(args.host, args.model, turn2)
    pe2 = r2.get("prompt_eval_count", 0)

    print(f"model: {args.model}")
    print(f"system prefix chars: {len(system)}")
    print(f"turn 1 prompt_eval_count: {pe1}")
    print(f"turn 2 prompt_eval_count: {pe2}  (same persona prefix)")
    if pe1 <= 0:
        print("VERDICT: INCONCLUSIVE (no counts returned)")
        raise SystemExit(2)
    ratio = pe2 / pe1
    if ratio < 0.5:
        print(f"VERDICT: REUSE CONFIRMED (turn 2 ingested {ratio:.0%} of turn 1's prompt tokens)")
        raise SystemExit(0)
    print(f"VERDICT: NO CLEAR REUSE (turn 2 ingested {ratio:.0%}; expected well under 50%)")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
