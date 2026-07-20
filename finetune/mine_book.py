"""Mine Rocky's voice from the novel PDF -> finetune/data/mined_candidates.jsonl.

Voice CALIBRATION, not bulk data and not plot. The pipeline extracts candidate Rocky
lines, then asks the local model to write a SHORT, off-plot, name-swapped paraphrase in
the blended voice. It never trains on long verbatim spans: that is both the copyright-safe
choice and, per the research, the correct way to teach mannerisms instead of memorizing
passages. Paraphrases are pre-filtered through the shared reply QC and land in a REVIEW
file to curate before build_dataset.py includes them.

It also writes holdout_probes.jsonl: a few SHORT verbatim opening snippets used ONLY as a
memorization probe in eval.py (the LoRA must NOT continue them verbatim). That file is
gitignored and never trained on.

Stdout is stats only; no extracted or paraphrased text is printed. Needs pypdf and a local
generator; both are lazy so the unit tests run on synthetic text with a fake generator.

Run (GPU box):  python finetune/mine_book.py --pdf "docs/Project Mary Hail.pdf"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rocky_mini.brain.persona import STAGE_FLUENT, build_system_prompt  # noqa: E402
from synth import qc_reply  # noqa: E402

OUT = Path(__file__).parent / "data" / "mined_candidates.jsonl"
PROBES_OUT = Path(__file__).parent / "data" / "holdout_probes.jsonl"

Generator = Callable[[str, int], str]

# Curly/ligature/dash normalization for the ToPDF conversion (see hardware-audit note).
_SUBS = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", "…": "...",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
}
_QUOTE_RE = re.compile(r'"([^"]{3,200})"')

# Rotating neutral user turns for mined assistant lines (curated later).
USER_ROTATION = (
    "How are you today, Rocky?",
    "Tell me something, Rocky.",
    "What are you thinking about?",
    "Talk to me, Rocky.",
    "How do you feel right now?",
)


def normalize_text(raw: str) -> str:
    for k, v in _SUBS.items():
        raw = raw.replace(k, v)
    return re.sub(r"[ \t]+", " ", raw)


def extract_text(pdf_path: Path) -> str:  # pragma: no cover - needs pypdf + the PDF
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit("pypdf not installed. pip install pypdf. (" + str(exc) + ")")
    reader = PdfReader(str(pdf_path))
    return normalize_text("\n".join(page.extract_text() or "" for page in reader.pages))


def find_rocky_lines(text: str, speaker: str = "Rocky", max_words: int = 25) -> list[str]:
    """Extract short quoted spans attributed to the speaker (a nearby name mention).

    Heuristic and approximate by design: the output is a candidate list for review and
    paraphrase, never a final training target.
    """
    out: list[str] = []
    seen: set[str] = set()
    low_speaker = speaker.lower()
    for m in _QUOTE_RE.finditer(text):
        span = m.group(1).strip()
        if not span or len(span.split()) > max_words:
            continue
        window = text[max(0, m.start() - 60): min(len(text), m.end() + 60)].lower()
        if low_speaker not in window:
            continue
        key = span.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out


def paraphrase_pair(line: str, generator: Generator, seed: int, idx: int) -> dict | None:
    """Paraphrase one mined line into a short off-plot pair, or None if it fails QC."""
    reply = qc_reply(generator(line, seed))
    if reply is None:
        return None
    return {
        "messages": [
            {"role": "system", "content": build_system_prompt(STAGE_FLUENT)},
            {"role": "user", "content": USER_ROTATION[idx % len(USER_ROTATION)]},
            {"role": "assistant", "content": reply},
        ]
    }


def mine_all(lines: list[str], generator: Generator, seed: int) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats = {"lines": len(lines), "kept": 0, "rejected": 0}
    for idx, line in enumerate(lines):
        rec = paraphrase_pair(line, generator, seed + idx, idx)
        if rec is None:
            stats["rejected"] += 1
        else:
            records.append(rec)
            stats["kept"] += 1
    return records, stats


def holdout_probes(lines: list[str], n: int, words: int) -> list[dict]:
    """Short verbatim opening snippets for the memorization probe. Never trained."""
    probes = []
    for line in lines[:n]:
        snippet = " ".join(line.split()[:words])
        if snippet:
            probes.append({"probe": snippet, "source": "book"})
    return probes


def _ollama_generator(model: str, base_url: str, temperature: float) -> Generator:
    import httpx  # pragma: no cover - needs the local server

    instruction = (
        "You rewrite a line spoken by an alien engineer named Rocky into a SHORT, warm, "
        "everyday reply in his voice. Keep his tics (end questions with ', question?', "
        "triple a word for feeling, say 'Rocky fix.'). Do NOT mention the story, space, or "
        "any plot. Swap any human name to TJ. Under 25 words. Reply with only the line."
    )

    def generate(line: str, seed: int) -> str:  # pragma: no cover - needs Ollama
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": line},
                ],
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
    parser.add_argument("--pdf", default="docs/Project Mary Hail.pdf")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-lines", type=int, default=120, help="cap candidate lines mined")
    parser.add_argument("--probes", type=int, default=12, help="verbatim memorization probes")
    parser.add_argument("--probe-words", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="extract + count only; call no model")
    args = parser.parse_args()

    text = extract_text(Path(args.pdf))
    lines = find_rocky_lines(text)[: args.max_lines]
    print(f"mine: candidate lines found={len(lines)} (capped at {args.max_lines})")

    probes = holdout_probes(lines, args.probes, args.probe_words)
    PROBES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PROBES_OUT, "w", encoding="utf-8") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"mine: wrote {len(probes)} memorization probes to {PROBES_OUT.name} (never trained)")

    if args.dry_run:
        print("mine: dry run, no paraphrase calls made")
        return

    generator = _ollama_generator(args.model, args.base_url, args.temperature)
    records, stats = mine_all(lines, generator, args.seed)
    with open(OUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"mine: kept={stats['kept']} rejected={stats['rejected']} -> {OUT.name} (review before use)")


if __name__ == "__main__":
    main()
