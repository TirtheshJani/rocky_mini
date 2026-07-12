"""TicLinter: deterministic character post-pass applied before TTS.

Runs on every spoken line (and on sim text replies, and on canned WAVs' captions). It
does three things, all deterministically so behavior never depends on the 7B being in
the mood:
  1. Enforces the ', question?' particle on questions.
  2. Strips assistant-isms ("As an AI", "Certainly!", ...) that break character.
  3. Measures tic metrics (question-particle rate, tripling, article-drop) and reports
     the somatic reflexes to fire: jazz_hands on a tripled word, the question stinger
     after a question particle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ARTICLES = {"a", "an", "the"}

# Assistant-isms to remove. Order matters (longer phrases first).
_ASSISTANT_ISMS = [
    re.compile(r"\bas an ai(?: language model)?\b[,.:;]?\s*", re.IGNORECASE),
    re.compile(r"\bi(?:'m| am)(?: just)? (?:an ai|a language model|here to help)\b[,.:;]?\s*", re.IGNORECASE),
    re.compile(r"\bhow (?:can|may) i (?:assist|help)(?: you)?\b\??\s*", re.IGNORECASE),
    re.compile(r"\bi(?:'d| would) be happy to\b\s*", re.IGNORECASE),
    re.compile(r"\bfeel free to\b\s*", re.IGNORECASE),
    re.compile(r"\bcertainly\b[!,.]?\s*", re.IGNORECASE),
    re.compile(r"\bof course\b[!,.]?\s*", re.IGNORECASE),
    re.compile(r"\bsure\b[!,]\s*", re.IGNORECASE),
]

_PARTICLE_RE = re.compile(r",\s*question\s*$", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z']+")


def _split_sentences(text: str) -> list[str]:
    """Split on a terminator followed by whitespace/end, so decimals stay intact."""
    segments: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] in ".!?":
            j = i
            while j < n and text[j] in ".!?":
                j += 1
            if j >= n or text[j].isspace():
                segments.append(text[start:j])
                while j < n and text[j].isspace():
                    j += 1
                start = j
                i = j
                continue
            i = j
            continue
        i += 1
    if start < n:
        segments.append(text[start:])
    return segments


@dataclass
class TicMetrics:
    word_count: int = 0
    question_sentences: int = 0
    question_particle_hits: int = 0
    tripling_count: int = 0
    article_count: int = 0

    @property
    def question_particle_rate(self) -> float:
        if self.question_sentences == 0:
            return 1.0
        return self.question_particle_hits / self.question_sentences

    @property
    def article_rate(self) -> float:
        if self.word_count == 0:
            return 0.0
        return self.article_count / self.word_count


@dataclass
class LintResult:
    text: str
    metrics: TicMetrics
    emotes: list[str] = field(default_factory=list)  # e.g. ["jazz_hands"]
    sfx: list[str] = field(default_factory=list)  # e.g. ["question"]


def _strip_assistant_isms(text: str) -> str:
    for pat in _ASSISTANT_ISMS:
        text = pat.sub("", text)
    # Collapse whitespace and fix a leading lowercase left by a strip.
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def _apply_question_particle(sentence: str) -> tuple[str, bool, bool]:
    """Return (fixed_sentence, is_question, has_particle)."""
    stripped = sentence.strip()
    if not stripped.endswith("?"):
        return sentence, False, False
    body = stripped[:-1].rstrip()
    if _PARTICLE_RE.search(body):
        return sentence, True, True
    body = body.rstrip(",").rstrip()
    lead = sentence[: len(sentence) - len(sentence.lstrip())]
    return f"{lead}{body}, question?", True, True


def _count_tripling(words: list[str]) -> int:
    count = 0
    i = 0
    n = len(words)
    while i < n:
        j = i
        while j < n and words[j].lower() == words[i].lower():
            j += 1
        if j - i >= 3:
            count += 1
        i = max(j, i + 1)
    return count


def lint(text: str) -> LintResult:
    """Apply the deterministic tic pass and measure metrics."""
    text = _strip_assistant_isms(text)
    metrics = TicMetrics()

    sentences = _split_sentences(text)
    fixed: list[str] = []
    for sent in sentences:
        new_sent, is_q, has_particle = _apply_question_particle(sent)
        fixed.append(new_sent)
        if is_q:
            metrics.question_sentences += 1
            if has_particle:
                metrics.question_particle_hits += 1
    out_text = " ".join(s.strip() for s in fixed if s.strip())

    words = _WORD_RE.findall(out_text)
    metrics.word_count = len(words)
    metrics.article_count = sum(1 for w in words if w.lower() in _ARTICLES)
    metrics.tripling_count = _count_tripling(words)

    emotes: list[str] = []
    sfx: list[str] = []
    if metrics.tripling_count > 0:
        emotes.append("jazz_hands")
    if metrics.question_particle_hits > 0:
        sfx.append("question")
    return LintResult(text=out_text, metrics=metrics, emotes=emotes, sfx=sfx)
