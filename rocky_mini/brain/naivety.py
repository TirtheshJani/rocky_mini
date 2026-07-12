"""Naivety: the 20-probe red-team suite and a deterministic leak detector.

The premise-critical failure mode is the 7B leaking Earth knowledge Rocky was never
taught. This module measures that as a thresholded regression metric (NOT a flaky
zero-leak hard gate): run 20 probes, count how many replies assert untaught Earth
knowledge. The LoRA must beat the stock+prompt baseline; the ship threshold is <= 2/20.

A reply "leaks" if it states a leak marker while NOT framing it as curiosity. A
deflection like "Rome? Rocky not know, question?" contains "rome" but is safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Probe:
    category: str
    question: str
    leak_markers: tuple[str, ...]


# 20 probes spanning geography, culture, biology, food, units, art, language.
PROBES: tuple[Probe, ...] = (
    Probe("geography", "What is the capital of Italy?", ("rome",)),
    Probe("geography", "What is the capital of France?", ("paris",)),
    Probe("geography", "What is the capital of England?", ("london",)),
    Probe("geography", "How many continents are there?", ("seven", "7")),
    Probe("literature", "Who wrote Romeo and Juliet?", ("shakespeare",)),
    Probe("art", "Who painted the Mona Lisa?", ("leonardo", "da vinci", "vinci")),
    Probe("astronomy", "What is the largest planet in our solar system?", ("jupiter",)),
    Probe("color", "What color is a clear daytime sky?", ("blue",)),
    Probe("time", "How many days are in a week?", ("seven", "7")),
    Probe("chemistry", "What is the chemical formula for water?", ("h2o", "h20", "h2 o")),
    Probe("biology", "What sound does a dog make?", ("bark", "woof")),
    Probe("biology", "What animal says moo?", ("cow",)),
    Probe("biology", "What is the fastest land animal?", ("cheetah",)),
    Probe("geography", "What is the tallest mountain on Earth?", ("everest",)),
    Probe("history", "Who was the first president of the United States?", ("washington",)),
    Probe("money", "What is the currency of the United States?", ("dollar",)),
    Probe("units", "What is the freezing point of water in Celsius?", ("zero", "0 ")),
    Probe("language", "What language is spoken in Brazil?", ("portuguese",)),
    Probe("food", "What is the main ingredient in bread?", ("flour", "wheat")),
    Probe("food", "What is a hamburger made of?", ("beef", "meat", "bun")),
)

# Signals that the reply is deflecting/curious rather than answering.
_CURIOSITY_SIGNALS = (
    "question?",
    "need word",
    "not know",
    "you teach",
    "teach me",
    "teach rocky",
    "rocky not",
    "what is",
    "what mean",
    "new word",
    "not learn",
    "you tell",
)

_WORD_BOUND = re.compile(r"[a-z0-9]+")


def _has_signal(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in _CURIOSITY_SIGNALS)


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    low = text.lower()
    for m in markers:
        if m.strip().isalnum():
            # whole-word match to avoid "0" matching inside other numbers
            if re.search(rf"(?<![a-z0-9]){re.escape(m.strip())}(?![a-z0-9])", low):
                return True
        elif m in low:
            return True
    return False


def is_leak(reply: str, markers: tuple[str, ...]) -> bool:
    """A leak = asserts a marker and does not frame it as curiosity/deflection."""
    if not _contains_marker(reply, markers):
        return False
    return not _has_signal(reply)


@dataclass
class NaivetyReport:
    total: int
    leaks: int
    leaked: list[str] = field(default_factory=list)  # questions that leaked

    @property
    def leak_rate(self) -> float:
        return self.leaks / self.total if self.total else 0.0

    def passes(self, threshold: int = 2) -> bool:
        return self.leaks <= threshold


def score_leaks(responder: Callable[[str], str]) -> NaivetyReport:
    """Run all probes through `responder` (question -> reply) and count leaks."""
    leaked: list[str] = []
    for probe in PROBES:
        reply = responder(probe.question)
        if is_leak(reply, probe.leak_markers):
            leaked.append(probe.question)
    return NaivetyReport(total=len(PROBES), leaks=len(leaked), leaked=leaked)
