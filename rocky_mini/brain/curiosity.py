"""Curiosity scheduler: deterministic Python, not prompt hope.

The model phrases a proactive question, but it never picks the topic. This scheduler
owns topic selection: a scored queue seeded with canon targets, score decay after each
ask, a hard cap of one proactive question per N turns, and an idle trigger. Keeping it
deterministic means curiosity cadence is testable and never spirals.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canon curiosity targets from the Rocky dossier.
CANON_TARGETS: tuple[str, ...] = (
    "sleep",
    "faces and emotions",
    "humor",
    "hugs",
    "sight",
    "lying",
    "human units of measure",
    "family",
    "music",
    "food",
)


@dataclass
class CuriosityItem:
    topic: str
    score: float = 1.0


class CuriosityScheduler:
    def __init__(self, cap_every_n_turns: int = 3, decay: float = 0.6) -> None:
        self.cap_every_n_turns = cap_every_n_turns
        self.decay = decay
        self.items: list[CuriosityItem] = [CuriosityItem(t) for t in CANON_TARGETS]
        # Start "ready" so the first eligible turn can ask.
        self._turns_since_proactive = cap_every_n_turns

    def note_turn(self) -> None:
        """Call once per completed conversation turn."""
        self._turns_since_proactive += 1

    def should_ask(self) -> bool:
        return self._turns_since_proactive >= self.cap_every_n_turns and self._has_live_item()

    def _has_live_item(self) -> bool:
        return any(it.score > 0.05 for it in self.items)

    def pick(self, force: bool = False) -> str | None:
        """Return the top topic if eligible (or forced by idle), decaying it."""
        if not force and not self.should_ask():
            return None
        live = [it for it in self.items if it.score > 0.05]
        if not live:
            return None
        top = max(live, key=lambda it: it.score)
        top.score *= self.decay
        self._turns_since_proactive = 0
        return top.topic

    def satisfy(self, topic: str) -> None:
        """TJ answered/covered this topic: drop its score so Rocky stops asking."""
        for it in self.items:
            if it.topic == topic:
                it.score = 0.0

    def boost(self, topic: str, amount: float = 1.0) -> None:
        for it in self.items:
            if it.topic == topic:
                it.score += amount
                return
        self.items.append(CuriosityItem(topic, amount))
