"""Memory record models.

Confidence lifecycle for a Fact: heard_once -> confirmed -> mastered. A fact starts
heard_once; hearing it again or an explicit confirm_fact call promotes it. Deletes are
tombstones (deleted=True) so the JSONL log stays append-only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

HEARD_ONCE = "heard_once"
CONFIRMED = "confirmed"
MASTERED = "mastered"

_RANK = {HEARD_ONCE: 0, CONFIRMED: 1, MASTERED: 2}


def confidence_rank(confidence: str) -> int:
    return _RANK.get(confidence, 0)


def next_confidence(confidence: str) -> str:
    """Promote one step; mastered stays mastered."""
    if confidence == HEARD_ONCE:
        return CONFIRMED
    if confidence == CONFIRMED:
        return MASTERED
    return MASTERED


class Fact(BaseModel):
    id: str
    category: str
    text: str
    source_quote: str = ""
    confidence: str = HEARD_ONCE
    heard_count: int = 1
    created_at: str
    updated_at: str
    deleted: bool = False


class OpenQuestion(BaseModel):
    id: str
    text: str
    created_at: str
    answered: bool = False
    deleted: bool = False


class SessionRecord(BaseModel):
    id: str
    summary: str = ""
    follow_ups: list[str] = Field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""
