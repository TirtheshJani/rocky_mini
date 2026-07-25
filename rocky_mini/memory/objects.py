"""Visual object memory: a JSONL sidecar of taught objects under ~/.rocky_mini.

Companion to the text MemoryStore (decision 21). TJ shows Rocky an object and names it;
enroll() stores one or more DINOv2 embeddings (shots) of what the camera saw, keyed by a
normalized name. Later, match() re-identifies an object by max cosine over its shots.

Same event-sourced discipline as store.py: every mutation appends a full record (fsynced
via _JsonlLog), load replays the log keeping the latest record per id, and deletes are
tombstones. The log lives in the home dir so it survives app reinstalls (footgun 5) and a
SIGINT mid-write never corrupts prior objects. It rides the same export_zip/import_zip as
the text memory (objects.jsonl is in store._LOG_FILES).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from ..vision.embed import cosine
from .store import _JsonlLog, _TRAILING_INT, _default_now, _normalize

# Few-shot cap: at most this many enrollment shots per object (oldest dropped). Three
# angles is plenty for whole-frame re-identification and bounds the match cost.
MAX_SHOTS = 3


class ObjectRecord(BaseModel):
    """One taught object: a name plus up to MAX_SHOTS embedding shots."""

    id: str
    name: str
    embeddings: list[list[float]] = Field(default_factory=list)
    created_at: str
    updated_at: str
    deleted: bool = False


class ObjectStore:
    def __init__(self, base_dir: str | Path, now: Callable[[], str] | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._now = now or _default_now
        self._log = _JsonlLog(self.base_dir / "objects.jsonl")
        self._objects: dict[str, ObjectRecord] = {}
        self._seq = 0
        self.load()

    # -- loading -----------------------------------------------------------
    def load(self) -> None:
        self._objects.clear()
        for rec in self._log.read_all():
            self._objects[rec["id"]] = ObjectRecord(**rec)
        self._seq = self._max_seq()

    def _max_seq(self) -> int:
        best = 0
        for key in self._objects:
            m = _TRAILING_INT.search(key)
            if m:
                best = max(best, int(m.group(1)))
        return best

    def _next_id(self) -> str:
        self._seq += 1
        return f"o{self._seq}"

    # -- enrollment --------------------------------------------------------
    def enroll(self, name: str, embedding: list[float]) -> str:
        """Add an embedding shot for `name`, creating the object if new. Returns its id.

        Re-teaching an existing (non-deleted) name appends a shot, capped at MAX_SHOTS
        with the oldest dropped. Name matching is normalized (case/whitespace-insensitive),
        matching how the text store dedups facts.
        """
        norm = _normalize(name)
        for obj in self._objects.values():
            if obj.deleted:
                continue
            if _normalize(obj.name) == norm:
                obj.embeddings.append(list(embedding))
                if len(obj.embeddings) > MAX_SHOTS:
                    obj.embeddings = obj.embeddings[-MAX_SHOTS:]
                obj.updated_at = self._now()
                self._log.append(obj.model_dump())
                return obj.id
        now = self._now()
        oid = self._next_id()
        obj = ObjectRecord(
            id=oid,
            name=name,
            embeddings=[list(embedding)],
            created_at=now,
            updated_at=now,
        )
        self._objects[oid] = obj
        self._log.append(obj.model_dump())
        return oid

    def delete_object(self, object_id: str) -> None:
        obj = self._objects.get(object_id)
        if obj is None:
            raise KeyError(f"no such object: {object_id}")
        obj.deleted = True
        obj.updated_at = self._now()
        self._log.append(obj.model_dump())

    # -- queries -----------------------------------------------------------
    def active_objects(self) -> list[ObjectRecord]:
        return [o for o in self._objects.values() if not o.deleted]

    def names(self) -> list[str]:
        return [o.name for o in self.active_objects()]

    def match(
        self, embedding: list[float] | None, threshold: float, margin: float
    ) -> tuple[str, float] | None:
        """Best matching object name and its score, or None.

        Score is the max cosine over an object's shots. Accepts the top object only if
        its score >= threshold AND (when two or more objects are enrolled) it beats the
        runner-up by at least `margin`. The margin guards against a confident-but-wrong
        pick between two similar-looking objects; it is skipped with a single object,
        where there is nothing to confuse it with.
        """
        if not embedding:
            return None
        scored: list[tuple[str, float]] = []
        for obj in self.active_objects():
            if not obj.embeddings:
                continue
            best = max(cosine(embedding, shot) for shot in obj.embeddings)
            scored.append((obj.name, best))
        if not scored:
            return None
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top_name, top_score = scored[0]
        if top_score < threshold:
            return None
        if len(scored) >= 2 and (top_score - scored[1][1]) < margin:
            return None
        return (top_name, top_score)
