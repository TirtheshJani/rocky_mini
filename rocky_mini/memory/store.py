"""JSONL memory store under ~/.rocky_mini.

Append-only, event-sourced: every mutation appends a full record (fsynced), and load
replays the logs, keeping the latest record per id. Deletes are tombstones. This gives
durability across app reinstalls (the home dir survives pip --force-reinstall) and a
SIGINT mid-write never corrupts prior facts.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .models import (
    CONFIRMED,
    HEARD_ONCE,
    Fact,
    OpenQuestion,
    SessionRecord,
    confidence_rank,
    next_confidence,
)

_WS = re.compile(r"\s+")
_TRAILING_INT = re.compile(r"(\d+)$")
_LOG_FILES = ("facts.jsonl", "open_questions.jsonl", "sessions.jsonl")


def _default_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


class _JsonlLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


class MemoryStore:
    def __init__(self, base_dir: str | Path, now: Callable[[], str] | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._now = now or _default_now
        self._facts_log = _JsonlLog(self.base_dir / "facts.jsonl")
        self._q_log = _JsonlLog(self.base_dir / "open_questions.jsonl")
        self._s_log = _JsonlLog(self.base_dir / "sessions.jsonl")
        self._facts: dict[str, Fact] = {}
        self._questions: dict[str, OpenQuestion] = {}
        self._sessions: dict[str, SessionRecord] = {}
        self._seq = 0
        self.load()

    # -- loading -----------------------------------------------------------
    def load(self) -> None:
        self._facts.clear()
        self._questions.clear()
        self._sessions.clear()
        for rec in self._facts_log.read_all():
            self._facts[rec["id"]] = Fact(**rec)
        for rec in self._q_log.read_all():
            self._questions[rec["id"]] = OpenQuestion(**rec)
        for rec in self._s_log.read_all():
            self._sessions[rec["id"]] = SessionRecord(**rec)
        self._seq = self._max_seq()

    def _max_seq(self) -> int:
        best = 0
        for coll in (self._facts, self._questions, self._sessions):
            for key in coll:
                m = _TRAILING_INT.search(key)
                if m:
                    best = max(best, int(m.group(1)))
        return best

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{self._seq}"

    # -- facts -------------------------------------------------------------
    def remember_fact(self, category: str, text: str, source_quote: str = "") -> str:
        norm = _normalize(text)
        for fact in self._facts.values():
            if fact.deleted:
                continue
            if fact.category.lower() == category.lower() and _normalize(fact.text) == norm:
                # Heard again: restate-then-confirm bumps confidence heard_once->confirmed.
                fact.heard_count += 1
                if fact.confidence == HEARD_ONCE:
                    fact.confidence = CONFIRMED
                fact.updated_at = self._now()
                self._facts_log.append(fact.model_dump())
                return fact.id
        now = self._now()
        fid = self._next_id("f")
        fact = Fact(
            id=fid,
            category=category,
            text=text,
            source_quote=source_quote,
            confidence=HEARD_ONCE,
            heard_count=1,
            created_at=now,
            updated_at=now,
        )
        self._facts[fid] = fact
        self._facts_log.append(fact.model_dump())
        return fid

    def confirm_fact(self, fact_id: str) -> str:
        fact = self._facts.get(fact_id)
        if fact is None or fact.deleted:
            raise KeyError(f"no such fact: {fact_id}")
        fact.confidence = next_confidence(fact.confidence)
        fact.updated_at = self._now()
        self._facts_log.append(fact.model_dump())
        return fact.confidence

    def delete_fact(self, fact_id: str) -> None:
        fact = self._facts.get(fact_id)
        if fact is None:
            raise KeyError(f"no such fact: {fact_id}")
        fact.deleted = True
        fact.updated_at = self._now()
        self._facts_log.append(fact.model_dump())

    def active_facts(self) -> list[Fact]:
        return [f for f in self._facts.values() if not f.deleted]

    def get_fact(self, fact_id: str) -> Fact | None:
        fact = self._facts.get(fact_id)
        return fact if fact and not fact.deleted else None

    # -- open questions ----------------------------------------------------
    def note_open_question(self, text: str) -> str:
        norm = _normalize(text)
        for q in self._questions.values():
            if not q.deleted and _normalize(q.text) == norm:
                return q.id
        qid = self._next_id("q")
        q = OpenQuestion(id=qid, text=text, created_at=self._now())
        self._questions[qid] = q
        self._q_log.append(q.model_dump())
        return qid

    def answer_question(self, qid: str) -> None:
        q = self._questions.get(qid)
        if q is None:
            raise KeyError(f"no such question: {qid}")
        q.answered = True
        self._q_log.append(q.model_dump())

    def active_questions(self) -> list[OpenQuestion]:
        return [q for q in self._questions.values() if not q.deleted and not q.answered]

    # -- sessions ----------------------------------------------------------
    def add_session(self, summary: str, follow_ups: list[str] | None = None) -> str:
        sid = self._next_id("s")
        now = self._now()
        rec = SessionRecord(
            id=sid, summary=summary, follow_ups=follow_ups or [], started_at=now, ended_at=now
        )
        self._sessions[sid] = rec
        self._s_log.append(rec.model_dump())
        return sid

    def last_session(self) -> SessionRecord | None:
        if not self._sessions:
            return None
        return sorted(self._sessions.values(), key=lambda s: s.id)[-1]

    # -- digest / persona feed --------------------------------------------
    def digest_facts(self, limit: int | None = None) -> list[str]:
        facts = sorted(
            self.active_facts(),
            key=lambda f: (confidence_rank(f.confidence), f.updated_at),
            reverse=True,
        )
        texts = [f.text for f in facts]
        return texts[:limit] if limit else texts

    def fuzzy_facts(self, limit: int = 2) -> list[str]:
        """Up to `limit` stale heard_once facts, stalest first, for the (memory fuzzy) marker."""
        stale = [f for f in self.active_facts() if f.confidence == HEARD_ONCE]
        stale.sort(key=lambda f: f.updated_at)
        return [f.text for f in stale[:limit]]

    def known_words(self) -> int:
        """Vocabulary proxy that drives the growth arc: count of active facts."""
        return len(self.active_facts())

    # -- export / import ---------------------------------------------------
    def export_zip(self, path: str | Path) -> Path:
        path = Path(path)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in _LOG_FILES:
                fpath = self.base_dir / name
                if fpath.exists():
                    zf.write(fpath, arcname=name)
        return path

    def import_zip(self, path: str | Path) -> None:
        """Replace this store's logs with an exported snapshot, then reload."""
        path = Path(path)
        with zipfile.ZipFile(path, "r") as zf:
            for name in _LOG_FILES:
                if name in zf.namelist():
                    data = zf.read(name)
                    (self.base_dir / name).write_bytes(data)
        self.load()
