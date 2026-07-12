"""Sentence chunker for streamed LLM output.

Feeds token-by-token from the streaming chat completion and emits complete sentences
as soon as they are ready, so TTS can start on sentence one while the model is still
generating sentence two. Each chunk:
  - has inline [emote:NAME] / [sfx:NAME] tags extracted (they become motion/chord cues),
  - is run through the TicLinter (question particle, assistant-ism strip, reflexes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .tics import lint

_TAG_RE = re.compile(r"\[(emote|sfx):([a-z_]+)\]")
_TERMINATOR_RE = re.compile(r"[.!?]+")


@dataclass
class Chunk:
    text: str  # linted, tag-free text ready for TTS
    emotes: list[str] = field(default_factory=list)
    sfx: list[str] = field(default_factory=list)


def _extract_tags(raw: str) -> tuple[str, list[str], list[str]]:
    emotes: list[str] = []
    sfx: list[str] = []
    for kind, name in _TAG_RE.findall(raw):
        (emotes if kind == "emote" else sfx).append(name)
    text = _TAG_RE.sub("", raw)
    return text, emotes, sfx


def _make_chunk(raw: str) -> Chunk | None:
    text, tag_emotes, tag_sfx = _extract_tags(raw)
    result = lint(text)
    if not result.text:
        # Tags may still fire even if the linted text is empty.
        if tag_emotes or tag_sfx:
            return Chunk(text="", emotes=tag_emotes, sfx=tag_sfx)
        return None
    # Merge tag cues with the linter's somatic reflexes (dedup, order-stable).
    emotes = _dedup(tag_emotes + result.emotes)
    sfx = _dedup(tag_sfx + result.sfx)
    return Chunk(text=result.text, emotes=emotes, sfx=sfx)


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _find_boundary(buf: str) -> int:
    """Index just past the first sentence terminator followed by whitespace/end."""
    for m in _TERMINATOR_RE.finditer(buf):
        end = m.end()
        if end >= len(buf) or buf[end].isspace():
            return end
    return -1


class SentenceChunker:
    """Stateful streaming chunker. Call feed() with tokens, flush() at the end."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[Chunk]:
        self._buf += token
        chunks: list[Chunk] = []
        while True:
            end = _find_boundary(self._buf)
            if end < 0:
                break
            sentence = self._buf[:end]
            self._buf = self._buf[end:].lstrip()
            chunk = _make_chunk(sentence)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def flush(self) -> list[Chunk]:
        remainder = self._buf.strip()
        self._buf = ""
        if not remainder:
            return []
        chunk = _make_chunk(remainder)
        return [chunk] if chunk is not None else []


def chunk_text(text: str) -> list[Chunk]:
    """Convenience: chunk a complete (non-streamed) string."""
    chunker = SentenceChunker()
    chunks = chunker.feed(text)
    chunks.extend(chunker.flush())
    return chunks
