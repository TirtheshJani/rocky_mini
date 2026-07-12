"""ReflectionWorker: one cheap post-session pass, plus the greeting ritual.

After a session ends, one local-model call distills a two-sentence summary and mines a
couple of follow-up questions for next time. At the next session start, the greeting
ritual replays that memory in Rocky's voice ("TJ! You return. Yesterday you teach
sandwich. Is taco also sandwich, question?"). A deterministic fallback runs when no LLM
is configured, so sim and tests need no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLM

_REFLECT_PROMPT = (
    "You are Rocky reflecting on a lesson. In two short sentences, summarize what TJ "
    "taught. Then, on a new line starting with 'ASK:', write one short curious "
    "follow-up question in Rocky's telegraphic voice ending with ', question?'."
)


@dataclass
class Reflection:
    summary: str
    follow_ups: list[str] = field(default_factory=list)


class ReflectionWorker:
    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    async def reflect(self, new_facts: list[str]) -> Reflection:
        if self._llm is not None:
            return await self._reflect_llm(new_facts)
        return self._reflect_local(new_facts)

    def _reflect_local(self, new_facts: list[str]) -> Reflection:
        if not new_facts:
            return Reflection(summary="Quiet session. Rocky listen.", follow_ups=[])
        head = new_facts[:2]
        summary = "Rocky learn: " + "; ".join(head) + "."
        follow_ups = [f"Tell more about {fact}, question?" for fact in head[:1]]
        return Reflection(summary=summary, follow_ups=follow_ups)

    async def _reflect_llm(self, new_facts: list[str]) -> Reflection:  # pragma: no cover
        facts_block = "\n".join(f"- {f}" for f in new_facts) or "- (nothing new)"
        messages = [
            {"role": "system", "content": _REFLECT_PROMPT},
            {"role": "user", "content": f"Facts learned this session:\n{facts_block}"},
        ]
        text = ""
        async for ev in self._llm.stream(messages):
            text += ev.delta
        summary_lines: list[str] = []
        follow_ups: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("ASK:"):
                follow_ups.append(line[4:].strip())
            elif line:
                summary_lines.append(line)
        return Reflection(summary=" ".join(summary_lines).strip(), follow_ups=follow_ups)

    def greeting(self, last_summary: str | None, follow_ups: list[str] | None = None) -> str:
        """Build the next-session greeting from the previous reflection."""
        if not last_summary:
            return "You are new voice. Rocky is Rocky. You teach, question?"
        parts = ["TJ! You return.", last_summary]
        if follow_ups:
            parts.append(follow_ups[0])
        return " ".join(p.strip() for p in parts if p.strip())
