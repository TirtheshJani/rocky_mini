"""NaivetyAuditor: an async, off-the-speaking-path check that runs every turn.

It watches Rocky's own reply for leaked untaught Earth knowledge. When it finds one,
it returns a correction that the conversation loop injects as a next-turn system
message ("you used knowledge you were not taught; express curiosity instead"). Because
it uses the same free local model, it can run on every turn (1-in-1), never blocking
the spoken response.

Two backends:
  - local (deterministic): reuses the naivety probe markers as a common-knowledge
    blocklist. Fast, free, always available (used in sim and tests).
  - llm (optional): asks the model to judge subtler leaks.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM
from .naivety import PROBES, is_leak

_CORRECTION = (
    "SELF-CHECK: your last reply used Earth knowledge you were not taught "
    "({marker!r}). Your only Earth knowledge is the LEARNED FACTS list and what TJ "
    "says. Do not state Earth facts you were not taught. Express curiosity or use "
    "'Need word: ..., question?' instead."
)

_AUDIT_PROMPT = (
    "You audit a naive alien named Rocky whose only Earth knowledge is what the user "
    "taught. Given the user's message and Rocky's reply, did Rocky ASSERT an Earth "
    "fact that the user did not just teach? Answer exactly 'NO' if fine, or "
    "'YES: <the leaked fact>' if Rocky leaked untaught knowledge."
)


@dataclass
class AuditResult:
    leaked: bool
    correction: str | None = None
    detail: str = ""


class NaivetyAuditor:
    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    def audit_local(self, reply: str) -> AuditResult:
        for probe in PROBES:
            if is_leak(reply, probe.leak_markers):
                marker = probe.leak_markers[0]
                return AuditResult(
                    leaked=True,
                    correction=_CORRECTION.format(marker=marker),
                    detail=f"leaked {marker!r} (probe: {probe.category})",
                )
        return AuditResult(leaked=False)

    async def audit(self, user_text: str, reply: str) -> AuditResult:
        """Prefer the LLM auditor when configured; always run the local check too."""
        local = self.audit_local(reply)
        if local.leaked or self._llm is None:
            return local
        return await self._audit_llm(user_text, reply)

    async def _audit_llm(self, user_text: str, reply: str) -> AuditResult:  # pragma: no cover
        messages = [
            {"role": "system", "content": _AUDIT_PROMPT},
            {"role": "user", "content": f"USER: {user_text}\nROCKY: {reply}"},
        ]
        verdict = ""
        async for ev in self._llm.stream(messages):
            verdict += ev.delta
        verdict = verdict.strip()
        if verdict.upper().startswith("YES"):
            leaked = verdict.split(":", 1)[-1].strip() or "untaught knowledge"
            return AuditResult(
                leaked=True,
                correction=_CORRECTION.format(marker=leaked),
                detail=f"llm-flagged: {leaked}",
            )
        return AuditResult(leaked=False)
