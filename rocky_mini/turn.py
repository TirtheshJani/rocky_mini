"""ConversationLoop: the asyncio state machine that turns input into Rocky's response.

Pipeline per turn:
  ack chord + thinking pose (<=150 ms perceived response)
    -> stream LLM (persona prefix + digest + history + corrections)
    -> sentence chunker (tags + tics)
    -> per chunk: fire emote/chord cues, synth TTS, push to the Mixer (generation-tagged)
    -> handle tool calls (pydantic-validated, one retry on malformed)
    -> run the NaivetyAuditor off the speaking path; stash any correction for next turn
    -> advance curiosity cadence; return metrics.

Barge-in bumps the generation counter and cancels the in-flight task; the Mixer drops
any voice tagged with the now-stale generation, so interrupted speech stops at once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from .audio.chords import STINGER_BANK, ChordBank
from .audio.output import Mixer
from .brain.auditor import AuditResult, NaivetyAuditor
from .brain.chunker import Chunk, SentenceChunker
from .brain.curiosity import CuriosityScheduler
from .brain.llm import LLM
from .brain.persona import PersonaProfile, build_messages, stage_for_words
from .brain.tools import ToolCallError, ToolExecutor
from .memory.store import MemoryStore
from .speech.stt import STT
from .speech.tts import TTS

IDLE = "IDLE"
THINKING = "THINKING"
SPEAKING = "SPEAKING"
SLEEPWATCH = "SLEEPWATCH"

_HISTORY_MAX = 24
_HISTORY_TRIM = 8


@dataclass
class TurnMetrics:
    generation: int = 0
    t_start: float = 0.0
    t_ack: float | None = None
    t_first_token: float | None = None
    t_first_audio: float | None = None
    t_done: float | None = None
    prompt_eval_count: int = 0
    eval_count: int = 0
    tool_calls: int = 0
    tool_errors: int = 0

    def _delta(self, t: float | None) -> float | None:
        return None if t is None else t - self.t_start

    @property
    def ack_ms(self) -> float | None:
        d = self._delta(self.t_ack)
        return None if d is None else d * 1000.0

    @property
    def time_to_first_audio_s(self) -> float | None:
        return self._delta(self.t_first_audio)

    @property
    def total_s(self) -> float | None:
        return self._delta(self.t_done)


@dataclass
class TurnResult:
    reply: str
    chunks: list[Chunk] = field(default_factory=list)
    tool_results: list[tuple[str, str]] = field(default_factory=list)
    metrics: TurnMetrics = field(default_factory=TurnMetrics)
    audit: AuditResult | None = None
    proactive_topic: str | None = None


def build_profile(memory: MemoryStore) -> PersonaProfile:
    known = memory.known_words()
    return PersonaProfile(
        stage=stage_for_words(known),
        known_words=known,
        learned_facts=memory.digest_facts(limit=40),
        open_questions=[q.text for q in memory.active_questions()][:10],
        fuzzy_facts=memory.fuzzy_facts(2),
    )


class ConversationLoop:
    def __init__(
        self,
        *,
        llm: LLM,
        tts: TTS,
        mixer: Mixer,
        memory: MemoryStore,
        tools: ToolExecutor,
        auditor: NaivetyAuditor | None = None,
        curiosity: CuriosityScheduler | None = None,
        chords: ChordBank | None = None,
        stt: STT | None = None,
        emote_sink: Callable[[str], None] | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.mixer = mixer
        self.memory = memory
        self.tools = tools
        self.auditor = auditor or NaivetyAuditor()
        self.curiosity = curiosity or CuriosityScheduler()
        self.chords = chords or ChordBank(sample_rate=mixer.sample_rate)
        self.stt = stt
        self.emote_sink = emote_sink
        self.clock = clock

        self.generation = 0
        self.state = IDLE
        self.sleep_watch = False
        self.history: list[dict] = []
        self._pending_system: list[str] = []
        self._current_task: asyncio.Task | None = None

    # -- public entry points ----------------------------------------------
    async def handle_voice_turn(self, audio, sample_rate: int, t_start: float | None = None) -> TurnResult:
        if self.stt is None:
            raise RuntimeError("no STT configured for voice turns")
        user_text = await self.stt.transcribe(audio, sample_rate)
        return await self.handle_text_turn(user_text, t_start=t_start)

    async def handle_text_turn(self, user_text: str, t_start: float | None = None) -> TurnResult:
        metrics = TurnMetrics(generation=self.generation)
        metrics.t_start = self.clock() if t_start is None else t_start

        # Perceived response: ack chord + thinking pose within ~150 ms.
        self.state = THINKING
        self._fire_chord("thinking")
        self._fire_emote("thinking")
        metrics.t_ack = self.clock()

        messages = self._build_messages(user_text)
        chunker = SentenceChunker()
        reply_parts: list[str] = []
        collected: list[Chunk] = []
        tool_calls: list = []

        self.state = SPEAKING
        async for ev in self.llm.stream(messages):
            if ev.delta:
                if metrics.t_first_token is None:
                    metrics.t_first_token = self.clock()
                reply_parts.append(ev.delta)
                for chunk in chunker.feed(ev.delta):
                    await self._speak_chunk(chunk, metrics)
                    collected.append(chunk)
            if ev.done:
                metrics.prompt_eval_count = ev.prompt_eval_count
                metrics.eval_count = ev.eval_count
                tool_calls = ev.tool_calls
        for chunk in chunker.flush():
            await self._speak_chunk(chunk, metrics)
            collected.append(chunk)

        reply = "".join(reply_parts).strip()
        tool_results = self._run_tools(tool_calls, metrics)

        # Off-path naivety audit; correction (if any) rides the next turn.
        audit = await self.auditor.audit(user_text, reply)
        if audit.leaked and audit.correction:
            self._pending_system.append(audit.correction)

        self._record_history(user_text, reply)

        # Curiosity cadence.
        self.curiosity.note_turn()
        proactive = self.curiosity.pick()
        if proactive:
            self._pending_system.append(
                f"CURIOSITY: you may ask TJ one short question about {proactive}."
            )

        metrics.t_done = self.clock()
        self.state = SLEEPWATCH if self.sleep_watch else IDLE
        return TurnResult(
            reply=reply,
            chunks=collected,
            tool_results=tool_results,
            metrics=metrics,
            audit=audit,
            proactive_topic=proactive,
        )

    def barge_in(self) -> None:
        """Interrupt current speech.

        The primary mechanism is the generation bump + Mixer flush: the Mixer drops any
        queued voice tagged with the now-stale generation, so speech stops immediately.
        The task cancel is a best-effort guard for when a turn is driven as a background
        task (set _current_task before awaiting it); in the synchronous sim path there is
        no such task and cancel is a safe no-op.
        """
        self.generation += 1
        self.mixer.set_generation(self.generation)
        self._fire_chord("interrupted")
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

    # -- internals ---------------------------------------------------------
    def _build_messages(self, user_text: str) -> list[dict]:
        profile = build_profile(self.memory)
        messages = build_messages(profile)
        for correction in self._pending_system:
            messages.append({"role": "system", "content": correction})
        self._pending_system = []
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        return messages

    async def _speak_chunk(self, chunk: Chunk, metrics: TurnMetrics) -> None:
        for sfx in chunk.sfx:
            self._fire_chord(sfx)
        for emote in chunk.emotes:
            self._fire_emote(emote)
        if chunk.text:
            pcm = await self.tts.synthesize(chunk.text)
            if pcm.size:
                # src_rate lets the Mixer convert Piper's 22050 to the device rate
                # once at the boundary (no-op when the rates already match).
                self.mixer.push_voice(
                    pcm, self.generation, src_rate=getattr(self.tts, "sample_rate", None)
                )
                if metrics.t_first_audio is None:
                    metrics.t_first_audio = self.clock()

    def _fire_chord(self, name: str) -> None:
        if name in STINGER_BANK:
            self.mixer.push_chord(self.chords.render(name))

    def _fire_emote(self, name: str) -> None:
        if self.emote_sink is not None:
            self.emote_sink(name)

    def _run_tools(self, tool_calls: list, metrics: TurnMetrics) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        for tc in tool_calls:
            metrics.tool_calls += 1
            try:
                out = self.tools.execute(tc.name, tc.arguments)
                results.append((tc.name, out))
                if tc.name == "set_sleep_watch":
                    self.sleep_watch = "on" in out.lower() or out.lower() == "true"
            except ToolCallError:
                metrics.tool_errors += 1
        return results

    def _record_history(self, user_text: str, reply: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        # Trim in large chunks so the KV-cache prefix (system messages) stays stable.
        if len(self.history) > _HISTORY_MAX:
            self.history = self.history[_HISTORY_TRIM:]
