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
import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from .audio.chords import STINGER_BANK, ChordBank
from .audio.output import Mixer
from .brain.auditor import AuditResult, NaivetyAuditor
from .brain.chunker import Chunk, SentenceChunker
from .brain.curiosity import CuriosityScheduler
from .brain.llm import LLM
from .brain.persona import PersonaProfile, build_messages, stage_for_words
from .brain.tools import ToolCallError, ToolExecutor
from .memory.store import MemoryStore
from .motion.emotes import ROCKY_EMOTES
from .speech.stt import STT
from .speech.tts import TTS

logger = logging.getLogger("rocky_mini.turn")

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
        fallback_tts: TTS | None = None,
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
        self.fallback_tts = fallback_tts
        self.emote_sink = emote_sink
        self.clock = clock
        self._tts_down = False  # per-turn: first TTS failure plays a canned clip, rest skip
        # Clean emote voice clips (Mixer adds ring mod at playout). Empty in sim.
        self._emote_sounds: dict[str, tuple[np.ndarray, int]] = self._load_emote_sounds()

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
        try:
            user_text = await self.stt.transcribe(audio, sample_rate)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Brain server unreachable or STT died. Say so in character instead of
            # going silent, and keep the loop alive for the next utterance.
            logger.exception("STT failed; playing canned fallback")
            await self._play_canned("cannot_reach.wav")
            self.state = SLEEPWATCH if self.sleep_watch else IDLE
            return TurnResult(reply="")
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
        self._tts_down = False
        try:
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
        except asyncio.CancelledError:
            raise
        except Exception:
            # LLM stream died mid-turn (Ollama down, network drop). Keep whatever
            # already played, say the signal is bad if nothing did, and stay alive.
            logger.exception("LLM stream failed; degrading turn")
            if not reply_parts:
                await self._play_canned("signal_bad.wav")

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
        if chunk.text and not self._tts_down:
            try:
                pcm = await self.tts.synthesize(chunk.text)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Speech server unreachable: announce once in character, then stop
                # trying for the rest of this turn (each attempt costs a timeout).
                logger.exception("TTS failed; playing canned fallback")
                self._tts_down = True
                await self._play_canned("pc_asleep.wav")
                return
            if pcm.size:
                # src_rate lets the Mixer convert Piper's 22050 to the device rate
                # once at the boundary (no-op when the rates already match).
                self.mixer.push_voice(
                    pcm, self.generation, src_rate=getattr(self.tts, "sample_rate", None)
                )
                if metrics.t_first_audio is None:
                    metrics.t_first_audio = self.clock()

    async def _play_canned(self, clip: str) -> None:
        """Best-effort in-character fallback audio; never raises."""
        if self.fallback_tts is None:
            return
        try:
            pcm = await self.fallback_tts.synthesize(clip)
        except Exception:
            logger.exception("canned fallback failed")
            return
        if pcm.size:
            self.mixer.push_voice(
                pcm, self.generation, src_rate=getattr(self.fallback_tts, "sample_rate", None)
            )

    def _fire_chord(self, name: str) -> None:
        if name in STINGER_BANK:
            self.mixer.push_chord(self.chords.render(name))

    def _load_emote_sounds(self) -> dict[str, tuple[np.ndarray, int]]:
        """Preload clean emote voice clips from the CannedTTS canned_dir/emotes.

        Returns name -> (float32 mono PCM, framerate). On hardware fallback_tts is a
        CannedTTS with a canned_dir; in sim it is None, so the dict stays empty and
        emotes remain motion-only. A missing directory or file is skipped, not raised.
        """
        sounds: dict[str, tuple[np.ndarray, int]] = {}
        canned_dir = getattr(self.fallback_tts, "canned_dir", None)
        if canned_dir is None:
            return sounds
        emotes_dir = Path(canned_dir) / "emotes"
        if not emotes_dir.is_dir():
            return sounds
        for name in ROCKY_EMOTES:
            path = emotes_dir / f"{name}.wav"
            if not path.exists():
                continue
            try:
                with wave.open(str(path), "rb") as w:
                    frames = w.readframes(w.getnframes())
                    rate = w.getframerate()
                    nchannels = w.getnchannels()
                pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if nchannels > 1:
                    pcm = pcm.reshape(-1, nchannels).mean(axis=1).astype(np.float32)
                sounds[name] = (pcm, rate)
            except Exception:  # noqa: BLE001 - a bad clip must not break startup
                logger.exception("failed to load emote clip %s", path)
        logger.info("loaded %d emote voice clips", len(sounds))
        return sounds

    def _fire_emote(self, name: str) -> None:
        if self.emote_sink is not None:
            self.emote_sink(name)
        # Play the matching clean voice clip if one is loaded (Mixer ring-mods it at
        # playout). Tagged with the current generation, so barge-in drops it. No
        # stinger fallback here: callers pair _fire_emote with _fire_chord already.
        clip = self._emote_sounds.get(name)
        if clip is not None:
            pcm, rate = clip
            self.mixer.push_voice(pcm, self.generation, src_rate=rate)

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
