"""Phase 3 latency harness: real turns against live Ollama + the speech server.

Run on the PC with the 4080 (both services up):

    ollama pull qwen2.5:7b-instruct-q5_K_M && ollama serve   # or the service
    uvicorn server.app:app --host 0.0.0.0 --port 8123
    python scripts/measure_latency.py --turns 8

Honest timing (footgun 7): each turn's clock starts at the user's last voiced
frame, and the harness actually sleeps the 0.6 s VAD hangover between that anchor
and the turn starting, exactly as the live gate does. Numbers quoted from this
harness therefore include the hangover.

The spoken input is synthesized by the speech server's own /tts and fed back
through /stt, so STT cost is measured on realistic speech, not beeps.

KV-cache prefix reuse (footgun 4): the report prints prompt_eval_count per turn,
but on Ollama 0.32.x that field reports the request's total prompt size whether or
not the prefix was cached, so a flat count does NOT mean reuse is broken. The
authoritative check is scripts/check_kv_reuse.py, which verdicts on the native
/api/chat prompt_eval_duration (turn 2 prefill collapses on a cache hit).

Memory writes go to a temp dir: a measurement run must not teach Rocky junk facts.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np

from rocky_mini.audio.chords import ChordBank
from rocky_mini.audio.io import FakeAudioIO
from rocky_mini.audio.output import Mixer
from rocky_mini.brain.auditor import NaivetyAuditor
from rocky_mini.brain.curiosity import CuriosityScheduler
from rocky_mini.brain.llm import OllamaLLM
from rocky_mini.brain.tools import ToolExecutor
from rocky_mini.config import load_settings
from rocky_mini.memory.store import MemoryStore
from rocky_mini.speech.stt import RemoteSTT
from rocky_mini.speech.tts import RemoteTTS
from rocky_mini.turn import ConversationLoop

QUESTIONS = [
    "What do you think clouds are made of, Rocky?",
    "I fixed the leaky faucet in the kitchen today.",
    "Do you remember what I told you about my sister?",
    "Why do humans sleep at night?",
    "The power went out for an hour this afternoon.",
    "What would you like to learn about tomorrow?",
    "I think the neighbor's dog is afraid of thunder.",
    "Can you explain what you find strange about water?",
]

HANGOVER_S = 0.6


class _Timed:
    """Observation wrapper: record how long the wrapped async call takes."""

    def __init__(self, inner, method: str) -> None:
        self._inner = inner
        self._method = method
        self.samples: list[float] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def _call(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = await getattr(self._inner, self._method)(*args, **kwargs)
        self.samples.append(time.perf_counter() - t0)
        return out


class TimedSTT(_Timed):
    def __init__(self, inner) -> None:
        super().__init__(inner, "transcribe")

    async def transcribe(self, audio, sample_rate):
        return await self._call(audio, sample_rate)


class TimedTTS(_Timed):
    def __init__(self, inner) -> None:
        super().__init__(inner, "synthesize")

    async def synthesize(self, text):
        return await self._call(text)


async def tts_wav(client: httpx.AsyncClient, base_url: str, text: str) -> tuple[np.ndarray, int]:
    """Ask the speech server to speak the question, for feeding back into /stt."""
    r = await client.post(f"{base_url}/tts", json={"text": text}, timeout=60.0)
    r.raise_for_status()
    # The server returns raw int16 PCM frames (same contract as speech/tts.py's
    # runtime client), not a WAV container. Piper medium voices run at 22050 Hz.
    pcm = np.frombuffer(r.content, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, 22050


def p50(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument(
        "--text-only", action="store_true",
        help="skip STT (no speech server needed); measures LLM + TTS only",
    )
    args = parser.parse_args()

    settings = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="rocky_latency_"))
    memory = MemoryStore(tmp)
    mixer = Mixer(FakeAudioIO(), sample_rate=settings.sample_rate_out)
    llm = OllamaLLM(
        base_url=settings.llm_base_url,
        model=settings.model,
        api_key=settings.llm_api_key,
        keep_alive=settings.keep_alive,
    )
    stt = TimedSTT(RemoteSTT(settings.speech_base_url, token=settings.lan_token))
    tts = TimedTTS(RemoteTTS(settings.speech_base_url, token=settings.lan_token))
    # Same tool handlers as the app, so turns that call tools measure honestly
    # (they write to the throwaway memory dir, never to ~/.rocky_mini).
    from rocky_mini.motion.manager import MovementManager

    motion = MovementManager()
    handlers = {
        "remember_fact": lambda a: memory.remember_fact(a.category, a.fact, a.source_quote),
        "note_open_question": lambda a: memory.note_open_question(a.question),
        "confirm_fact": lambda a: memory.confirm_fact(a.id),
        "set_sleep_watch": lambda a: (motion.set_sleep_watch(a.on), "on" if a.on else "off")[1],
    }
    loop = ConversationLoop(
        llm=llm, tts=tts, mixer=mixer, memory=memory,
        tools=ToolExecutor(handlers=handlers),
        auditor=NaivetyAuditor(), curiosity=CuriosityScheduler(),
        chords=ChordBank(sample_rate=settings.sample_rate_out),
        stt=stt,
    )

    print(f"model={settings.model}  llm={settings.llm_base_url}  speech={settings.speech_base_url}")
    print(f"memory (throwaway): {tmp}")
    rows = []
    async with httpx.AsyncClient() as client:
        for i in range(args.turns):
            text = QUESTIONS[i % len(QUESTIONS)]
            if args.text_only:
                audio, rate = None, None
            else:
                audio, rate = await tts_wav(client, settings.speech_base_url, text)

            t_last_voiced = time.perf_counter()  # footgun 7 anchor
            await asyncio.sleep(HANGOVER_S)  # the hangover really elapses
            if args.text_only:
                result = await loop.handle_text_turn(text, t_start=t_last_voiced)
            else:
                result = await loop.handle_voice_turn(audio, rate, t_start=t_last_voiced)
            mixer.render()
            m = result.metrics
            first_token = (m.t_first_token - m.t_start) if m.t_first_token else float("nan")
            rows.append(
                dict(
                    turn=i + 1,
                    stt=stt.samples[-1] if (not args.text_only and stt.samples) else 0.0,
                    first_token=first_token,
                    first_audio=m.time_to_first_audio_s or float("nan"),
                    total=m.total_s or float("nan"),
                    prompt_eval=m.prompt_eval_count,
                    eval=m.eval_count,
                )
            )
            r = rows[-1]
            print(
                f"turn {r['turn']:>2}: stt={r['stt']:.3f}s first_token={r['first_token']:.3f}s "
                f"first_audio={r['first_audio']:.3f}s total={r['total']:.3f}s "
                f"prompt_eval={r['prompt_eval']} eval={r['eval']}"
            )

    print("\n== p50 over", len(rows), "turns (all measured from last voiced frame, hangover included) ==")
    print(f"STT:            {p50([r['stt'] for r in rows]):.3f} s")
    print(f"first token:    {p50([r['first_token'] for r in rows]):.3f} s")
    print(f"first audio:    {p50([r['first_audio'] for r in rows]):.3f} s")
    print(f"total:          {p50([r['total'] for r in rows]):.3f} s   (budget: {settings.latency_p50_budget_s} s)")
    print(f"TTS per chunk:  {p50(tts.samples):.3f} s")
    pe = [r["prompt_eval"] for r in rows]
    print(f"prompt_eval_count per turn: {pe}")
    if len(pe) >= 2 and pe[0] > 0 and 0 < pe[1] < pe[0] * 0.5:
        print("KV-cache prefix reuse: LOOKS LIVE (turn 2 ingested far fewer prompt tokens)")
    else:
        print(
            "KV-cache prefix reuse: INCONCLUSIVE from usage counters (prompt_eval_count "
            "reports total prompt size on 0.32.x); run scripts/check_kv_reuse.py for the "
            "duration-based verdict"
        )


if __name__ == "__main__":
    asyncio.run(main())
