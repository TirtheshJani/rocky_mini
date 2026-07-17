"""Application assembly: wire components into an AppState and expose the settings UI.

create_app(state) builds the FastAPI app that backs the settings UI: live state, the
sim chat box (drives a real ConversationLoop turn), the fact table (confirm/delete),
emote buttons, an SSE metrics stream, and memory export. Defaults to sim mode (Fakes +
SimResponder) so it runs with no robot, no Ollama, and no GPU.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio.chords import ChordBank
from .audio.io import FakeAudioIO
from .audio.output import Mixer
from .brain.auditor import NaivetyAuditor
from .brain.curiosity import CuriosityScheduler
from .brain.llm import LLM, FakeLLM, OllamaLLM
from .brain.reflection import ReflectionWorker
from .brain.tools import ToolExecutor
from .config import Settings, load_settings
from .memory.store import MemoryStore
from .motion.manager import MovementManager
from .sim_brain import SimResponder
from .speech.tts import FakeTTS
from .turn import ConversationLoop, TurnMetrics, build_profile

STATIC_DIR = Path(__file__).parent / "static"


def metrics_to_dict(m: TurnMetrics) -> dict:
    return {
        "generation": m.generation,
        "ack_ms": round(m.ack_ms, 1) if m.ack_ms is not None else None,
        "time_to_first_audio_s": round(m.time_to_first_audio_s, 3)
        if m.time_to_first_audio_s is not None
        else None,
        "total_s": round(m.total_s, 3) if m.total_s is not None else None,
        "prompt_eval_count": m.prompt_eval_count,
        "eval_count": m.eval_count,
        "tool_calls": m.tool_calls,
        "tool_errors": m.tool_errors,
    }


def _build_llm(settings: Settings) -> LLM:
    """Select the LLM backend.

    Defaults to the in-process Fake (driven by SimResponder) so the sim and test path
    needs no Ollama, no openai client, and no GPU (decision 4's seam). Set
    ROCKY_LLM_BACKEND=ollama to point the running app at the real local brain; the
    openai client is imported lazily inside OllamaLLM, so it is only required on that path.
    """
    if settings.llm_backend == "ollama":
        return OllamaLLM(
            base_url=settings.llm_base_url,
            model=settings.model,
            api_key=settings.llm_api_key,
            keep_alive=settings.keep_alive,
        )
    return FakeLLM(SimResponder())


@dataclass
class AppState:
    settings: Settings
    memory: MemoryStore
    mixer: Mixer
    motion: MovementManager
    loop: ConversationLoop
    reflection: ReflectionWorker
    last_metrics: TurnMetrics | None = None
    fired_emotes: list[str] = field(default_factory=list)
    _subscribers: list[asyncio.Queue] = field(default_factory=list)

    @classmethod
    def build(cls, settings: Settings | None = None) -> "AppState":
        settings = settings or load_settings()
        memory = MemoryStore(settings.home_dir)
        io_dev = FakeAudioIO()
        mixer = Mixer(
            io_dev,
            frame=settings.mixer_frame,
            ring_carrier_hz=settings.ring_carrier_hz,
            sample_rate=settings.sample_rate_out,
        )
        motion = MovementManager()
        handlers = {
            "remember_fact": lambda a: memory.remember_fact(a.category, a.fact, a.source_quote),
            "note_open_question": lambda a: memory.note_open_question(a.question),
            "confirm_fact": lambda a: memory.confirm_fact(a.id),
            "set_sleep_watch": lambda a: _sleep_handler(motion, a.on),
        }
        state = cls(
            settings=settings,
            memory=memory,
            mixer=mixer,
            motion=motion,
            loop=None,  # set below
            reflection=ReflectionWorker(),
        )
        loop = ConversationLoop(
            llm=_build_llm(settings),
            tts=FakeTTS(sample_rate=settings.sample_rate_out),
            mixer=mixer,
            memory=memory,
            tools=ToolExecutor(handlers=handlers),
            auditor=NaivetyAuditor(),
            curiosity=CuriosityScheduler(),
            chords=ChordBank(sample_rate=settings.sample_rate_out),
            emote_sink=state._on_emote,
        )
        state.loop = loop
        return state

    # -- helpers -----------------------------------------------------------
    def _on_emote(self, name: str) -> None:
        self.fired_emotes.append(name)
        self.motion.start_emote(name)

    def snapshot(self) -> dict:
        return {
            "model": self.settings.model,
            "llm_base_url": self.settings.llm_base_url,
            "known_words": self.memory.known_words(),
            "stage": build_profile(self.memory).stage,
            "sleep_watch": self.loop.sleep_watch,
            "generation": self.loop.generation,
            "fact_count": len(self.memory.active_facts()),
            "question_count": len(self.memory.active_questions()),
            "sim_mode": self.settings.llm_backend != "ollama",
        }

    async def publish(self, event: dict) -> None:
        for q in list(self._subscribers):
            await q.put(event)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)


def _sleep_handler(motion: MovementManager, on: bool) -> str:
    motion.set_sleep_watch(on)
    return "on" if on else "off"


# -- request models --------------------------------------------------------
class ChatIn(BaseModel):
    text: str


class SettingsIn(BaseModel):
    model: str | None = None
    llm_base_url: str | None = None


class EmoteIn(BaseModel):
    name: str


def create_app(state: AppState | None = None, app: FastAPI | None = None) -> FastAPI:
    """Build the settings-UI FastAPI app, or attach the API routes to an existing one.

    When the SDK's ReachyMiniApp base class owns the webserver (custom_app_url set),
    it already serves "/" and mounts static/; pass its settings_app here and only the
    /api routes are added (audit F3/D4). With app=None (sim CLI, tests) a standalone
    app is built, unchanged behaviour.
    """
    state = state or AppState.build()
    attach_only = app is not None
    if app is None:
        app = FastAPI(title="Rocky Mini", version="0.1.0")
    app.state.rocky = state

    if not attach_only:
        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            index_file = STATIC_DIR / "index.html"
            if index_file.exists():
                return index_file.read_text(encoding="utf-8")
            return "<h1>Rocky Mini</h1><p>Settings UI not built.</p>"

    @app.get("/api/state")
    async def get_state() -> dict:
        return state.snapshot()

    @app.get("/api/facts")
    async def get_facts() -> dict:
        # active_facts() preserves first-insertion (chronological) order. Do NOT sort by
        # id string: ids like f1..f10 sort lexically, putting f10 before f2.
        facts = [
            {
                "id": f.id,
                "category": f.category,
                "text": f.text,
                "confidence": f.confidence,
                "heard_count": f.heard_count,
            }
            for f in state.memory.active_facts()
        ]
        return {"facts": facts}

    @app.post("/api/facts/{fact_id}/confirm")
    async def confirm_fact(fact_id: str) -> dict:
        try:
            conf = state.memory.confirm_fact(fact_id)
        except KeyError:
            raise HTTPException(404, "no such fact")
        return {"id": fact_id, "confidence": conf}

    @app.delete("/api/facts/{fact_id}")
    async def delete_fact(fact_id: str) -> dict:
        try:
            state.memory.delete_fact(fact_id)
        except KeyError:
            raise HTTPException(404, "no such fact")
        return {"deleted": fact_id}

    @app.post("/api/chat")
    async def chat(body: ChatIn) -> dict:
        result = await state.loop.handle_text_turn(body.text)
        state.mixer.render()  # consume queued audio (sim playout)
        if hasattr(state.mixer.io, "pushed"):
            state.mixer.io.pushed.clear()  # avoid unbounded growth in the sim FakeAudioIO
        state.last_metrics = result.metrics
        spoken = " ".join(c.text for c in result.chunks if c.text)
        payload = {
            "reply": result.reply,
            "spoken": spoken,
            "chunks": [
                {"text": c.text, "emotes": c.emotes, "sfx": c.sfx} for c in result.chunks
            ],
            "tool_results": result.tool_results,
            "proactive_topic": result.proactive_topic,
            "leaked": bool(result.audit and result.audit.leaked),
            "metrics": metrics_to_dict(result.metrics),
            "state": state.snapshot(),
        }
        await state.publish({"type": "turn", "metrics": payload["metrics"], "reply": result.reply})
        return payload

    @app.post("/api/emote")
    async def emote(body: EmoteIn) -> dict:
        state.motion.start_emote(body.name)
        state.fired_emotes.append(body.name)
        return {"fired": body.name}

    @app.post("/api/barge_in")
    async def barge_in() -> dict:
        state.loop.barge_in()
        return {"generation": state.loop.generation}

    @app.post("/api/settings")
    async def update_settings(body: SettingsIn) -> dict:
        if body.model is not None:
            state.settings.model = body.model
        if body.llm_base_url is not None:
            state.settings.llm_base_url = body.llm_base_url
        return state.snapshot()

    @app.get("/api/export")
    async def export_memory() -> FileResponse:
        zpath = state.settings.home_dir / "rocky_memory_export.zip"
        state.memory.export_zip(zpath)
        return FileResponse(str(zpath), filename="rocky_memory.zip", media_type="application/zip")

    @app.get("/api/metrics/stream")
    async def metrics_stream(request: Request) -> StreamingResponse:
        q = state.subscribe()

        async def gen():
            try:
                # Send the last known metrics immediately so the panel is not blank.
                if state.last_metrics is not None:
                    yield _sse({"type": "turn", "metrics": metrics_to_dict(state.last_metrics)})
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield _sse(event)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                state.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _sse(data: dict) -> str:
    import json

    return f"data: {json.dumps(data)}\n\n"
