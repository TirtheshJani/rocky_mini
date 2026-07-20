"""Rocky brain speech server: /stt (faster-whisper) + /tts (Piper).

Runs on TJ's PC (RTX 4080). The robot app reaches it over the LAN. Kept OUT of the robot
app's install (its heavy deps live only here). faster-whisper and Piper are lazy-loaded
and import-guarded, so /health works even before the models are present and the file
imports on any machine.

Run:  uvicorn server.app:app --host 0.0.0.0 --port 8123
"""

from __future__ import annotations

import io
import wave

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

app = FastAPI(title="Rocky brain speech server", version="0.1.0")

_WHISPER = None
_PIPER = None
WHISPER_SIZE = "small"
PIPER_VOICE = "en_US-lessac-medium"


def get_whisper():  # pragma: no cover - needs the model + GPU
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel

        _WHISPER = WhisperModel(WHISPER_SIZE, device="cuda", compute_type="float16")
    return _WHISPER


def get_piper():  # pragma: no cover - needs the voice files
    global _PIPER
    if _PIPER is None:
        from piper import PiperVoice

        _PIPER = PiperVoice.load(f"{PIPER_VOICE}.onnx")
    return _PIPER


class TTSIn(BaseModel):
    text: str


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "stt_loaded": _WHISPER is not None,
            "tts_loaded": _PIPER is not None,
            "whisper_size": WHISPER_SIZE,
            "piper_voice": PIPER_VOICE,
        }
    )


@app.post("/stt")
async def stt(request: Request, sample_rate: int = 16000) -> dict:  # pragma: no cover - GPU path
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "empty audio body")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        model = get_whisper()
    except Exception as exc:
        raise HTTPException(503, f"STT model unavailable: {exc}")
    segments, _ = model.transcribe(audio, language="en", vad_filter=True)
    text = " ".join(s.text for s in segments).strip()
    return {"text": text}


@app.post("/tts")
async def tts(body: TTSIn) -> Response:  # pragma: no cover - needs the voice
    try:
        voice = get_piper()
    except Exception as exc:
        raise HTTPException(503, f"TTS voice unavailable: {exc}")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        # piper-tts >= 1.3 renamed the wav-writing entry point to synthesize_wav
        # (synthesize now yields raw AudioChunks and does not set wave params).
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(body.text, wav)
        else:
            voice.synthesize(body.text, wav)
    buf.seek(0)
    with wave.open(buf, "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return Response(content=frames, media_type="application/octet-stream")
