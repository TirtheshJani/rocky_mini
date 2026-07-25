"""Rocky brain speech server: /stt (faster-whisper) + /tts (Piper).

Runs on TJ's PC (RTX 4080). The robot app reaches it over the LAN. Kept OUT of the robot
app's install (its heavy deps live only here). faster-whisper and Piper are lazy-loaded
and import-guarded, so /health works even before the models are present and the file
imports on any machine.

Run:  uvicorn server.app:app --host 0.0.0.0 --port 8123
"""

from __future__ import annotations

import io
import os
import wave

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

app = FastAPI(title="Rocky brain speech server", version="0.1.0")

_WHISPER = None
_PIPER = None
_DINO = None
_XTTS = None
_CHATTERBOX = None
WHISPER_SIZE = os.environ.get("ROCKY_WHISPER_SIZE", "small")
PIPER_VOICE = os.environ.get("ROCKY_PIPER_VOICE", "en_US-lessac-medium")

# Visual object memory embedder (decision 21): frozen DINOv2, no training. Lives here
# because Ollama has no image-embedding API. ViT-S/14 is ~85 MB and its output is 384-dim.
# ROCKY_EMBED_DEVICE=cpu offloads it from VRAM if the GPU is tight (~100-300 ms on CPU).
EMBED_MODEL = os.environ.get("ROCKY_EMBED_MODEL", "dinov2_vits14")
EMBED_DEVICE = os.environ.get("ROCKY_EMBED_DEVICE", "cuda")

# ImageNet normalization, DINOv2's training convention.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Movie Rocky delivery: deadpan, slightly slow, low prosody variation. Piper's
# noise_scale controls acoustic variation and noise_w controls phoneme-duration
# variation; lowering both flattens the read toward a computer-voice-box sound.
TTS_LENGTH_SCALE = float(os.environ.get("ROCKY_TTS_LENGTH_SCALE", "1.05"))
TTS_NOISE_SCALE = float(os.environ.get("ROCKY_TTS_NOISE_SCALE", "0.3"))
TTS_NOISE_W = float(os.environ.get("ROCKY_TTS_NOISE_W", "0.4"))

# Movie-Rocky voice clone (TJ, 2026-07-23): XTTS v2 zero-shot cloning against scrubbed
# film audio of James Ortiz's Rocky (approach from pedramamini's rocky_say gist). This is
# the backend that actually sounds like the movie; Piper+DSP stays as the fast fallback.
# ROCKY_TTS_BACKEND=piper reverts. Tempo mirrors the gist's ffmpeg atempo default.
TTS_BACKEND = os.environ.get("ROCKY_TTS_BACKEND", "piper")
CLONE_MODEL = os.environ.get("ROCKY_CLONE_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
CLONE_REF = os.environ.get(
    "ROCKY_CLONE_REF", os.path.expanduser("~/.rocky_say/rocky_training_audio_scrubbed.wav")
)
CLONE_TEMPO = float(os.environ.get("ROCKY_CLONE_TEMPO", "1.35"))
CLONE_RATE_OUT = 22050  # match the robot client's expected sample rate


def get_xtts():  # pragma: no cover - needs coqui-tts + the model download
    global _XTTS
    if _XTTS is None:
        os.environ.setdefault("COQUI_TOS_AGREED", "1")  # XTTS CPML license prompt
        from TTS.api import TTS as CoquiTTS

        _XTTS = CoquiTTS(CLONE_MODEL).to(EMBED_DEVICE)
    return _XTTS


def _clone_tts(text: str) -> bytes:  # pragma: no cover - needs the model
    """Synthesize with the Rocky voice clone; returns int16 PCM at CLONE_RATE_OUT."""
    import librosa

    model = get_xtts()
    wav = np.asarray(model.tts(text=text, speaker_wav=CLONE_REF, language="en"), dtype=np.float32)
    sr = model.synthesizer.output_sample_rate
    if CLONE_TEMPO != 1.0:
        wav = librosa.effects.time_stretch(wav, rate=CLONE_TEMPO)
    if sr != CLONE_RATE_OUT:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=CLONE_RATE_OUT)
    peak = float(np.max(np.abs(wav))) or 1.0
    if peak > 0.95:
        wav = wav * (0.95 / peak)
    return (wav * 32767.0).astype(np.int16).tobytes()


# Chatterbox clone backend (round 2, after TJ rejected XTTS as still robotic): Resemble
# AI's Chatterbox is a newer open model with far more natural prosody. Same movie
# reference, short clean cut. Needs the chatterbox-tts package (its transformers pin
# conflicts with coqui's, so a serving venv carries one clone backend, not both).
# TJ picked the "early" cut (5-17 s of the scrubbed film audio) with these exact
# settings on 2026-07-23; the loudest-window auto-pick drifted accent, so keep this ref.
CHATTERBOX_REF = os.environ.get(
    "ROCKY_CHATTERBOX_REF", os.path.expanduser("~/.rocky_say/rocky_ref_early.wav")
)
CHATTERBOX_EXAGGERATION = float(os.environ.get("ROCKY_CHATTERBOX_EXAGGERATION", "0.5"))
CHATTERBOX_CFG = float(os.environ.get("ROCKY_CHATTERBOX_CFG", "0.6"))
# Distinct from ROCKY_EMBED_DEVICE: DINO can afford CPU, the voice cannot (40 s vs ~2 s).
CHATTERBOX_DEVICE = os.environ.get("ROCKY_CHATTERBOX_DEVICE", "cuda")


def get_chatterbox():  # pragma: no cover - needs chatterbox-tts + the model download
    global _CHATTERBOX
    if _CHATTERBOX is None:
        from chatterbox.tts import ChatterboxTTS

        _CHATTERBOX = ChatterboxTTS.from_pretrained(device=CHATTERBOX_DEVICE)
    return _CHATTERBOX


def _chatterbox_tts(text: str) -> bytes:  # pragma: no cover - needs the model
    """Synthesize with the Chatterbox Rocky clone; returns int16 PCM at CLONE_RATE_OUT."""
    import librosa

    model = get_chatterbox()
    wav = model.generate(
        text,
        audio_prompt_path=CHATTERBOX_REF,
        exaggeration=CHATTERBOX_EXAGGERATION,
        cfg_weight=CHATTERBOX_CFG,
    )
    wav = wav.squeeze(0).cpu().numpy().astype(np.float32)
    if model.sr != CLONE_RATE_OUT:
        wav = librosa.resample(wav, orig_sr=model.sr, target_sr=CLONE_RATE_OUT)
    peak = float(np.max(np.abs(wav))) or 1.0
    if peak > 0.95:
        wav = wav * (0.95 / peak)
    return (wav * 32767.0).astype(np.int16).tobytes()


# Eridian voice-box (movie Rocky): the translated voice is flat and synthesized through
# Rocky's musical chords. Post-process Piper output: pitch the whole read down a little
# and blend quiet copies a fifth and an octave below, so every utterance carries a chord.
# ROCKY_TTS_FX=0 returns the dry Piper voice.
TTS_FX_ENABLED = os.environ.get("ROCKY_TTS_FX", "1") not in ("0", "false", "False")
TTS_FX_PITCH = float(os.environ.get("ROCKY_TTS_FX_PITCH", "-2"))
TTS_FX_FIFTH_GAIN = float(os.environ.get("ROCKY_TTS_FX_FIFTH_GAIN", "0.22"))
TTS_FX_OCTAVE_GAIN = float(os.environ.get("ROCKY_TTS_FX_OCTAVE_GAIN", "0.14"))


def _rocky_voicebox(pcm16: bytes, sample_rate: int) -> bytes:  # pragma: no cover - needs librosa
    """Apply the Eridian chord timbre; on any failure return the dry voice unchanged."""
    try:
        import librosa

        y = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        lead = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=TTS_FX_PITCH)
        fifth = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=TTS_FX_PITCH - 7)
        octave = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=TTS_FX_PITCH - 12)
        mix = lead + TTS_FX_FIFTH_GAIN * fifth + TTS_FX_OCTAVE_GAIN * octave
        peak = float(np.max(np.abs(mix))) or 1.0
        if peak > 0.95:
            mix *= 0.95 / peak
        return (mix * 32767.0).astype(np.int16).tobytes()
    except Exception:
        return pcm16


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


def get_dino():  # pragma: no cover - needs torch + the hub weights
    global _DINO
    if _DINO is None:
        import torch

        # First call downloads the weights via torch.hub (PC-side, one time).
        model = torch.hub.load("facebookresearch/dinov2", EMBED_MODEL)
        model.eval().to(EMBED_DEVICE)
        _DINO = model
    return _DINO


@app.on_event("startup")
def _warm_models() -> None:  # pragma: no cover - needs the models
    """Pre-load whisper/piper/dino in a background thread at boot.

    Lazy-loading alone means the first robot turn after a server restart pays the
    full model load and times out client-side (httpx.ReadTimeout, seen 2026-07-23).
    /health stays responsive while this runs.
    """
    import threading

    def load() -> None:
        # Chatterbox loads first: it is the biggest torch stack, and loading it after
        # ctranslate2 (whisper) has bound its cudnn once crashed the process outright.
        if TTS_BACKEND == "chatterbox":
            try:
                _chatterbox_tts("Rocky online.")
                print("warm-up: chatterbox clone loaded")
            except Exception as exc:
                print(f"warm-up: chatterbox failed to load: {exc}")
        for name, fn in (("whisper", get_whisper), ("piper", get_piper), ("dino", get_dino)):
            try:
                fn()
            except Exception as exc:
                print(f"warm-up: {name} failed to load: {exc}")
        if TTS_FX_ENABLED:
            # librosa's numba JIT costs ~30 s on first pitch_shift; pay it here, not on
            # Rocky's first spoken turn.
            _rocky_voicebox(np.zeros(22050, dtype=np.int16).tobytes(), 22050)
            print("warm-up: voicebox FX jitted")
        if TTS_BACKEND == "clone":
            try:
                _clone_tts("Rocky online.")
                print("warm-up: voice clone loaded")
            except Exception as exc:
                print(f"warm-up: voice clone failed to load: {exc}")

    threading.Thread(target=load, name="model-warmup", daemon=True).start()


class TTSIn(BaseModel):
    text: str


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "stt_loaded": _WHISPER is not None,
            "tts_loaded": _PIPER is not None,
            "embed_loaded": _DINO is not None,
            "whisper_size": WHISPER_SIZE,
            "piper_voice": PIPER_VOICE,
            "embed_model": EMBED_MODEL,
            "tts_length_scale": TTS_LENGTH_SCALE,
            "tts_noise_scale": TTS_NOISE_SCALE,
            "tts_noise_w": TTS_NOISE_W,
            "tts_fx": TTS_FX_ENABLED,
            "tts_fx_pitch": TTS_FX_PITCH,
            "tts_backend": TTS_BACKEND,
            "clone_loaded": _XTTS is not None,
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


@app.post("/embed_image")
async def embed_image(request: Request) -> dict:  # pragma: no cover - needs torch + PIL
    """Embed a raw JPEG with frozen DINOv2 for visual object memory (decision 21).

    Body is the raw JPEG bytes; returns {"vector": [...]} L2-normalized server-side, so
    the robot client's cosine becomes a plain dot product with stable threshold semantics.
    torch and PIL are imported lazily and only after the model loads, so /health and the
    other endpoints work on a machine without them.
    """
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "empty image body")
    try:
        model = get_dino()
    except Exception as exc:
        raise HTTPException(503, f"embed model unavailable: {exc}")
    import io as _io

    import torch
    from PIL import Image

    img = Image.open(_io.BytesIO(raw)).convert("RGB")
    # DINOv2 eval convention: resize shortest side to 256, center-crop 224.
    w, h = img.size
    scale = 256.0 / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    w, h = img.size
    left = (w - 224) // 2
    top = (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(_IMAGENET_MEAN, dtype=np.float32)) / np.array(
        _IMAGENET_STD, dtype=np.float32
    )
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(EMBED_DEVICE)
    with torch.no_grad():
        feats = model(tensor)
    vec = feats[0].float()
    vec = vec / (vec.norm() + 1e-12)  # L2-normalize server-side.
    return {"vector": vec.cpu().tolist()}


@app.post("/tts")
async def tts(body: TTSIn) -> Response:  # pragma: no cover - needs the voice
    if TTS_BACKEND in ("clone", "chatterbox"):
        try:
            synth = _chatterbox_tts if TTS_BACKEND == "chatterbox" else _clone_tts
            return Response(content=synth(body.text), media_type="application/octet-stream")
        except Exception:
            import traceback

            traceback.print_exc()
            # Fall through to Piper so the robot is never voiceless.
    try:
        voice = get_piper()
    except Exception as exc:
        raise HTTPException(503, f"TTS voice unavailable: {exc}")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        # piper-tts >= 1.3 renamed the wav-writing entry point to synthesize_wav
        # (synthesize now yields raw AudioChunks and does not set wave params).
        if hasattr(voice, "synthesize_wav"):
            syn_config = None
            try:
                from piper import SynthesisConfig

                syn_config = SynthesisConfig(
                    length_scale=TTS_LENGTH_SCALE,
                    noise_scale=TTS_NOISE_SCALE,
                    noise_w_scale=TTS_NOISE_W,
                )
            except (ImportError, TypeError):
                pass  # older 1.3.x without SynthesisConfig: default prosody
            if syn_config is not None:
                voice.synthesize_wav(body.text, wav, syn_config=syn_config)
            else:
                voice.synthesize_wav(body.text, wav)
        else:
            voice.synthesize(
                body.text,
                wav,
                length_scale=TTS_LENGTH_SCALE,
                noise_scale=TTS_NOISE_SCALE,
                noise_w=TTS_NOISE_W,
            )
    buf.seek(0)
    with wave.open(buf, "rb") as wav:
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if TTS_FX_ENABLED:
        frames = _rocky_voicebox(frames, rate)
    return Response(content=frames, media_type="application/octet-stream")
