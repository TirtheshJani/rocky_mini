# Rocky brain speech server

FastAPI service that runs the heavy models on TJ's PC (RTX 4080): `/stt`
(faster-whisper), `/tts` (Piper, en_US-lessac-medium), and `/embed_image` (frozen DINOv2
for visual object memory). The robot app reaches it over the LAN; its deps are kept out of
the robot app's install.

## Endpoints

- `GET /health` -> `{ok, stt_loaded, tts_loaded, embed_loaded, embed_model, ...}` (works before models load)
- `POST /stt?sample_rate=16000` -> body is raw int16 PCM; returns `{"text": "..."}`
- `POST /tts` -> `{"text": "..."}`; returns raw int16 PCM bytes
- `POST /embed_image` -> body is raw JPEG bytes; returns `{"vector": [...]}` (L2-normalized, 384-dim)

`/stt` and `/tts` match `rocky_mini.speech.stt.RemoteSTT` and `rocky_mini.speech.tts.RemoteTTS`;
`/embed_image` matches `rocky_mini.vision.embed.HttpImageEmbedder` (decision 21).

## Visual object memory (`/embed_image`)

Frozen, pretrained DINOv2 ViT-S/14 (~85 MB), no training: TJ shows Rocky an object, the
robot enrolls one to three embedding shots, and later re-identifies it by cosine
similarity. The model loads lazily on the first request (`torch.hub` downloads the weights
once, PC-side). It needs `torch` and `pillow` (in `requirements.txt`). Env:

- `ROCKY_EMBED_MODEL` (default `dinov2_vits14`)
- `ROCKY_EMBED_DEVICE` (`cuda` default, or `cpu` to keep it off the GPU: ~100-300 ms/frame)

Quick check once the deps are installed:

```bash
curl -X POST --data-binary @test.jpg http://<pc-lan-ip>:8123/embed_image
# -> {"vector": [...384 floats...]}  (<100 ms GPU, <300 ms CPU)
```

## Run

```bash
pip install -r server/requirements.txt
# Piper voice: download en_US-lessac-medium.onnx (+ .json) next to where you run it.
uvicorn server.app:app --host 0.0.0.0 --port 8123
```

Then point the robot app at it: set `ROCKY_SPEECH_BASE_URL=http://<pc-lan-ip>:8123`
in `~/.rocky_mini/.env` (and `ROCKY_LLM_BASE_URL=http://<pc-lan-ip>:11434/v1` for Ollama).

## Voice backends (`/tts`)

- `ROCKY_TTS_BACKEND=clone` (movie voice): XTTS v2 zero-shot cloning against ~2 min of
  scrubbed film audio of Rocky (James Ortiz), approach borrowed from pedramamini's
  rocky_say gist. Reference WAV: `~/.rocky_say/rocky_training_audio_scrubbed.wav`
  (https://pedramamini.com/dropbox/rocky_training_audio_scrubbed.wav). Tempo via
  `ROCKY_CLONE_TEMPO` (default 1.35). Falls back to Piper on any failure.
- `ROCKY_TTS_BACKEND=piper` (default, fast): Piper with flattened prosody, plus the
  optional Eridian chord DSP (`ROCKY_TTS_FX`, pitch/fifth/octave gains env-tunable).

The server runs best in a dedicated venv (`C:\Users\TJ\rocky_speech_venv` on the PC)
because coqui-tts pins transformers.

## The LLM is separate

Ollama serves the model directly on the same PC (`ollama serve`, default port 11434).
This speech server only handles STT + TTS. See the top-level README for the full stack.
