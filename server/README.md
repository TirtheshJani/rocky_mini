# Rocky brain speech server

FastAPI service that runs the heavy speech models on TJ's PC (RTX 4080): `/stt`
(faster-whisper) and `/tts` (Piper, en_US-lessac-medium). The robot app reaches it over
the LAN; its deps are kept out of the robot app's install.

## Endpoints

- `GET /health` -> `{ok, stt_loaded, tts_loaded, ...}` (works before models load)
- `POST /stt?sample_rate=16000` -> body is raw int16 PCM; returns `{"text": "..."}`
- `POST /tts` -> `{"text": "..."}`; returns raw int16 PCM bytes

These match `rocky_mini.speech.stt.RemoteSTT` and `rocky_mini.speech.tts.RemoteTTS`.

## Run

```bash
pip install -r server/requirements.txt
# Piper voice: download en_US-lessac-medium.onnx (+ .json) next to where you run it.
uvicorn server.app:app --host 0.0.0.0 --port 8123
```

Then point the robot app at it: set `ROCKY_SPEECH_BASE_URL=http://<pc-lan-ip>:8123`
in `~/.rocky_mini/.env` (and `ROCKY_LLM_BASE_URL=http://<pc-lan-ip>:11434/v1` for Ollama).

## The LLM is separate

Ollama serves the model directly on the same PC (`ollama serve`, default port 11434).
This speech server only handles STT + TTS. See the top-level README for the full stack.
