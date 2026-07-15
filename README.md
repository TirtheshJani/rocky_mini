# rocky_mini

Rocky, the Eridian engineer from *Project Hail Mary*, role-played by a **Reachy Mini
Wireless**. A curious alien child on Earth: TJ teaches him about the world, he learns and
remembers across sessions, and he speaks with chords and a ring-modulated alien voice.

**Fully local. Zero paid-API cost.** The brain is Ollama-served Qwen2.5-7B-Instruct (with
a custom Rocky LoRA); STT is faster-whisper, TTS is Piper, all on TJ's PC (RTX 4080) acting
as the LAN brain server. The robot is a thin client. Development is **sim-first on Windows**.

## What this is (and honest scope)

The software core is built and **tested green (164 tests)**, and the app **runs and is
driven end-to-end in sim** (a real browser Playwright pass over the settings UI). External
services sit behind Protocols with Fakes, so the whole pipeline runs with **no robot, no
Ollama, and no GPU**:

- A rule-based `SimResponder` plays Rocky in character for the UI/sim; the real path swaps
  in `OllamaLLM`.
- The Reachy SDK, Ollama, faster-whisper, Piper, MuJoCo, and WSL2+Unsloth were **not present
  in the build environment**, so those paths are implemented behind seams and documented for
  the real machine; they are not exercised here. The `finetune/` and `server/` trees are
  faithful, runnable scaffolds (a training run needs a GPU/WSL2; the speech server needs the
  models).

See `agents.local.md` and `decisions.md` for the full deviation log.

## Architecture

One process, four threads + one asyncio loop, each owning a latency domain (full design in
`plan.md`). Structural footgun rules live in `CLAUDE.md`.

- **MotionThread (100 Hz)** - the single `set_target` owner. Layered breathing/emotes +
  additive wobble/DoA, clamped for safety (`rocky_mini/motion/`).
- **AudioIn** - mic -> VAD -> speech events (`rocky_mini/audio/io.py`).
- **Mixer (AudioOut)** - the single `push_audio_sample` owner. VoiceBus (generation-tagged
  TTS) + ChordBus (Eridian stingers), ring-mod, soft-clip (`rocky_mini/audio/output.py`).
- **ConversationLoop (asyncio)** - ack chord -> streamed LLM -> chunker (tics/tags) -> TTS
  -> Mixer; tools, naivety auditor, curiosity, memory (`rocky_mini/turn.py`).

Character (blind gaze, tics, chords, jazz-hands, sleep-watch) and learning (naivety
enforcement, JSONL memory with a confidence lifecycle, curiosity scheduler, reflection,
growth stages) are in `rocky_mini/brain/`, `rocky_mini/memory/`, and `rocky_mini/motion/`.

## Quick start (sim, no robot/Ollama/GPU)

```bash
# Python >= 3.11
pip install -e .           # core deps only: numpy, scipy, fastapi, uvicorn, pydantic
python -m rocky_mini.main --sim --host 127.0.0.1 --port 8042
# open http://127.0.0.1:8042  -> teach Rocky, watch the fact table + metrics
pytest -q                  # 164 tests
```

The settings UI: sim chat (drives a real conversation turn), fact table (confirm/delete),
emote buttons, live SSE metrics with a latency meter vs the 2.5 s budget, model toggle,
memory export.

## Full stack (real robot + brain)

1. **Brain server (PC).** Install Ollama, then:
   ```bash
   ollama pull qwen2.5:7b-instruct-q5_K_M     # ~5.5 GB, fits 16 GB VRAM
   pip install -r server/requirements.txt      # faster-whisper + Piper
   uvicorn server.app:app --host 0.0.0.0 --port 8123
   ```
2. **Config.** In `~/.rocky_mini/.env` (never in the repo) set:
   ```
   ROCKY_LLM_BASE_URL=http://<pc-lan-ip>:11434/v1
   ROCKY_SPEECH_BASE_URL=http://<pc-lan-ip>:8123
   ROCKY_MODEL=qwen2.5:7b-instruct         # or rocky:latest once the LoRA is trained
   ROCKY_AUDIO_BACKEND=reachy
   ```
3. **Robot app.** `pip install -e ".[robot,llm,vad]"`, then install to the Reachy dashboard
   (the `reachy_mini_apps` entry point). The daemon launches `RockyMiniApp` with a live
   `mini`; MotionThread and Mixer take over the hardware surfaces.
4. **Rocky LoRA (optional, $0).** See `finetune/README.md` (WSL2 + Unsloth). Train -> GGUF
   Q5_K_M -> `ollama create rocky` -> gate with `finetune/eval.py` -> flip `ROCKY_MODEL`.

## Data provenance

- Brain: Qwen2.5-7B-Instruct (Apache-2.0) served locally by Ollama; optional Rocky LoRA
  trained on authored + curated-transcript data (`finetune/data/`).
- STT: faster-whisper. TTS: Piper `en_US-lessac-medium`. No cloud, no keys.
- VAD: Silero VAD, vendored as `rocky_mini/assets/models/silero_vad.onnx` (MIT,
  Silero Team; from the silero-vad 6.2.1 wheel) and run with onnxruntime, no torch.
  See `rocky_mini/assets/models/README.md` and decisions.md #11.
- Memory lives in `~/.rocky_mini/` (survives app reinstalls); export/import zip migrates a
  sim-raised Rocky onto the robot.

## Reproduce / verify

```bash
pytest -q                          # unit + API + sim e2e integration
python finetune/eval.py --sim      # the LoRA ship gate, demoed on the sim brain
python finetune/make_seed.py       # regenerate the seed dataset
python scripts/make_canned.py      # regenerate offline fallback WAVs
# UI: start the sim server, then tests/ui/pw-rocky-ui.js via the playwright-skill
```

## Milestone status

| Milestone | Status |
|---|---|
| M0 scaffold, config, footgun rules | done |
| M1 motion core (layered, clamped, RecordedMove de-risk) | done, tested |
| M2 Eridian audio (DSP, chords, Mixer, barge-in) | done, tested |
| M3 brain loop (persona, chunker, tics, tools, sim chat) | done, tested + UI-driven |
| M4 naivety gate (20-probe metric, auditor) | done, tested |
| M4.5 Rocky LoRA | scaffolded + eval gate runnable (training needs GPU/WSL2) |
| M5 memory (JSONL, lifecycle, digest, export) | done, tested + UI-driven |
| M6 voice-in (VAD/STT/barge-in) | seams + Fakes done; real STT via brain server |
| M7 choreography & rituals (wobble, sleep-watch, curiosity) | done, tested |
| M8 hardware bring-up | seams + guarded paths documented (needs robot) |
| M9 soak & failure drills (canned fallback, Eridian-only mode) | canned path built; drills need hardware |

## License

MIT (see `LICENSE`).
