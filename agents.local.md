# agents.local.md - rocky_mini environment and session context

## Robot

- Model: Reachy Mini **Wireless** (Raspberry Pi CM4, 4 GB).
- Role at runtime: thin client. VAD + motion + audio mixing only. No local inference.

## Brain server (TJ's PC)

- GPU: RTX 4080, 16 GB VRAM.
- Ollama serves `qwen2.5:7b-instruct-q5_K_M` (~5.5 GB) with `keep_alive=-1`.
- FastAPI speech service exposes `/stt` (faster-whisper) and `/tts` (Piper).
- In sim everything is localhost; on hardware the robot reaches the PC over LAN.

## Dev environment (as built)

- Windows 11, Python 3.13. Sim-first.
- Installed and used by the tested core: numpy, scipy, fastapi, uvicorn, pydantic,
  pydantic-settings, pytest, pytest-asyncio, pytest-mock.
- Optional (documented, not required for the sim/test path): reachy-mini SDK, ollama,
  openai client, onnxruntime + silero-vad, sounddevice.

## Deviations from plan (honest log)

- The plan calls for scaffolding via `reachy-mini-app-assistant create`. Neither `uv`
  nor the reachy SDK was installed in this environment and there is no reference clone,
  so the package tree was hand-created with the same layout the assistant produces
  (entry point `reachy_mini_apps`, `pyproject.toml`, `static/`). Logged in decisions.md.
- Ollama / faster-whisper / Piper / MuJoCo / WSL2+Unsloth are not present here, so
  those paths are built behind Protocols and exercised with Fakes. Real-client run
  instructions are in README.md and server/README.md; finetune/ is scaffolded but not
  executed (no GPU here).

## SDK pin (hardware audit, 2026-07-15)

- reachy-mini **1.9.0** (PyPI) installed and audited against; every finding in
  `docs/hardware-audit.md` is specific to this version. Re-audit on upgrade.
- Audited on Linux/Python 3.11 with the daemon's mockup backend
  (`reachy-mini-daemon --mockup-sim --headless --no-media --autostart`); live probes
  in the audit's evidence appendix were run against that daemon.

## Local-machine session (2026-07-20): LoRA-first + real-brain wiring

Done here (no robot, no GPU needed; all tests green with and without SDK/Ollama):
- Wired the real LLM behind `ROCKY_LLM_BACKEND` (default still Fake). decisions.md 13.
- Blended Rocky's voice (kept tics, loosened rigidity, added warmth). decisions.md 12.
- Built the LoRA dataset pipeline: `make_seed.py` (authored gold), `neutral.py`
  (anti-forgetting replay), `synth.py` (off-novel bulk via Ollama), `mine_book.py`
  (paraphrased book calibration + memorization probes), `build_dataset.py` (QC that
  reuses the runtime character utilities). Data is gitignored and local. decisions.md 14.
- eval.py is now the full ship gate: A/B baseline, memorization probe, capability
  regression. train.py tuned (2 epochs, num_proc=1). Merge-then-quantize serve runbook in
  finetune/README. decisions.md 15.
- scripts/check_kv_reuse.py closes the audit's INCONCLUSIVE footgun-4 gap.

Needs TJ's machine (documented in docs/tutorial.md and docs/bring-up.md):
- Ollama pull + stock baseline turn; synth/mine_book bulk generation to reach 250-350;
  WSL2 QLoRA train + merge + `ollama create rocky`; run the eval gate; flip ROCKY_MODEL.
- Speech server real STT/TTS models; `scripts/measure_latency.py` real numbers;
  `scripts/check_kv_reuse.py` verdict. These produce the honest latency figures the
  README still states as a budget.

## Footgun reminders

See CLAUDE.md. Most load-bearing: one set_target owner, one push_audio_sample owner,
byte-stable persona, memory in ~/.rocky_mini, no paid APIs, no face tracking.
