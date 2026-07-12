# decisions.md - binding decisions for rocky_mini

Format: date | decision | reasoning | reversal condition.

## 2026-07-11

1. **Local zero-cost stack locked.** Ollama-served Qwen2.5-7B-Instruct (+ Rocky LoRA)
   for the brain, faster-whisper for STT, Piper for TTS, all on TJ's PC (RTX 4080).
   No paid APIs. Reasoning: personal project, privacy, zero recurring cost, 4080 gives
   sub-second first token. Reversal: PC unavailable as a LAN brain (then re-scope).

2. **From-scratch custom ReachyMiniApp, not a fork.** Reasoning: full control over the
   4-thread architecture and footgun avoidance. Reversal: none anticipated.

3. **Hand-scaffolded package tree (deviation).** The plan mandates
   `reachy-mini-app-assistant create`. That tool, `uv`, and the reference clone were all
   absent in the build environment. Hand-created the identical layout (entry point
   `reachy_mini_apps`, `pyproject.toml`, `static/`). Reversal: when the SDK is installed,
   re-run the assistant and diff; adopt any structural differences.

4. **External services behind Protocols with Fakes.** LLM, STT, TTS, and the robot SDK
   are Protocols. The sim/test path uses Fakes and never imports real clients (imports
   are lazy + guarded). Reasoning: the whole software core must be testable on a Windows
   PC with no robot, no Ollama, no GPU. Reversal: none; this is the seam that lets sim
   and hardware share one codebase.

5. **Audio backend seam: SoundDeviceIO (dev) vs ReachyMediaIO (hardware).** Sim defaults
   to `sounddevice` / push-to-talk. Insulates Windows dev from GStreamer fragility.
   Reversal: none.

6. **Half-duplex barge-in by default.** No AEC on the PC; open-mic barge-in ships off by
   default with a settings switch, to be tuned at hardware bring-up (M8).

7. **Naivety is a thresholded regression metric, not a hard zero-leak gate.** 20-probe
   red-team suite tracked over time; LoRA must beat the stock+prompt baseline. Reasoning:
   a flaky zero-leak gate would block all progress. Reversal: none.
