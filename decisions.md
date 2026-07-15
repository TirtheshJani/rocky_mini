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

## 2026-07-15

8. **Runtime deps are core deps; only the SDK and dev tooling stay extras.** The
   dashboard installs apps with plain `pip install` into the shared apps_venv, never
   with extras, so the `[llm]` and `[vad]` extras would silently not exist on the robot
   (no brain client, no VAD). Folded into core dependencies; imports stay lazy and
   guarded so the sim/test path still runs with none of them importable (decision 4's
   seam is unchanged, only the packaging moved). Reasoning: verified against reachy-mini
   1.9.0 `apps/sources/local_common_venv.py`. Reversal: if a future SDK installs apps
   with extras or per-app dependency config, split them back out.

9. **Decision 6 premise corrected: the Wireless has AEC; half-duplex stays the default
   anyway.** SDK 1.9.0 routes CM4 audio through the XMOS echo-cancelled loopback and
   adds webrtcdsp software AEC on PCs, so "no AEC on the PC" no longer justifies the
   default. Half-duplex remains the default because AEC quality on the real robot is
   unmeasured, and motor noise is not echo (it has no loopback reference to cancel
   against). The settings switch stays; a tuned XVF3800 startup config (from the
   official Conversation App) ships with the hardware path. Reversal: flip the default
   to open-mic after the bring-up.md AEC test passes on the physical robot (speak over
   a loud chord; VAD must not trigger on Rocky's own voice).

10. **Decision 3's reversal condition fired: scaffold re-diffed with the SDK installed.**
    Adopted the structural items: daemon launch contract in `__main__` (wrapped_run +
    KeyboardInterrupt -> stop), `run(self, reachy_mini, stop_event)` signature,
    `custom_app_url` as a literal class attribute, `keywords = ["reachy-mini-app"]`.
    Rejected as cosmetic: flat layout, wildcard package-data, scaffold landing page,
    3.10 floor. HF Space README front matter is adopted at publish time. Reversal:
    re-run the diff on SDK upgrade (same as the audit itself).

11. **Silero VAD is vendored as an ONNX file, not a package dependency.** The
    `silero-vad` PyPI package (v6+) requires torch + torchaudio, which pulls the CUDA
    stack: gigabytes of wheels on a 4 GB CM4 that plan.md explicitly avoids ("Silero
    via onnxruntime, no torch"). The 2.3 MB `silero_vad.onnx` (MIT) is vendored under
    `rocky_mini/assets/models/` with provenance in its README, and `audio/vad.py` runs
    it with onnxruntime pinned to single-threaded sessions to protect the motion loop.
    Reversal: if Silero ships a torch-free package again, or the app moves off the CM4,
    depend on the package instead.
