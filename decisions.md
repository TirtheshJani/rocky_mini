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

11. **Silero VAD is vendored as an ONNX file, not a package dependency.** Every
    published `silero-vad` wheel declares `Requires-Dist: torch` + `torchaudio`
    (checked: 6.2.1 and 5.1.2 both do), which pulls the CUDA stack: gigabytes of
    wheels on a 4 GB CM4 that plan.md explicitly avoids ("Silero via onnxruntime,
    no torch"). Pinning `<6` was considered and rejected on that evidence. The 2.3 MB
    `silero_vad.onnx` (MIT) is vendored under `rocky_mini/assets/models/` with
    provenance in its README, and `audio/vad.py` runs it with onnxruntime pinned to
    single-threaded sessions to protect the motion loop. Side benefit, not the
    reason: the model ships with the app, so a robot that boots without network
    still has ears. Reversal: if Silero ships a torch-free package, or the app moves
    off the CM4, depend on the package instead.

## 2026-07-20

12. **Rocky's voice is blended: signature tics kept, telegraphic rigidity loosened,
    warmth added.** The persona (`brain/persona.py`) keeps every tic (the ', question?'
    particle, word-tripling, 'Rocky fix.', clipped 'Thank.'/'Understand.', the epistemic
    ledger) but drops the hard under-sixty-word cap and the seventy-percent article-drop
    in favour of fuller, warmer sentences and an explicit friendship register. Reasoning:
    TJ's call; the pure telegraphic style read as colder than the novel's Rocky, and the
    LoRA can carry a warmer voice without a larger prompt. This edits the character-system
    spec in plan.md (article-drop ~70%, <60 words). tics.py is unchanged: it still forces
    the particle (kept) and only measures article-drop (now a soft target). The eval gate's
    particle threshold stays. Reversal: if the warmer voice measurably raises naivety leaks
    or weakens the tics in eval.py, tighten the speech rules back toward telegraphic.

13. **The running app selects its LLM via a config flag; the Fake stays the default.**
    `AppState.build` now calls `_build_llm(settings)`: `ROCKY_LLM_BACKEND=ollama` wires the
    real `OllamaLLM`, otherwise the in-process `FakeLLM(SimResponder)` runs. Reasoning: the
    app needed the real brain wired for Phase 3, but decision 4's seam must hold. The openai
    import stays lazy inside `OllamaLLM`, so importing the app never pulls it in and the
    sim/test path stays green with no Ollama, no openai, no GPU. Reversal: none; this is the
    same lazy-import discipline as the other real clients.

14. **The LoRA dataset is local, non-distributed, and built to teach voice not plot.**
    The training data is authored gold plus off-novel synthetic Q&A plus a neutral-replay
    slice, with the novel used only for short PARAPHRASED voice calibration (mine_book.py),
    never long verbatim spans. A held-out verbatim probe set (`holdout_probes.jsonl`) is
    never trained on and is used by eval.py to confirm the model does not regurgitate book
    text. Reasoning: fine-tuning on reproduction-shaped tasks is exactly what unlocks
    verbatim memorization; off-novel + paraphrase teaches mannerisms and is the copyright-
    safe design. All `finetune/data/*.jsonl` are gitignored build artifacts; only the
    builder scripts and schema are tracked. Reversal: none; this is a standing constraint.

15. **Serve the LoRA by merge-then-quantize, not an adapter on the quantized base.**
    train.py produces a LoRA; the runbook merges it into a full-precision Qwen2.5-7B and
    quantizes the single result to GGUF q5_K_M for `ollama create rocky`. Reasoning:
    llama.cpp will apply an f16 adapter over a quantized base, but Ollama's own docs warn a
    quantization mismatch between the adapter's base and the served base gives erratic
    results; merging sidesteps it. Reversal: adopt the ADAPTER directive path only if
    swappable per-character adapters are needed, and then pair it with a full-precision base.
