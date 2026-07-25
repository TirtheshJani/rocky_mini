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

## 2026-07-23

18. **Footgun 9 amended: client-side vision is allowed; daemon head tracking is not.**
    Rocky gains eyes (face tracking plus a look-around ability), a deliberate move away from
    the book's eyeless echolocation to use the full Reachy Mini. The official Conversation App
    does face tracking daemon-side via `mini.start_head_tracking()`, which lets the daemon
    drive the head and would fight our single 100 Hz `set_target` owner (footgun 1). We do NOT
    use it. Instead a new VisionThread (5 Hz, Protocol plus Fake, sim-first) captures frames
    and detects a face, converts it to yaw/pitch, and feeds `MovementManager.set_gaze()`, a
    setter parallel to `set_doa`. The manager reconciles gaze (fresh face wins) against DoA and
    still emits exactly one `set_target` per tick. Persona keeps the alien register by framing
    the camera as an instrument Rocky built, not organic sight. Reasoning: TJ wants the robot's
    full utility with Rocky's mannerisms. Reversal: `ROCKY_VISION_ENABLED=0` (DoA-only) if the
    VisionThread load degrades the motion loop or audio on the CM4, measured at bring-up.

19. **Emotes are sampled from vendored Pollen recordings; sounds are Rocky's own voice.**
    Movie-flavored emotes (amaze, happy, sad, scared, laugh, surprised, grateful, yes, no,
    welcome, bored, sleep) reuse Pollen's `pollen-robotics/reachy-mini-emotions-library`
    RecordedMove JSON trajectories, vendored into `assets/emotes/` by a dev-PC build script so
    the robot needs no HuggingFace access. A `RecordedTrajectory` adapter SAMPLES them inside
    MotionThread through our existing `Trajectory` interface; we never call the SDK `play_move`
    (footgun 2). The recordings' own audio sidecars are discarded (generic robot voice); each
    emote is paired with a short canned WAV rendered in Rocky's Piper voice plus ring mod, and
    played through the Mixer VoiceBus with generation tagging (footgun 3), falling back to a
    chord stinger when no WAV exists. Reasoning: TJ wants film-accurate emote sound, not the
    library's generic voice. Reversal: none; this is the same sampling and single-owner audio
    discipline as the rest of the motion and mixer code.

20. **(pending probe P2) Local vision model for the look-around tool.** Default
    `ROCKY_VISION_MODEL=qwen2.5vl:7b` served by the existing PC Ollama; `moondream` is the
    lighter fallback if p50 latency on a 1280x720 JPEG exceeds roughly 8 s. To be confirmed by
    timing both on the PC (10.0.0.95) once hardware is back; the measured choice replaces this
    default here. Reversal: swap the env value; no code change needed.

21. **Visual object memory: frozen DINOv2 embeddings, few-shot cosine match, no training.**
    TJ shows Rocky a physical object and names it; a new `teach_object` tool embeds the
    current camera frame and stores it as one to three shots in an event-sourced
    `objects.jsonl` sidecar under `~/.rocky_mini` (footgun 5), keyed by name. Later,
    recognition rides the existing `_look_handler` (so it covers both the `look_around`
    tool and the spontaneous curious glance with no new threads and no VisionThread
    change; footgun 1 untouched): the frame is embedded and matched by max cosine over an
    object's shots, accepted only if score >= threshold AND (with two or more objects) it
    beats the runner-up by a margin. The embedder is NOT in Ollama, which has no
    image-embedding API (issues #4296, #5304); it is a `/embed_image` endpoint on the
    speech server (frozen DINOv2 ViT-S/14, ~85 MB, L2-normalized server-side), reached by
    the robot via `HttpImageEmbedder` behind the same Protocol+Fake seam as the observer.
    `teach_object` ALSO records a plain `remember_fact("object", ...)`, so the object
    enters the epistemic ledger even in sim, with the camera down, or with the switch off,
    where it degrades to the fact only and never raises. Threshold 0.60 / margin 0.08 are
    provisional, to be tuned from logged top-2 scores at hardware bring-up (probe-style,
    like decision 20). v1 embeds the whole frame; a qwen2.5vl bbox-crop is the noted
    upgrade if real-world matching is noisy. Kill switch: `ROCKY_OBJECT_MEMORY_ENABLED=0`
    makes the sidecar inert and recognition a no-op. Reversal: flip the switch; the jsonl
    sidecar sits idle, carried by export_zip but never read.

16. **(2026-07-22) Rocky's voice and mannerisms align to the film, not the book's raw telegraphic stage.**
    The Project Hail Mary film gives Rocky a deadpan, slightly degraded computer-voice-box
    delivery (James Ortiz; "a little Mr. Moviefone, a little Siri, only not as clean") over
    earnest, understandable sentences. Three changes: (a) the persona now frames the flat
    voice as Rocky's translator box, keeps the tics (', question?', triple repeat, 'Amaze.',
    'Thank.') and adds a worked introduction example so "introduce yourself" cannot collapse
    into catchphrases; (b) the Mixer ring mod gained a wet/dry `ring_mix` (default 0.2), so
    the voice is mostly natural with a metallic edge instead of unintelligible sidebands;
    (c) the speech server flattens Piper prosody (noise_scale 0.3, noise_w 0.4, length_scale
    1.05, env-overridable) for the deadpan read. The LoRA `rocky:latest` was trained on the
    old telegraphic persona and overfits catchphrases; until it is retrained on the new
    prompt, `ROCKY_MODEL=qwen2.5:7b-instruct` is the recommended robot setting. Reversal:
    if the softer ring mix loses the alien character on the robot speaker, raise
    ROCKY_RING_MIX toward 0.4; if eval.py shows weaker tic adherence, tighten speech rules.

17. **(2026-07-22) A brain-server failure degrades in character instead of going silent.**
    ConversationLoop now guards STT, LLM streaming, and per-chunk TTS; the first failure in
    a turn plays a canned in-character WAV (assets/canned) via the previously unwired
    CannedTTS, and the loop stays alive for the next utterance. Reasoning: on hardware the
    only failure surface was a logged exception; Rocky appearing to die whenever the PC
    slept was indistinguishable from a crash. Reversal: none; silence is never the right UX.

22. **(2026-07-23) Lively-conversation pass: LoRA v2 + persona reply-shape rule + wake greeting.**
    Live conversation felt clipped and robotic; root cause was the training mix itself:
    gold replies averaged 13 words and the persona ordered "one main idea per turn".
    Studied the official reachy_mini_conversation_app for reference patterns and adopted
    three: an explicit reply-shape rule in the prompt, robot-speaks-first greeting, and
    varied wording guidance. Changes: (a) persona speech rule is now react + one own
    thought + hand the turn back, two to four short sentences; (b) 18 new lively authored
    exemplars, livelier synth few-shots plus a generation-only STYLE_HINT, and 18 new
    conversational topics (opinions, stories, banter); (c) neutral replay grown to 60 rows
    to hold the 15-20 percent anti-forgetting band (369 total, 16 percent neutral);
    (d) ConversationLoop.handle_wake_greeting() speaks a varied in-character greeting at
    hardware startup (ROCKY_WAKE_GREETING=0 disables). LoRA v2 trains on the new persona
    bytes so training and runtime prefixes stay byte-identical. Also this pass: speech
    server pre-warms all models at boot (a cold faster-whisper load caused an
    httpx.ReadTimeout and a canned-fallback first turn), client speech timeouts 15->30 s,
    whisper small->medium (ROCKY_WHISPER_SIZE), and the Eridian voice-box DSP on /tts
    (pitch -2 semitones plus fifth/octave underlay, ROCKY_TTS_FX=0 reverts). Shipped 2026-07-23: eval gate PASS (0/20 naivety leaks, particle rate 1.0, tool
    validity 1.0, memorization pass, capability 1.0 vs baseline 0.875); rocky:latest
    re-tagged to the v2 build (hash d2aa4e6f8bd4, rocky-v2 kept as an alias). Reversal:
    if live conversation regresses, ollama create rocky -f finetune/Modelfile (v1 gguf); if TJ finds the chord timbre muddy on the
    robot speaker, lower ROCKY_TTS_FX_FIFTH_GAIN / _OCTAVE_GAIN.

23. **(2026-07-23) Rocky's voice is the Chatterbox movie-audio clone, "early" reference cut.**
    TJ rejected Piper (robotic), Piper+chord DSP (more robotic), and an XTTS v2 clone
    (dated, stilted). Chatterbox (Resemble AI, open) cloning a clean 12 s cut (5-17 s) of
    the scrubbed film audio from pedramamini's rocky_say gist won TJ's A/B across three
    reference cuts. Live settings: exaggeration 0.5, cfg_weight 0.6 (an auto-picked
    loudest-window reference plus cfg 0.35 drifted into a wrong accent; adherence and a
    hand-checked cut fixed it). Serving venv is C:\Users\TJ\rocky_tts_trial (chatterbox's
    transformers pin conflicts with coqui's); chatterbox loads FIRST in warm-up (loading
    it after ctranslate2 binds cudnn crashed the process), DINO moved to CPU and whisper
    back to small for VRAM headroom (peak ~10 GB of 12 GB with the 7B resident). Cost:
    ~5.5 s per spoken sentence on the shared GPU, so first audio trails the ack chord by
    ~5 s. Reversal: ROCKY_TTS_BACKEND=piper for instant-but-robotic; if latency grates,
    revisit streaming generation or a faster clone model, not the DSP path.
