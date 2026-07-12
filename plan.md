# rocky_mini — Rocky-from-Project-Hail-Mary conversation app for Reachy Mini

## Context

TJ owns a **Reachy Mini Wireless** (Raspberry Pi CM4, 4GB) and wants a conversation app where the robot role-plays **Rocky, the Eridian engineer from *Project Hail Mary*** — a curious alien child on Earth whom TJ teaches about the world, and who remembers what it learns across sessions ("like having a curious alien child"). Decisions locked with TJ: from-scratch custom `ReachyMiniApp` (not a fork), **fully local, zero-API-cost stack** — Ollama-served **Qwen2.5-7B-Instruct** with a custom **Rocky LoRA** as the brain, **faster-whisper** STT, **Piper** TTS, all hosted on TJ's PC (**RTX 4080, 16GB VRAM**) which acts as the LAN brain server; the robot is a thin client at hardware bring-up. Chords+voice blend for Rocky's voice, repo name `rocky_mini`, sim-first development on Windows. **No paid API keys anywhere.**

This plan was produced from: a full read of the cloned `pollen-robotics/reachy_mini` SDK (v1.9.0), the official HF docs, a verified Rocky character dossier, a study of Pollen's official conversation app, and a 3-design panel judged by 2 adversarial reviewers (unanimous winner: the engineering-first design, with character/learning grafts below). Revised 2026-07-11 to replace the paid Claude + ElevenLabs stack with the local stack above and add the fine-tuning track.

## What gets created

1. `C:\Users\TJ\Documents\GitHub\reachy_mini` — reference clone of `pollen-robotics/reachy_mini` (read-only; docs, examples, AGENTS.md, `skills/`).
2. `C:\Users\TJ\Documents\GitHub\rocky_mini` — the new app repo, scaffolded ONLY via `reachy-mini-app-assistant create rocky_mini <path>` (AGENTS.md rule: never hand-create app folders), then git-committed locally. Pushing to a private GitHub remote (`gh repo create --private`) offered to TJ at the end, not done unprompted.
3. **Brain server on the PC** (lives inside the repo under `server/`): Ollama serves the LLM directly (OpenAI-compatible API at `http://<pc>:11434/v1`); a small FastAPI service exposes `/stt` (faster-whisper) and `/tts` (Piper). In sim everything is localhost; at M8 the robot reaches these over LAN.

Environment setup (Windows 11): `uv venv` (Python ≥3.11) → `uv pip install "reachy-mini[mujoco]"` → `reachy-mini-daemon --sim` (MuJoCo, zero hardware) → scaffold → `pip install -e .` makes the app dashboard-visible via the `reachy_mini_apps` entry point (no publication needed for dev). Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`. Brain setup: install Ollama for Windows → `ollama pull qwen2.5:7b-instruct-q5_K_M` (~5.5 GB, fits comfortably in 16GB VRAM with headroom for whisper). Training-only extra: WSL2 + CUDA + Unsloth env (never needed to *run* the app). **No API keys**; `~/.rocky_mini/.env` holds only the brain-server URL and an optional LAN auth token.

## Architecture (winning design + mandatory grafts)

One process, launched by the daemon (`python -u -m rocky_mini.main`, SIGINT to stop). **4 threads + 1 asyncio loop**, each owning one latency domain, communicating via queues + a locked `MotionState` + a global `generation` counter (barge-in flush):

1. **Main** — `RockyMiniApp.run()`: config, thread supervision, FastAPI settings UI (`custom_app_url="http://0.0.0.0:8042"`, serves `static/`), ordered shutdown (audio → asyncio → mixer drain → 0.6 s ramp to neutral → fsync memory).
2. **MotionThread (100 Hz)** — owns ALL `set_target()` calls, one per tick. Never calls `goto_target`/`play_move`/`cancel_move`/`play_sound` → the SDK's media-kill and baked-audio footguns are structurally unreachable. Primary layer (emote trajectories sampled from `RecordedMoves` data, internal minjerk gotos, breathing idle) + additive secondary offsets (speech wobble, DoA orient, listening freeze), world-frame, 0.4 s blends, clamped (pitch/roll ±40°, |head−body yaw| ≤65°, antennas rest ~10° never 0°).
3. **AudioInThread** — `mini.media.get_audio_sample()` 16 kHz → mono → 30 s ring buffer → Silero VAD (onnxruntime, no torch); publishes speech events; polls `get_DoA()` ~10 Hz on hardware.
4. **AudioOutThread (Mixer, 50 Hz / 320-sample frames)** — owns ALL `push_audio_sample()` calls. VoiceBus (generation-tagged TTS PCM) + ChordBus (Eridian stingers/underlay) → ring-mod DSP → sum → tanh clip → push. Maintains playout clock for wobble sync (+0.2 s).
5. **ConversationLoop (asyncio)** — state machine IDLE→LISTENING→TRANSCRIBING→THINKING→SPEAKING(→SLEEPWATCH): STT, streaming chat completions via the `openai` client pointed at Ollama (`base_url=http://<pc>:11434/v1`), sentence chunker, 2-deep pipelined TTS, background tools. Barge-in = `generation++` + task cancellation.

**Pipeline:** mic → VAD close (600 ms hangover) → *ack chord + thinking pose ≤150 ms (perceived response)* → faster-whisper STT on the PC GPU (~0.3–0.5 s) → **`rocky:latest`** (Qwen2.5-7B + Rocky LoRA, Ollama, streamed; `keep_alive=-1` so the model stays resident; byte-stable persona prefix so Ollama's KV-cache prefix reuse kicks in) → chunker (sentence splits, `[emote:]`/`[sfx:]` tag extraction, deterministic `', question?'` tic insertion) → Piper TTS (`en_US-lessac-medium`, 22.05 kHz — its slightly robotic timbre suits Rocky) → `resample_poly` → ring-mod (carrier ~140 Hz, phase carried across chunks) → Mixer → speaker; RMS envelope (delayed to playout +0.2 s) → head wobble.

**Tool calling on a 7B:** the 4 tools go through Ollama's OpenAI-compatible `tools` param (Qwen2.5 is the strongest tool-caller in class). Mitigations for 7B flakiness: pydantic validation of every tool-call payload, one silent retry with `format: json` on malformed calls, and tool-call exemplars baked into the fine-tune dataset.

**Honest latency budget** (improves on the cloud plan — first token on a 4080 is ~0.2–0.5 s): measured **from the user's last voiced frame including the 0.6 s VAD hangover** — time-to-first-voiced-audio ≈ **1.5–2.2 s p50** (0.6 s hangover + ~0.4 s STT + ~0.3 s first token + chunking + ~0.2 s Piper + DSP). The ≤150 ms ack-chord + thinking pose remains the perceived-response mask. `TurnMetrics` timestamps every stage; integration test asserts p50 ≤ 2.5 s and that steady-state turns show a small Ollama `prompt_eval_count` (proof the persona prefix is KV-cache-reused, not re-ingested).

**AudioIO seam:** `Protocol` with `ReachyMediaIO` (mini.media) and `SoundDeviceIO` (`media_backend="no_media"` + sounddevice) — first-class tested path, insulates Windows dev from GStreamer fragility. Sim defaults to push-to-talk/UI chat (no AEC on PC); open-mic barge-in ships **half-duplex by default** with a settings switch, tuned at hardware bring-up.

## Character system (grafts from the character design — mandatory)

- **Blind gaze:** NO face tracking anywhere (Rocky has no eyes — echolocates). Orient by DoA only: head leads 150 ms, body follows minjerk; ±5° jitter, slight overshoot, resting aim a few degrees below face height; idle "sonar sweep".
- **Persona prompt:** 5 byte-stable blocks (identity / speech rules / Eridian lore pack / behavior engine / output contract + guardrails). Byte-stability matters for Ollama KV-cache prefix reuse (and for reproducible fine-tune evals), so the prompt-builder is a pure function with a byte-stability test. Speech rules: `', question?'` particle forever; word tripling = intensity; dropped articles/copulas at ~70% (perfect consistency is less authentic); "Thank." / "Understand." / "Good. Good."; base-6 math, Eridian second = 2.366 s; "Rocky fix."; <60 words, one idea per turn; in-character deflection ("Rocky not help break things. Rocky FIX things.").
- **TicLinter** (deterministic post-pass before TTS, also on sim text replies and canned WAVs): question-particle insertion, assistant-ism strip, tripling/article-rate metrics; drift watchdog converts metric trends into a one-line mid-conversation `{"role":"system"}` nudge (single code path — local model, no vendor split).
- **Chord language:** base-6 just-intonation scale (ratios 1, 9/8, 5/4, 11/8, 3/2, 5/3); precomputed additive-synth stinger bank (~12: wake, listening, thinking, understand, error, interrupted, remember, jazz-hands, sleepy, signal-lost, still-watching, question); **question-interval stinger** (unresolved interval) auto-fired after every spoken `', question?'`; **tripled staccato hits** mirroring word tripling; 200–400 ms stinger-then-voice-fade-in opener (audiobook "chords first"); -12 dB underlay bed while speaking.
- **Jazz hands** = yes/excitement: procedural antenna shake ±20° @ 6 Hz, 1.2 s — auto-fired by the linter whenever a tripled word is detected (somatic reflex, not model-dependent).
- **SLEEPWATCH mode:** trigger on "I'm tired" (mother-hen: Eridians are paralyzed asleep, companion must watch) → `set_sleep_watch` tool → head lowered toward TJ's last DoA, breathing slowed to 0.05 Hz, drooped antennas, low two-note "still watching" chord every ~5 min, instant VAD wake.
- **One-time friendship ritual:** "Be careful. You are friend now." fired exactly once at the stage-2 growth transition, gated by a stored event flag (never repeats), unlocks affectionate teasing.

## Learning system (grafts from the learning design — mandatory)

- **Naivety enforcement (THE premise-critical problem — the model must not leak Earth knowledge Rocky wasn't taught):** (1) epistemic-ledger contract in the persona ("your ENTIRE Earth knowledge is the LEARNED FACTS list + TJ's own words; anything else → express curiosity") with paired WRONG/RIGHT few-shots; (2) the "Need word: …" ritual framed as the designated pressure valve when the model feels the pull to use untaught knowledge; (3) async **NaivetyAuditor** — an extra call to the same local model, free, so it runs on **every turn** (1-in-1), strictly off the speaking path — whose corrections inject as next-turn system messages; (4) the naivety contract is also *trained in* via the Rocky LoRA (see fine-tuning track). A **20-probe red-team suite** ("what's the capital of Italy?" etc.) runs as a **thresholded regression metric** — NOT a flaky zero-leak hard gate.
- **Memory:** JSONL store under `Path.home()/.rocky_mini/` (survives `pip --force-reinstall` app updates — nothing app-scoped does; brain-server URL in `~/.rocky_mini/.env` too). `facts.jsonl` (fsync per write, tombstone deletes), `open_questions.jsonl`, `sessions.jsonl`. **Fact confidence lifecycle:** heard_once → confirmed (restate-then-confirm rule) → mastered; at session start up to 2 stale heard_once facts render with a "(memory fuzzy)" marker so Rocky asks TJ to re-check (honest prompting, charming self-correction).
- **Digest injection:** built once at session start (prefix-stable), ≤2000 tokens; **mid-session facts** appended as `{"role":"system"}` messages (single path — local model handles system messages fine) so a fact taught 25 turns ago never silently vanishes. History trimmed in large chunks (not per-turn) to preserve the KV-cache prefix.
- **Tools (hard cap 4, sorted, pydantic-validated):** `remember_fact(category, fact, source_quote)`, `note_open_question(question)`, `confirm_fact(id)`, `set_sleep_watch(on)`. Called at END of reply (prompt rule + dataset exemplars); tool stall covered by the "remember" stinger and recorded in TurnMetrics.
- **Curiosity scheduler (deterministic Python, not prompt hope):** scored queue with decay, seeded with canon targets (sleep, faces/emotions, humor, hugs, sight, lying, human units); the model phrases the question but never picks it; hard cap 1 proactive question per 3 turns; >20 s idle silence triggers the top item.
- **ReflectionWorker:** one cheap local-model call post-session → 2-sentence summary + mined follow-ups → next session's greeting ritual ("TJ! You return. Yesterday you teach sandwich. Is taco also sandwich, question?").
- **Growth arc:** vocabulary count → stage 0/1/2 (telegraphic → "Need word:"-heavy → fluent-but-ticced); stage changes apply at session start only (byte-stable prompt-builder test must pass). Settings UI shows "Rocky knows N words · Stage X"; memory export/import (zip) migrates the sim-raised Rocky onto the physical robot.

## Fine-tuning track (Rocky LoRA — the zero-cost persona engine)

A 7B holds Rocky's voice far better when the tics, naivety contract, and tool habits are trained in rather than prompted in. Everything below runs on the 4080 for $0.

- **Dataset:** `finetune/data/rocky_dialogues.jsonl`, chat-format JSONL (system/user/assistant turns, tool calls included). Sources:
  - (a) **~150–300 seed conversations** authored inside Claude Code sessions (covered by TJ's subscription — no API spend): coverage matrix over the speech rules, naivety deflections (all 20 probe categories), "Need word:" ritual, restate-then-confirm memory turns, tool-call turns, curiosity questions, SLEEPWATCH trigger, growth stages 0/1/2.
  - (b) **Real session transcripts** accumulate post-M3; curated into the set using TicLinter metrics as the filter (only turns that scored in-character get in).
- **Training:** Unsloth QLoRA (r=16, 4-bit base, Qwen2.5-7B-Instruct) under **WSL2** (Unsloth is Linux-first; WSL2 + CUDA is the reliable Windows path). `finetune/train.py` + `finetune/README.md` document the exact env. ~20–40 min per run on the 4080. Training is an offline path — it never blocks or touches the running app.
- **Export:** merge LoRA → GGUF **Q5_K_M** → `finetune/Modelfile` → `ollama create rocky` → model tag `rocky:latest`.
- **Eval harness (`finetune/eval.py`):** fixed 40-prompt set scored on (1) the 20-probe naivety suite, threshold ≤2/20 leaks; (2) TicLinter metrics (question-particle rate, tripling frequency, article-drop rate ≈70%); (3) tool-call validity rate ≥90%. **A LoRA ships only if it beats the stock+prompt baseline on all three.** Config keeps a model toggle `rocky:latest` vs `qwen2.5:7b-instruct` (stock) for A/B-ing.
- **Cadence:** v1 trained at milestone M4.5; retrain whenever curated real transcripts grow by ~100 turns.

## Repo layout (rocky_mini)

```
rocky_mini/
├── pyproject.toml       # entry point reachy_mini_apps; deps: reachy-mini, openai (client only, pointed
│                        #   at Ollama), httpx, numpy, scipy, onnxruntime, silero-vad, pydantic-settings;
│                        #   dev: sounddevice, pytest(-asyncio). NO anthropic, NO elevenlabs.
├── .env.example         # brain-server URL + optional LAN token; real values in ~/.rocky_mini/.env
├── index.html/style.css # HF Space landing (assistant-generated)
├── rocky_mini/
│   ├── main.py          # RockyMiniApp(ReachyMiniApp): run(), supervision, FastAPI routes, ordered shutdown
│   ├── config.py        # pydantic-settings; model toggle rocky:latest|qwen2.5:7b-instruct (LoRA A/B)
│   ├── turn.py          # asyncio state machine, generation counter, TurnMetrics
│   ├── audio/           # io.py (AudioIO Protocol + 2 impls), input.py, vad.py, output.py (Mixer),
│   │                    #   dsp.py (ring_mod/soft_clip/resample, pure fns), chords.py (just-intonation bank)
│   ├── speech/          # stt.py, tts.py (Protocols + brain-server clients [faster-whisper/Piper]
│   │                    #   + Fakes + canned-WAV fallback)
│   ├── brain/           # llm.py (openai-client-on-Ollama + Fake), chunker.py (splits/tags/tic),
│   │                    #   persona.py (pure builder), tools.py (pydantic-validated), auditor.py
│   │                    #   (NaivetyAuditor), curiosity.py (scheduler), reflection.py
│   ├── memory/          # models.py, store.py (JSONL, digest, lifecycle, export/import)
│   ├── motion/          # manager.py (100 Hz), emotes.py (RecordedMoves loader + jazz_hands), idle.py
│   │                    #   (breathing/listening/thinking/sleepwatch), wobble.py
│   ├── static/          # settings UI: server URL, model toggle, fact table, sim chat, SSE metrics, emote buttons
│   └── assets/canned/   # ~12 in-character fallback WAVs ("Signal bad. Wait, wait, wait.")
├── server/              # PC brain server: FastAPI /stt (faster-whisper) + /tts (Piper); own deps
│                        #   (faster-whisper, piper-tts) kept OUT of the robot app's install; run notes
├── finetune/            # data/rocky_dialogues.jsonl, train.py (Unsloth QLoRA, WSL2), eval.py
│                        #   (naivety+tic+tool-call harness), Modelfile, README.md
└── tests/               # test_dsp (FFT sidebands, phase continuity), test_mixer (generation flush),
                         #   test_chunker (tic insertion, tags), test_persona (byte-stability, token budget),
                         #   test_memory, test_turn (Fakes), test_naivety (20-probe metric),
                         #   integration/test_sim_e2e.py (prefix-reuse + honest-latency asserts)
```

Reuse from the SDK (import, never copy): `ReachyMini`, `ReachyMiniApp`, `create_head_pose`, `RecordedMoves`, `mini.media.*`, `get_DoA`. Patterns copied from Pollen's conversation app (studied, not vendored): layered MovementManager, wobbler delay + generation counter, BackgroundToolManager.

## Milestones (each independently demoable; sim-first)

- **M0 Scaffold & heartbeat:** clone reference repo; scaffold `rocky_mini`; daemon `--sim` on Windows; breathing idle at 100 Hz; settings UI up; clean SIGINT ramp. *Also validates Windows GStreamer/local media or flips to SoundDeviceIO.* Install Ollama + pull `qwen2.5:7b-instruct-q5_K_M`.
- **M1 Motion core:** MovementManager + RecordedMoves emote sampling — **de-risks the one unverified SDK surface**: `RecordedMove.evaluate(t)` exists (`motion/recorded_move.py:101`) but RAISES at `t ≥ duration` → clamp before boundary; fallback = record own pose banks via `start_recording()/stop_recording()`. Jazz hands. UI emote buttons blend over breathing with no snap.
- **M2 Eridian audio:** Mixer + chord bank + ring-mod; FFT tests green; alien-voiced WAV + stingers audible from PC speakers.
- **M3 Brain loop (typed):** UI chat → **stock qwen2.5 via Ollama** stream → chunker → Piper TTS (brain server up) → mixer; persona v1; TurnMetrics panel. First in-character conversation.
- **M4 Naivety gate:** epistemic contract + auditor + 20-probe red-team suite as tracked metric (threshold, e.g. ≤2/20 leaks) — measured on the **stock+prompt baseline** (this number is what the LoRA must beat).
- **M4.5 Rocky LoRA v1:** author the seed dataset → Unsloth QLoRA in WSL2 → GGUF export → `ollama create rocky` → `finetune/eval.py` beats the M4 baseline on naivety, tic metrics, and tool-call validity → `rocky:latest` becomes the default model.
- **M5 Memory:** tools + store + digest + mid-session system-append + growth stages + fact review UI. Teach → restart → still knows it; prefix-reuse (`prompt_eval_count`) assert.
- **M6 Voice in:** VAD + faster-whisper STT; ack ≤150 ms; barge-in via generation flush (UI button in sim); honest latency asserts (p50 ≤ 2.5 s).
- **M7 Choreography & rituals:** wobble locked to playout clock; tags→motion/chords; question-stinger twin; SLEEPWATCH; curiosity scheduler over a scripted 10-turn FakeLLM session; greeting ritual.
- **M8 Hardware bring-up (Wireless):** publish private HF Space, dashboard install; brain-server URL via settings UI + **reachability check** (static IP or mDNS note for finding the PC); LAN latency re-measured end-to-end; DoA gaze + hardware VAD AND-gate; **verify `imu['temperature']` exists or drop thermal input** (duty governor stays); CM4 profile (<35% core, <300 MB RSS — easier now: the Pi only does VAD + motion + mixing, no local inference); motor-noise VAD threshold sweep; enable open-mic barge-in only after tuning.
- **M9 Soak & failure drills (ship gate):** 30-min soak; Wi-Fi pull mid-turn (canned "Signal bad" + Eridian-only mode: chords + UI subtitles); **"PC asleep / Ollama down" drill** (canned-WAV path + Eridian-only mode); brain-server auth-token revocation; SIGINT mid-TTS (facts intact); rapid barge-in stress.

## Key risks (accepted/mitigated)

- 7B persona drift / instruction slippage over long sessions — Rocky LoRA (trained-in tics + naivety) + deterministic TicLinter + every-turn NaivetyAuditor.
- 7B tool-calling flakiness — pydantic validation, one `format: json` retry, tool exemplars in the fine-tune dataset, tool-call validity gate in eval.
- Robot is dead without the PC awake on the LAN — accepted for a personal project; canned-WAV + Eridian-only mode is the offline fallback (M9 drill).
- WSL2/CUDA/Unsloth setup friction — training is an offline-only path; it can never block the running app; documented in `finetune/README.md`.
- Motor noise isn't AEC-cancelled — raised in-speech VAD threshold AND DoA flag; half-duplex default until M8 tuning.
- Memory poisoning — restate-before-store, confidence lifecycle, review UI (TJ is the human in the loop).
- Naivety can be reduced, not eliminated — LoRA + auditor + regression suite keep it honest.

## Skills & guides wired into execution

**SDK-bundled skill guides** (in the reference clone at `reachy_mini/skills/*.md` — read the listed guide BEFORE coding that milestone; AGENTS.md mandates this workflow):

| Milestone | SDK skill guides to read first |
|---|---|
| M0 | `setup-environment.md` (env + `agents.local.md` convention), `create-app.md` (scaffold rules), `testing-apps.md` (sim vs physical) |
| M1 | `motion-philosophy.md` (goto vs set_target), `control-loops.md` (100 Hz loop patterns), `symbolic-motion.md` (procedural breathing/jazz-hands/sonar-sweep math), `safe-torque.md` (no-jerk torque transitions) |
| M3–M5 | `ai-integration.md` (LLM-app patterns, tool dispatch) |
| M7 | `interaction-patterns.md` (antennas-as-buttons — antenna-touch as a tactile "wake Rocky" input), `symbolic-motion.md` again for SLEEPWATCH/rituals |
| M8 | `rest-api.md` (DoA/state endpoints, dashboard interplay), `testing-apps.md` (physical checklist) |
| Any blocker | `debugging.md`, `deep-dive-docs.md` |

AGENTS.md conventions adopted: keep a `plan.md` inside the app repo (this plan, adapted, lands there at M0) and an `agents.local.md` (robot type = Wireless, env notes, session context). Also write a repo `CLAUDE.md` at M0 encoding the non-negotiable footgun rules so future sessions never regress them: never call `play_move`/`cancel_move`/`play_sound`; one `set_target` owner; one `push_audio_sample` owner; byte-stable persona builder; memory lives in `~/.rocky_mini/`; antennas never commanded to 0°; honest latency measured from last voiced frame; no paid APIs — brain is Ollama/local only.

**Claude Code session skills** (invoked at execution time):

- `executing-plans` — start implementation from this plan with review checkpoints at each milestone.
- `karpathy-guidelines` — load before writing code (surgical changes, no overengineering — this app has many small modules that invite gold-plating).
- `run` — launch daemon `--sim` + brain server + app to observe real behavior at every milestone.
- `verify` + `verification-before-completion` — before declaring any milestone done: drive the actual flow (spoken/typed turn in sim, watch MuJoCo + listen to output), not just green tests; no success claims without command output.
- `playwright-skill` — automated browser tests of the settings UI (chat box round-trip, fact table CRUD, SSE metrics panel) at M3/M5/M7.
- `frontend-design` — review/polish pass on the settings UI (M5: fact review table; M7: metrics panel readability).
- `dataviz` — the TurnMetrics latency panel chart in the settings UI (M3).
- `code-review` — after M1 (motion core), M4.5 (fine-tune pipeline), M5 (memory), and pre-M8 (hardware) on the working diff; `/code-review ultra` optional before ship.
- `update-config` / `fewer-permission-prompts` — at M0, add project-level allowlist for `uv`, `reachy-mini-daemon`, `ollama`, `pytest` so the long build isn't interrupted by prompts.
- `research-lookup` — only if blocked on undocumented SDK/Ollama/Unsloth behavior after `debugging.md`.

## Verification

- **Unit:** DSP FFT sidebands/phase continuity, mixer generation-flush, chunker tic table, persona byte-stability + token budget, memory lifecycle, naivety 20-probe metric, tool-call pydantic validation + retry path.
- **Fine-tune gate:** `finetune/eval.py` — LoRA must beat the stock+prompt baseline on naivety (≤2/20), tic metrics, and tool-call validity (≥90%) before `rocky:latest` becomes default.
- **Integration (sim):** `tests/integration/test_sim_e2e.py` against `reachy-mini-daemon --sim` + local Ollama + brain server — typed turn produces pushed audio; steady-state `prompt_eval_count` small (KV-prefix reuse); honest p50 latency ≤ 2.5 s.
- **End-to-end (per milestone):** via the `run`/`verify` skills — launch daemon + brain server + app, drive the real flow (MuJoCo viewer for motion, ears for audio, settings UI for state), demo listed under each milestone; `playwright-skill` covers the UI surfaces.
- **Manual demos per milestone** as listed; M9 checklist is the ship gate.
- Run instructions land in the repo README (`uv` setup, Ollama pull, brain server, daemon --sim, dashboard, hardware install).
