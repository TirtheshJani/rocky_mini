# rocky_mini — Rocky-from-Project-Hail-Mary conversation app for Reachy Mini

## Context

TJ owns a **Reachy Mini Wireless** (Raspberry Pi CM4, 4GB) and wants a conversation app where the robot role-plays **Rocky, the Eridian engineer from *Project Hail Mary*** — a curious alien child on Earth whom TJ teaches about the world, and who remembers what it learns across sessions ("like having a curious alien child"). Decisions locked with TJ: from-scratch custom `ReachyMiniApp` (not a fork), Claude brain, chords+voice blend for Rocky's voice, repo name `rocky_mini`, sim-first development on Windows.

This plan was produced from: a full read of the cloned `pollen-robotics/reachy_mini` SDK (v1.9.0), the official HF docs, a verified Rocky character dossier, a study of Pollen's official conversation app, and a 3-design panel judged by 2 adversarial reviewers (unanimous winner: the engineering-first design, with character/learning grafts below).

## What gets created

1. `C:\Users\TJ\Documents\GitHub\reachy_mini` — reference clone of `pollen-robotics/reachy_mini` (read-only; docs, examples, AGENTS.md, `skills/`).
2. `C:\Users\TJ\Documents\GitHub\rocky_mini` — the new app repo, scaffolded ONLY via `reachy-mini-app-assistant create rocky_mini <path>` (AGENTS.md rule: never hand-create app folders), then git-committed locally. Pushing to a private GitHub remote (`gh repo create --private`) offered to TJ at the end, not done unprompted.

Environment setup (Windows 11): `uv venv` (Python ≥3.11) → `uv pip install "reachy-mini[mujoco]"` → `reachy-mini-daemon --sim` (MuJoCo, zero hardware) → scaffold → `pip install -e .` makes the app dashboard-visible via the `reachy_mini_apps` entry point (no publication needed for dev). Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`. Keys needed: `ANTHROPIC_API_KEY` + `ELEVENLABS_API_KEY` (ElevenLabs is the STT+TTS vendor; TJ needs an account — flag at M3).

## Architecture (winning design + mandatory grafts)

One process, launched by the daemon (`python -u -m rocky_mini.main`, SIGINT to stop). **4 threads + 1 asyncio loop**, each owning one latency domain, communicating via queues + a locked `MotionState` + a global `generation` counter (barge-in flush):

1. **Main** — `RockyMiniApp.run()`: config, thread supervision, FastAPI settings UI (`custom_app_url="http://0.0.0.0:8042"`, serves `static/`), ordered shutdown (audio → asyncio → mixer drain → 0.6 s ramp to neutral → fsync memory).
2. **MotionThread (100 Hz)** — owns ALL `set_target()` calls, one per tick. Never calls `goto_target`/`play_move`/`cancel_move`/`play_sound` → the SDK's media-kill and baked-audio footguns are structurally unreachable. Primary layer (emote trajectories sampled from `RecordedMoves` data, internal minjerk gotos, breathing idle) + additive secondary offsets (speech wobble, DoA orient, listening freeze), world-frame, 0.4 s blends, clamped (pitch/roll ±40°, |head−body yaw| ≤65°, antennas rest ~10° never 0°).
3. **AudioInThread** — `mini.media.get_audio_sample()` 16 kHz → mono → 30 s ring buffer → Silero VAD (onnxruntime, no torch); publishes speech events; polls `get_DoA()` ~10 Hz on hardware.
4. **AudioOutThread (Mixer, 50 Hz / 320-sample frames)** — owns ALL `push_audio_sample()` calls. VoiceBus (generation-tagged TTS PCM) + ChordBus (Eridian stingers/underlay) → ring-mod DSP → sum → tanh clip → push. Maintains playout clock for wobble sync (+0.2 s).
5. **ConversationLoop (asyncio)** — state machine IDLE→LISTENING→TRANSCRIBING→THINKING→SPEAKING(→SLEEPWATCH): STT, `AsyncAnthropic().messages.stream()`, sentence chunker, 2-deep pipelined TTS, background tools. Barge-in = `generation++` + task cancellation.

**Pipeline:** mic → VAD close (600 ms hangover) → *ack chord + thinking pose ≤150 ms (perceived response)* → ElevenLabs scribe_v1 STT (~0.54 s) → Claude **claude-opus-4-8** stream (thinking omitted; `output_config={"effort":"low"}` — legal on opus-4-8, ERRORS on haiku; cached system ≥4500 tokens) → chunker (sentence splits, `[emote:]`/`[sfx:]` tag extraction, deterministic `', question?'` tic insertion) → ElevenLabs flash_v2_5 TTS `pcm_24000` → `resample_poly` → ring-mod (carrier ~140 Hz, phase carried across chunks) → Mixer → speaker; RMS envelope (delayed to playout +0.2 s) → head wobble.

**Honest latency budget** (judge-corrected): measured **from the user's last voiced frame including the 0.6 s VAD hangover** — time-to-first-voiced-audio ≈ **2.7–3.3 s p50 on opus-4-8** (~2.3–2.7 s on haiku option); the ≤150 ms ack-chord + thinking pose is the load-bearing perceived-response mask. `TurnMetrics` timestamps every stage; integration test asserts p50 ≤ 3.3 s and `usage.cache_read_input_tokens > 0` per turn.

**AudioIO seam:** `Protocol` with `ReachyMediaIO` (mini.media) and `SoundDeviceIO` (`media_backend="no_media"` + sounddevice) — first-class tested path, insulates Windows dev from GStreamer fragility. Sim defaults to push-to-talk/UI chat (no AEC on PC); open-mic barge-in ships **half-duplex by default** with a settings switch, tuned at hardware bring-up.

## Character system (grafts from the character design — mandatory)

- **Blind gaze:** NO face tracking anywhere (Rocky has no eyes — echolocates). Orient by DoA only: head leads 150 ms, body follows minjerk; ±5° jitter, slight overshoot, resting aim a few degrees below face height; idle "sonar sweep".
- **Persona prompt:** 5 byte-stable blocks (identity / speech rules / Eridian lore pack / behavior engine / output contract + guardrails), ≥4500 tokens (clears the 4096 cache minimum), `cache_control: ephemeral` breakpoint. Speech rules: `', question?'` particle forever; word tripling = intensity; dropped articles/copulas at ~70% (perfect consistency is less authentic); "Thank." / "Understand." / "Good. Good."; base-6 math, Eridian second = 2.366 s; "Rocky fix."; <60 words, one idea per turn; in-character deflection ("Rocky not help break things. Rocky FIX things.").
- **TicLinter** (deterministic post-pass before TTS, also on sim text replies and canned WAVs): question-particle insertion, assistant-ism strip, tripling/article-rate metrics; drift watchdog converts metric trends into a one-line mid-conversation `{"role":"system"}` nudge (opus-4-8-only; haiku path uses a `<system-reminder>` block in the next user turn — both paths get tests).
- **Chord language:** base-6 just-intonation scale (ratios 1, 9/8, 5/4, 11/8, 3/2, 5/3); precomputed additive-synth stinger bank (~12: wake, listening, thinking, understand, error, interrupted, remember, jazz-hands, sleepy, signal-lost, still-watching, question); **question-interval stinger** (unresolved interval) auto-fired after every spoken `', question?'`; **tripled staccato hits** mirroring word tripling; 200–400 ms stinger-then-voice-fade-in opener (audiobook "chords first"); -12 dB underlay bed while speaking.
- **Jazz hands** = yes/excitement: procedural antenna shake ±20° @ 6 Hz, 1.2 s — auto-fired by the linter whenever a tripled word is detected (somatic reflex, not model-dependent).
- **SLEEPWATCH mode:** trigger on "I'm tired" (mother-hen: Eridians are paralyzed asleep, companion must watch) → `set_sleep_watch` tool → head lowered toward TJ's last DoA, breathing slowed to 0.05 Hz, drooped antennas, low two-note "still watching" chord every ~5 min, instant VAD wake.
- **One-time friendship ritual:** "Be careful. You are friend now." fired exactly once at the stage-2 growth transition, gated by a stored event flag (never repeats), unlocks affectionate teasing.

## Learning system (grafts from the learning design — mandatory)

- **Naivety enforcement (THE premise-critical problem — Claude must not leak Earth knowledge Rocky wasn't taught):** (1) epistemic-ledger contract in the cached persona ("your ENTIRE Earth knowledge is the LEARNED FACTS list + TJ's own words; anything else → express curiosity") with paired WRONG/RIGHT few-shots; (2) the "Need word: …" ritual framed as the designated pressure valve when the model feels the pull to use untaught knowledge; (3) sampled async **claude-haiku-4-5 NaivetyAuditor** (1-in-1 dev, 1-in-4 prod, strictly off the speaking path) whose corrections inject as next-turn system messages. A **20-probe red-team suite** ("what's the capital of Italy?" etc.) runs as a **thresholded regression metric** — NOT a flaky zero-leak hard gate.
- **Memory:** JSONL store under `Path.home()/.rocky_mini/` (survives `pip --force-reinstall` app updates — nothing app-scoped does; keys in `~/.rocky_mini/.env` too). `facts.jsonl` (fsync per write, tombstone deletes), `open_questions.jsonl`, `sessions.jsonl`. **Fact confidence lifecycle:** heard_once → confirmed (restate-then-confirm rule) → mastered; at session start up to 2 stale heard_once facts render with a "(memory fuzzy)" marker so Rocky asks TJ to re-check (honest prompting, charming self-correction).
- **Digest injection:** built once at session start (cache-stable), ≤2000 tokens; **mid-session facts** additionally appended as `{"role":"system"}` messages on opus-4-8 (verified cache-safe; haiku fallback = user-turn note) so a fact taught 25 turns ago never silently vanishes. History trimmed in large chunks (not per-turn) to preserve the cache prefix.
- **Tools (hard cap 4, sorted, `strict:true`):** `remember_fact(category, fact, source_quote)`, `note_open_question(question)`, `confirm_fact(id)`, `set_sleep_watch(on)`. Called at END of reply (prompt rule); ~1 s tool stall covered by the "remember" stinger and recorded in TurnMetrics.
- **Curiosity scheduler (deterministic Python, not prompt hope):** scored queue with decay, seeded with canon targets (sleep, faces/emotions, humor, hugs, sight, lying, human units); Claude phrases the question but never picks it; hard cap 1 proactive question per 3 turns; >20 s idle silence triggers the top item.
- **ReflectionWorker:** one cheap haiku call post-session → 2-sentence summary + mined follow-ups → next session's greeting ritual ("TJ! You return. Yesterday you teach sandwich. Is taco also sandwich, question?").
- **Growth arc:** vocabulary count → stage 0/1/2 (telegraphic → "Need word:"-heavy → fluent-but-ticced); stage changes apply at session start only (byte-stable prompt-builder test must pass). Settings UI shows "Rocky knows N words · Stage X"; memory export/import (zip) migrates the sim-raised Rocky onto the physical robot.

## Repo layout (rocky_mini)

```
rocky_mini/
├── pyproject.toml       # entry point reachy_mini_apps; deps: reachy-mini, anthropic, elevenlabs,
│                        #   numpy, scipy, onnxruntime, silero-vad, pydantic-settings; dev: sounddevice, pytest(-asyncio)
├── .env.example         # real keys live in ~/.rocky_mini/.env
├── index.html/style.css # HF Space landing (assistant-generated)
├── rocky_mini/
│   ├── main.py          # RockyMiniApp(ReachyMiniApp): run(), supervision, FastAPI routes, ordered shutdown
│   ├── config.py        # pydantic-settings; model toggle opus-4-8|haiku-4-5 with tradeoff copy
│   ├── turn.py          # asyncio state machine, generation counter, TurnMetrics
│   ├── audio/           # io.py (AudioIO Protocol + 2 impls), input.py, vad.py, output.py (Mixer),
│   │                    #   dsp.py (ring_mod/soft_clip/resample, pure fns), chords.py (just-intonation bank)
│   ├── speech/          # stt.py, tts.py (Protocols + ElevenLabs impls + Fakes + canned-WAV + Piper offline fallback)
│   ├── brain/           # claude.py, chunker.py (splits/tags/tic), persona.py (pure builder), tools.py,
│   │                    #   auditor.py (NaivetyAuditor), curiosity.py (scheduler), reflection.py
│   ├── memory/          # models.py, store.py (JSONL, digest, lifecycle, export/import)
│   ├── motion/          # manager.py (100 Hz), emotes.py (RecordedMoves loader + jazz_hands), idle.py
│   │                    #   (breathing/listening/thinking/sleepwatch), wobble.py
│   ├── static/          # settings UI: keys, model toggle, fact table, sim chat, SSE metrics, emote buttons
│   └── assets/canned/   # ~12 in-character fallback WAVs ("Signal bad. Wait, wait, wait.")
└── tests/               # test_dsp (FFT sidebands, phase continuity), test_mixer (generation flush),
                         #   test_chunker (tic insertion, tags), test_persona (byte-stability, token budget),
                         #   test_memory, test_turn (Fakes), test_naivety (20-probe metric),
                         #   integration/test_sim_e2e.py (cache-hit + honest-latency asserts)
```

Reuse from the SDK (import, never copy): `ReachyMini`, `ReachyMiniApp`, `create_head_pose`, `RecordedMoves`, `mini.media.*`, `get_DoA`. Patterns copied from Pollen's conversation app (studied, not vendored): layered MovementManager, wobbler delay + generation counter, BackgroundToolManager.

## Milestones (each independently demoable; sim-first)

- **M0 Scaffold & heartbeat:** clone reference repo; scaffold `rocky_mini`; daemon `--sim` on Windows; breathing idle at 100 Hz; settings UI up; clean SIGINT ramp. *Also validates Windows GStreamer/local media or flips to SoundDeviceIO.*
- **M1 Motion core:** MovementManager + RecordedMoves emote sampling — **de-risks the one unverified SDK surface**: `RecordedMove.evaluate(t)` exists (`motion/recorded_move.py:101`) but RAISES at `t ≥ duration` → clamp before boundary; fallback = record own pose banks via `start_recording()/stop_recording()`. Jazz hands. UI emote buttons blend over breathing with no snap.
- **M2 Eridian audio:** Mixer + chord bank + ring-mod; FFT tests green; alien-voiced WAV + stingers audible from PC speakers.
- **M3 Brain loop (typed):** UI chat → opus-4-8 stream → chunker → TTS → mixer; persona v1; TurnMetrics panel. First in-character conversation.
- **M4 Naivety gate:** epistemic contract + auditor + 20-probe red-team suite as tracked metric (threshold, e.g. ≤2/20 leaks, before voice work — not zero-leak).
- **M5 Memory:** tools + store + digest + mid-session system-append (both model paths tested) + growth stages + fact review UI. Teach → restart → still knows it; cache-hit assert.
- **M6 Voice in:** VAD + STT; ack ≤150 ms; barge-in via generation flush (UI button in sim); honest latency asserts.
- **M7 Choreography & rituals:** wobble locked to playout clock; tags→motion/chords; question-stinger twin; SLEEPWATCH; curiosity scheduler over a scripted 10-turn FakeClaude session; greeting ritual.
- **M8 Hardware bring-up (Wireless):** publish private HF Space, dashboard install; keys via settings UI; DoA gaze + hardware VAD AND-gate; **verify `imu['temperature']` exists or drop thermal input** (duty governor stays); CM4 profile (<35% core, <300 MB RSS); motor-noise VAD threshold sweep; enable open-mic barge-in only after tuning.
- **M9 Soak & failure drills (ship gate):** 30-min soak; Wi-Fi pull mid-turn (canned "Signal bad" + Eridian-only mode: chords + UI subtitles); "ElevenLabs down, Anthropic up" drill (Piper/canned path); key revocation; SIGINT mid-TTS (facts intact); rapid barge-in stress.

## Key risks (accepted/mitigated)

- Opus-4-8 first-token variance can push first-audio past 3 s — ack-chord mask + documented haiku toggle.
- Motor noise isn't AEC-cancelled — raised in-speech VAD threshold AND DoA flag; half-duplex default until M8 tuning.
- ElevenLabs single-vendor for STT+TTS — Protocol seams + Piper/canned fallbacks + M9 drill.
- Memory poisoning — restate-before-store, confidence lifecycle, review UI (TJ is the human in the loop).
- Naivety can be reduced, not eliminated — auditor + regression suite keep it honest.

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

AGENTS.md conventions adopted: keep a `plan.md` inside the app repo (this plan, adapted, lands there at M0) and an `agents.local.md` (robot type = Wireless, env notes, session context). Also write a repo `CLAUDE.md` at M0 encoding the non-negotiable footgun rules so future sessions never regress them: never call `play_move`/`cancel_move`/`play_sound`; one `set_target` owner; one `push_audio_sample` owner; byte-stable persona builder; memory lives in `~/.rocky_mini/`; antennas never commanded to 0°; honest latency measured from last voiced frame.

**Claude Code session skills** (invoked at execution time):

- `executing-plans` — start implementation from this plan with review checkpoints at each milestone.
- `karpathy-guidelines` — load before writing code (surgical changes, no overengineering — this app has many small modules that invite gold-plating).
- `claude-api` — reload when implementing `brain/claude.py`, the NaivetyAuditor, and both model paths (streaming, `cache_control`, tool `strict:true`, opus-vs-haiku parameter differences are all specified there).
- `run` — launch daemon `--sim` + app to observe real behavior at every milestone.
- `verify` + `verification-before-completion` — before declaring any milestone done: drive the actual flow (spoken/typed turn in sim, watch MuJoCo + listen to output), not just green tests; no success claims without command output.
- `playwright-skill` — automated browser tests of the settings UI (chat box round-trip, fact table CRUD, SSE metrics panel) at M3/M5/M7.
- `frontend-design` — review/polish pass on the settings UI (M5: fact review table; M7: metrics panel readability).
- `dataviz` — the TurnMetrics latency panel chart in the settings UI (M3).
- `code-review` — after M1 (motion core), M5 (memory), and pre-M8 (hardware) on the working diff; `/code-review ultra` optional before ship.
- `update-config` / `fewer-permission-prompts` — at M0, add project-level allowlist for `uv`, `reachy-mini-daemon`, `pytest` so the long build isn't interrupted by prompts.
- `research-lookup` — only if blocked on undocumented SDK/API behavior after `debugging.md`.

## Verification

- **Unit:** DSP FFT sidebands/phase continuity, mixer generation-flush, chunker tic table, persona byte-stability + token budget, memory lifecycle, naivety 20-probe metric.
- **Integration (sim):** `tests/integration/test_sim_e2e.py` against `reachy-mini-daemon --sim` — typed turn produces pushed audio; `cache_read_input_tokens > 0`; honest p50 latency ≤ 3.3 s.
- **End-to-end (per milestone):** via the `run`/`verify` skills — launch daemon + app, drive the real flow (MuJoCo viewer for motion, ears for audio, settings UI for state), demo listed under each milestone; `playwright-skill` covers the UI surfaces.
- **Manual demos per milestone** as listed; M9 checklist is the ship gate.
- Run instructions land in the repo README (`uv` setup, daemon --sim, dashboard, hardware install).
