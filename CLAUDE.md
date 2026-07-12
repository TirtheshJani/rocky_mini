# CLAUDE.md - rocky_mini non-negotiable rules

Rocky is the Eridian engineer from Project Hail Mary, role-played by a Reachy Mini
Wireless. This app is sim-first, fully local, and has zero paid-API cost. Future
sessions must not regress the footguns below. They are structural safety rules,
not style preferences.

## Footgun rules (never regress)

1. **One `set_target` owner.** Only `motion/manager.py` (MotionThread, 100 Hz) may
   call `mini.set_target()`, exactly one call per tick. No other module touches it.
2. **Never call `play_move`, `cancel_move`, `goto_target`, or `play_sound`.** These
   SDK helpers kill in-flight media and play baked audio. MotionThread samples
   trajectories itself and never calls them, so the footgun is unreachable.
3. **One `push_audio_sample` owner.** Only `audio/output.py` (the Mixer) may push
   audio. Everything else publishes PCM onto the VoiceBus / ChordBus.
4. **Byte-stable persona builder.** `brain/persona.py` is a pure function. Its output
   must be byte-identical for identical inputs (Ollama KV-cache prefix reuse and
   reproducible fine-tune evals depend on it). A byte-stability test guards this.
5. **Memory lives in `~/.rocky_mini/`,** never inside the repo/app dir. App dirs are
   wiped by `pip --force-reinstall`; the home dir survives updates.
6. **Antennas are never commanded to 0 degrees.** Rest is ~10 degrees. Zero looks dead.
7. **Honest latency is measured from the user's last voiced frame,** including the
   0.6 s VAD hangover. Do not quote latency from "after hangover".
8. **No paid APIs.** The brain is Ollama / local models only. No `anthropic`, no
   `elevenlabs`, no cloud keys. `~/.rocky_mini/.env` holds only a LAN URL + token.
9. **No face tracking, ever.** Rocky has no eyes (echolocates). Orient by DoA only.
10. **No em dashes** in code, comments, docstrings, or docs authored here.

## Architecture invariants

- 4 threads + 1 asyncio loop, each owning one latency domain (see plan.md Architecture).
- Barge-in = increment the global `generation` counter + cancel tasks; the Mixer drops
  any VoiceBus PCM tagged with a stale generation.
- External services (LLM, STT, TTS, robot SDK) sit behind Protocols with Fakes. The
  sim/test path never imports the real clients; imports are lazy and guarded.

## Where to look

- `plan.md` - the full design and milestones (source of truth for scope).
- `agents.local.md` - robot type, environment notes, session context.
- `decisions.md` - binding decisions with dates and reversal conditions.
