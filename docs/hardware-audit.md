# Hardware audit: rocky_mini vs reachy-mini SDK 1.9.0

Date: 2026-07-15. SDK pinned: **reachy-mini 1.9.0** (PyPI wheel, Python 3.11 venv).
Every claim below was checked against the installed source in `site-packages/reachy_mini/`,
and the load-bearing ones were exercised live against a running daemon
(`reachy-mini-daemon --mockup-sim --headless --no-media --autostart`). Findings marked
"cannot verify" need the physical robot. Everything is version-specific: re-audit on SDK upgrade.

## Read this first: what needs your decision vs what is just broken

**Decisions for TJ** (each has a section below with the tradeoff in Rocky's behaviour):

1. **D1, barge-in and AEC (decisions.md #6).** The Wireless has hardware echo cancellation
   in the XMOS chip, and the SDK also inserts software AEC on a PC. The "no AEC" reasoning
   behind half-duplex is wrong on the robot. Proposal: keep half-duplex as the default,
   add the open-mic switch and a tuned ReSpeaker config, flip it only after a bring-up test.
2. **D2, how Rocky gets installed on the robot.** The dashboard can only install apps from
   Hugging Face Spaces or a local folder in SDK 1.9.0. There is no "install from GitHub URL"
   path. Options: publish a private HF Space (assistant does this in one command), or SSH in
   once and pip install from the repo. This changes the tutorial's main path.
3. **D3, dependencies on the robot.** The dashboard installs the app with plain
   `pip install <target>`, no extras. Rocky's LLM client and VAD live in optional extras
   (`[llm]`, `[vad]`), so on the robot they would silently not be installed and Rocky would
   have no brain client and no ears. Proposal: move runtime deps needed on hardware into core
   dependencies (they are small; onnxruntime is the only heavy one).
4. **D4, settings UI serving.** The SDK base class wants to serve the settings UI itself
   (from `custom_app_url` + `static/`). Rocky currently starts its own uvicorn inside `run()`,
   which fights the SDK and blocks shutdown. Proposal: let the base class own the webserver
   and attach Rocky's API routes to `self.settings_app`. UI behaviour is unchanged.

**Bugs I will just fix** (tests first, no decision needed): F1, F2, F3, F4, F5, F6, F7,
F8 below. **One scope gap you should know about** before approving: G1, the voice input
thread was never built at all, on any path. It is in plan.md but not in the code.

## Verdict table: every SDK surface rocky_mini touches

| # | Surface | rocky_mini assumes | SDK 1.9.0 actually | Verdict |
|---|---|---|---|---|
| F1 | `ReachyMiniApp.run()` | `run(self, mini=None)` | Abstract `run(self, reachy_mini, stop_event)`; `wrapped_run()` passes both | **Broken** (proven live) |
| F2 | App launch path | Daemon calls the entry point class | Daemon spawns `python -u -m rocky_mini.main` as a subprocess and stops it with SIGINT (20 s, then SIGKILL). The module `__main__` must call `wrapped_run()` | **Broken** |
| F3 | Settings web UI | `uvicorn.run()` inside `run()`, own FastAPI | Base class serves `self.settings_app` in a thread when the class attribute `custom_app_url` is set, mounts `static/`, serves `index.html`, adds no-cache. Dashboard finds the UI link by regex-scanning `main.py` for `custom_app_url = "..."` | **Broken** |
| F4 | `set_target` | `mini.set_target(pose)` with rocky's `Pose` dataclass, degrees | `set_target(head=4x4 float64 matrix, antennas=[right, left] radians, body_yaw=radians)` | **Broken** (proven live: `AttributeError: 'Pose' object has no attribute 'shape'`) |
| F5 | Output audio | Mixer pushes 22050 Hz mono, no resample | Output is 16000 Hz; mono is auto-expanded to stereo by `MediaManager`, but nothing resamples. `push_audio_sample` also needs `media.start_playing()` first or frames are dropped with a log warning | **Broken** |
| F6 | Input audio | `media.get_audio_sample()` returns array; `.mean(axis=1)` if 2-D | Returns `None` unless `media.start_recording()` was called and data is ready; when present it is `(N, 2)` float32 at 16 kHz. `np.asarray(None)` crashes | **Broken** |
| F7 | DoA | `getattr(mini, "get_DoA", None)`, degrees, float | Lives on `mini.media`, not `mini` (proven live). Returns `(angle_radians, speech_detected)` or `None`. Convention per SDK docstring: 0 rad is left, pi/2 is front/back, pi is right. Requires the ReSpeaker over USB, firmware >= 2.1.0 | **Broken, and silently** |
| F8 | Barge-in flush | Generation-tagged drop inside the Mixer only | Audio already pushed into GStreamer keeps playing. SDK 1.9.0 has `clear_player()` exactly for barge-in; the official Conversation App calls it | **Gap** |
| L8 | 100 Hz `set_target` owner design | One owner thread, one call per tick, bypass interpolation | Confirmed correct. `set_target` is the documented high-frequency call; the official Conversation App uses the identical pattern (single worker thread, `set_target` once per tick, errors rate-limited) | **Holds** |
| L3 | `request_media_backend` | Never set | Class attribute; `"default"` auto-resolves to LOCAL when the app runs on the same machine as the daemon, which is right for on-device. Valid values now: `no_media`, `local`, `webrtc`, `default` (`gstreamer*` names are deprecated aliases) | **Holds by luck**, set it explicitly |
| M1 | Antenna rest | Never command 0 deg, rest ~10 deg | SDK's own `INIT_ANTENNAS_JOINT_POSITIONS` is +/-0.1745 rad, ~10 deg, "to reduce shaking". Footgun 6 agrees with the vendor | **Holds** |
| M2 | Forbidden helpers (footgun 2) | Never call `play_move`, `goto_target`, `play_sound`, `cancel_move` | All exist and are still dangerous. New in 1.9.0: `mini.wake_up()` and `mini.goto_sleep()` internally call `goto_target` AND `media.play_sound()`. Also `enable_wobbling()` adds a daemon-side audio-reactive head wobble that would fight Rocky's own wobble. None of these may be called | **Holds**, two new names added to the do-not-call list |
| M3 | `imu['temperature']` (plan.md M8) | Verify or drop | `mini.imu` exists and documents `'temperature'` in degrees C (Wireless only; None otherwise) | **Holds**, keep thermal input |
| M4 | Memory in `~/.rocky_mini/` (footgun 5) | Reinstalls wipe the app dir | Confirmed: dashboard updates run `pip install --force-reinstall` into a shared `apps_venv` | **Holds** |

Test suite note: `pytest -q` currently reports **128 passed**, README says 126. Cosmetic,
will correct the README number when the README is touched anyway.

## The findings in plain terms

**F1 + F2 (launch).** The daemon starts Rocky as `python -m rocky_mini.main`. Our
`__main__` runs the CLI, which serves the sim UI and never connects to the robot; even if
it did, `run()` has the wrong signature and dies with a TypeError (reproduced live against
the sim daemon). What TJ would observe: he presses Play on the dashboard, the app shows
"running", and Rocky sits motionless forever, or errors instantly, depending on which bug
wins. Fix: adopt the scaffold's `__main__` block (`wrapped_run()` + KeyboardInterrupt ->
`stop()`), change `run()` to `run(self, reachy_mini, stop_event)`, and poll `stop_event`.

**F3 (settings UI and Stop).** Because `uvicorn.run()` blocks inside `run()` and installs
its own signal handling, Stop from the dashboard would not set `stop_event`, the 20 s
timeout would expire, and the daemon would SIGKILL the process tree mid-audio. What TJ
would observe: every Stop is a hard kill; and the dashboard never shows the settings link
because it literally greps `main.py` for `custom_app_url = "http://..."`. Fix (D4): set
`custom_app_url = "http://0.0.0.0:8042"` as a class attribute and attach Rocky's routes to
`self.settings_app`; delete the second webserver on the hardware path.

**F4 (motion).** Every MotionThread tick would raise (rocky passes its own Pose dataclass;
the SDK wants a 4x4 matrix plus radians). And because `_set_target` looks the method up
with `getattr(..., None)`, on a handle without `set_target` it would do nothing, silently.
What TJ would observe: 100 exceptions per second in the app log, zero movement, no
breathing, no emotes. Fix: convert `Pose` to the SDK call with `create_head_pose(roll,
pitch, yaw, z)` + `np.deg2rad` for antennas (SDK order is [right, left]) + `body_yaw` in
radians, in one adapter at the seam. The clamp stays in degrees, tests stay pure.

**F5 (speaker).** Piper synthesises at 22050 Hz; the robot plays at 16000 Hz; nothing
converts. What TJ would observe: Rocky's first words come out 27 percent slower and lower,
like a dying tape deck, and out of sync with the wobble. Also nothing calls
`start_playing()`, so most likely no audio at all, just warnings in the log. Fix: the
hardware backend reports its true output rate (`get_output_audio_samplerate()`), the TTS
and canned-WAV path resamples once with `scipy.signal.resample_poly`, and `ReachyMediaIO`
opens the playback pipeline on construction. The Mixer keeps one sample rate end to end.

**F6 (microphone).** `get_audio_sample()` returns `None` until recording is started and
whenever the ring is empty; rocky's wrapper crashes on the first `None`. What TJ would
observe: instant crash the first time anything reads the mic. Fix: call
`start_recording()` on open, map `None` to the empty frame the Protocol already promises,
downmix stereo, and keep 16 kHz (input rate matches rocky's assumption, confirmed).

**F7 (DoA, the one that kills the character).** Wrong object (media, not mini), wrong
unit (radians, not degrees), wrong shape (tuple with a speech flag, not a float), and the
failure mode is `None` forever with no error, because of the `getattr(..., None)` pattern.
What TJ would observe: Rocky never once turns toward him, never orients to his voice, and
nothing in any log says why. Blind gaze by DoA is Rocky's defining behaviour (footgun 9),
so this is the difference between an alien companion and a desk toy that breathes. Fix:
read `mini.media.get_DoA()`, convert radians to the signed yaw degrees MotionManager
expects, use the speech flag as the VAD AND-gate input plan.md already calls for, and
**raise at startup** if the surface is missing on the hardware path. The exact angle sign
mapping (does pi/2 mean front or back) cannot be resolved from the docstring alone; it is
a two-minute bring-up test, listed in bring-up.md.

**F8 (barge-in reaches the speaker).** Rocky's generation counter stops queueing new
audio, but whatever was already handed to GStreamer plays out. The SDK added
`clear_player()` for exactly this; the Conversation App calls it on user interruption.
What TJ would observe without it: he interrupts Rocky, and Rocky keeps talking over him
for another second or two, which reads as rude rather than alien. Fix: hardware AudioIO
gains a `flush()` no-op on the fake/sounddevice paths and `clear_player()` on the robot;
the Mixer calls it when the generation bumps.

**G1 (scope gap: no ears anywhere).** plan.md's architecture names four threads.
AudioInThread (mic ring buffer, Silero VAD, speech events, ~10 Hz DoA poll) does not
exist in the tree: there is no `audio/input.py`, no `audio/vad.py`, no Mixer pump thread,
and nothing anywhere calls `set_doa()` except a unit test. The README's M6 line ("seams +
Fakes done") is honest about this, but it means "fix the seam" is not enough to get a
talking robot: the thread that feeds the seam has to be written. What TJ would observe
with only F1-F8 fixed: a robot that breathes, emotes from the settings page, and speaks
when poked over HTTP, but cannot hear. I want this called out explicitly because it is
new code, not a fix, roughly 150-250 lines plus tests, and it belongs in the Phase 1 plan
you are approving. The sim chat path stays as is; the thread is hardware-only and sits
behind the existing AudioIO Protocol, so the Fakes and all current tests keep working.

## D1 in full: AEC and half-duplex (decisions.md #6)

Decision 6 locked half-duplex because "no AEC on the PC". Two things changed:

1. On the Wireless CM4 the SDK's audio pipeline uses `.asoundrc` ALSA devices that route
   through the XMOS chip's echo-cancelled loopback (source: `audio_gstreamer.py`, which
   selects `reachymini_audio_src/sink` when the robot's asoundrc is present, with the
   comment "route through the XMOS AEC loopback properly").
2. Even on a PC with no ReSpeaker, the SDK now builds `webrtcdsp` + `webrtcechoprobe`
   software AEC into the capture pipeline when it falls back to the default mic.

So the mic samples Rocky hears on the robot have already had his own voice subtracted in
hardware. In behaviour terms: open-mic barge-in (you talk over him, he stops) is probably
viable on the robot, and the ReSpeaker is tunable via `apply_audio_config`; the official
Conversation App ships a tuned parameter set for the same chip (AGC max gain, noise
suppression floors, echo gammas) that we can start from. What I propose, per the
reversal-condition rule: half-duplex stays the default, the settings switch stays, a new
dated decisions.md entry records that the premise changed, and bring-up.md gets a
15-minute test (play a loud chord, speak during it, check VAD does not trigger on Rocky's
own voice). Flip the default only after that test passes on the physical robot. One
residual risk the AEC does not remove: motor noise is not echo (it does not pass through
the loopback reference), so the in-speech VAD threshold sweep in plan.md M8 still matters.

## D2 + D3 in full: getting Rocky onto the robot

The dashboard's install API accepts exactly two source kinds in 1.9.0: a Hugging Face
Space (public or private, via the stored HF token) or a local path. The tutorial as
imagined ("install from the GitHub URL") cannot work. The realistic options:

- **Private HF Space** (recommended): `reachy-mini-app-assistant publish --private` from
  the repo pushes it; the robot's dashboard lists it once TJ's HF account is logged in.
  Cost: Rocky's persona and code live in a private Space; nothing else is exposed.
- **SSH install**: one-time `pip install` into the robot's `apps_venv` from a git clone.
  Works, but it is exactly the kind of first-day SSH step the brief wants removed.

Either way, the shared `apps_venv` already has `reachy-mini` pre-installed by the daemon,
so the SDK does not need to be a dependency of rocky_mini. But the dashboard runs plain
`pip install` with no extras, so anything Rocky needs at runtime on the robot must be a
core dependency. Today that silently drops: `openai` + `httpx` (the Ollama client, so no
brain), `onnxruntime` + `silero-vad` (so no VAD even after G1 is built). Proposal: fold
`[llm]` and `[vad]` into core deps and keep `[dev]` as the only extra. The sim path
imports none of them at module scope (imports are lazy and guarded), so tests still run
without them installed, but a fresh `pip install -e .` gets a bit heavier.

## Scaffold diff (decisions.md #3 reversal condition, executed)

Ran `reachy-mini-app-assistant create scaffold_ref` (SDK 1.9.0) and diffed against the
repo. Structural differences to adopt:

1. **`__main__` block**: scaffold runs `app.wrapped_run()` with KeyboardInterrupt ->
   `app.stop()`. This is the daemon's actual launch contract (F2). Adopt.
2. **`run(self, reachy_mini, stop_event)`** signature and stop_event polling loop. Adopt (F1).
3. **`custom_app_url` class attribute** with a literal URL string (the dashboard regex
   depends on the literal). Adopt (F3).
4. **README front matter**: the scaffold README carries HF Space YAML front matter
   (title, sdk: static, tags including `reachy_mini`). Required for the Space listing if
   we go the D2 publish route. Adopt at publish time, in a way that keeps the GitHub
   README readable (front matter block on top is valid for both).
5. **`keywords = ["reachy-mini-app"]`** in pyproject. Store discovery uses it. Adopt.

Cosmetic differences rejected: flat package layout details, `include-package-data`
wildcard (we list assets explicitly, which is stricter), scaffold's landing `index.html`
(we have a real settings UI), `requires-python >= 3.10` (we require 3.11).

## Official Conversation App: what it does differently, what we take

Read at `pollen-robotics/reachy_mini_conversation_app` (same commit as the HF Space).
Architecture is one asyncio loop with a record task and a play task feeding a realtime
LLM websocket, so its brain shape is different from Rocky's on purpose. The hard-won
details worth stealing, all small:

- Calls `media.start_recording()` and `media.start_playing()` once at startup, then
  **sleeps ~1 s before touching the pipelines** and applies the ReSpeaker tuning config
  after that settle (their `apply_audio_startup_config`). We take the sequencing and the
  tuned XVF3800 values as our starting point.
- Barge-in calls `clear_player()` and then drains its own queues (F8 confirmation).
- Queries `get_input_audio_samplerate()` instead of assuming; pushes mono float32 and
  lets `MediaManager` expand channels (F5/F6 confirmation).
- Wraps `set_target` in try/except with rate-limited error logging, one line per second
  instead of 100. MotionThread should do the same so a transient daemon hiccup does not
  flood the log.
- Runs its control loop at 60 Hz nominal (docstring says "near 100 Hz", constant says
  60.0). Our 100 Hz target stands (plan.md), but it says 60 Hz is enough for lifelike
  motion if the CM4 budget gets tight, which is a useful fallback fact for Phase 3.
- Has an inactivity timeout thread that puts the robot to sleep. Rocky has sleep-watch
  already; noted, not taken.

Not taken: their whole realtime-websocket audio streaming stack, camera head tracking
(footgun 9 forbids it), their tool system (Rocky's is built and tested), their memory
(same).

## Reuse map (note for a future session, no action now)

For the two future apps (spaced-repetition tutor, narrow voice agent). "Generic" means:
no Rocky persona leakage, would move to a shared package mostly as-is. Boundary noted
only; extracting now would shape a framework around a sample size of one.

| Module | Rocky-specific or generic |
|---|---|
| `motion/manager.py`, `motion/pose.py`, `motion/idle.py` (breathing) | Generic (the layered 100 Hz owner + clamp is any-app) |
| `motion/emotes.py`, `motion/wobble.py`, sleep-watch parts of `idle.py` | Rocky (trajectories and wobble feel are character) |
| `audio/io.py` (AudioIO Protocol + Fake + backends) | Generic, this is the seam itself |
| `audio/output.py` Mixer (buses, generation flush, frame pump) | Generic core; ring-mod hook is Rocky |
| `audio/dsp.py` (ring mod, soft clip), `audio/chords.py` | Rocky (soft clip alone is generic but trivial) |
| `brain/persona.py`, `tics.py`, `naivety.py`, `auditor.py`, `curiosity.py`, `reflection.py` | Rocky |
| `brain/chunker.py` (sentence/tag splitting) | Generic (tic insertion hook is Rocky) |
| `brain/llm.py`, `speech/stt.py`, `speech/tts.py` (Protocols + Fakes + remote clients) | Generic |
| `brain/tools.py` (pydantic tool executor, one-retry) | Generic shell, Rocky tool set |
| `memory/store.py`, `memory/models.py` (JSONL, confidence lifecycle, digest, export) | Generic (the tutor inverts exactly this store) |
| `turn.py` ConversationLoop | Mixed: the state machine shape is generic, the ack-chord/tic/audit wiring is Rocky |
| `app.py` + `static/` settings UI | Generic shell, Rocky content |
| `config.py`, `main.py` app shell | Generic |
| `server/` (faster-whisper + Piper FastAPI) | Generic |
| `finetune/` | Rocky |

## What cannot be verified without the robot

These stay open no matter how good the sim work is, and each becomes a bring-up.md step:

1. DoA angle convention sign and zero direction on the actual XVF3800 (F7).
2. Whether hardware AEC suppresses Rocky's own voice well enough for open-mic (D1).
3. Motor-noise floor into the mic array, and the VAD threshold that rides above it.
4. Real speaker playback pitch (F5 fix correctness end to end).
5. CM4 headroom: ring mod + soft clip + 100 Hz motion + VAD concurrently (plan budget
   <35 percent core, <300 MB RSS). Per-tick PC profiling in Phase 3 gives an estimate,
   not an answer; the CM4 measurement can fail and has a fallback (60 Hz motion, F8 note).
6. ReSpeaker firmware version >= 2.1.0 (DoA returns None below it).
7. `.env` and settings-UI config flow on the robot filesystem (D2 aftermath).
8. `imu['temperature']` actually returning data on the Wireless (M3 says the surface
   exists; the value needs hardware).

## Phase 1a verification (sim daemon, real app path, post-fix)

Backend: the daemon's mockup backend, not MuJoCo. Reason: the mockup behaves
identically to a real robot from an app's point of view (daemon.py's own comment), the
MuJoCo package is not installed in this environment, and the mockup gives a cleaner
loop with no physics-step timing noise. All commands below ran with
`ROCKY_MEDIA_BACKEND=no_media ROCKY_AUDIO_BACKEND=fake` (this container has no audio
device; motion and lifecycle are the real path).

Launch through the daemon's exact shape (`python -u -m rocky_mini.main`), settings UI
served by the SDK base class on 8042:

```
GET / -> 200
GET /api/state -> {"model":"qwen2.5:7b-instruct", ... "sim_mode":true}
POST /api/emote {"name":"jazz_hands"} -> {"fired":"jazz_hands"}
```

Motion flowing end to end (daemon's present pose while Rocky breathes; antennas
oscillate near 10 deg, head z bobs at the 0.25 Hz breath):

```
/api/state/present_antenna_joint_positions -> [0.195, 0.197] then [0.176, 0.182] then [0.155, 0.159]
```

Stop (SIGINT, what the dashboard sends), ordered shutdown, and rest pose:

```
exit after SIGINT: 0.89 s   (daemon force-kills at 20 s; we are nowhere near)
log: "Rocky Mini shutdown complete; memory fsynced per write." / "App is stopping..."
final antennas: [0.17453, 0.17453] rad = exactly 10.0 deg (footgun 6 held through shutdown)
```

60 s at 100 Hz through `wrapped_run()` with an observation-only wrapper recording
every `set_target` call (timestamp + calling thread):

```
wall time incl. shutdown: 60.63 s
set_target calls: 5955  (expected ~6000 at 100 Hz + ramp)
calling threads: {'MotionThread'}          <- exactly one owner (footgun 1)
mean rate: 98.79 Hz
tick interval ms: p50=10.11 p95=10.19 p99=10.33 max=11.50
ticks > 15 ms late (interval > 25 ms): 0
```

The mean is 98.79 Hz, not 100: the sleep-based loop overshoots by ~0.1 ms per tick.
Honest number, kept as is; the CM4 measurement in bring-up.md is the one that matters.

## Evidence appendix (live probe output, mockup daemon, SDK 1.9.0)

```
$ python probe_lead1.py          # RockyMiniApp launched exactly as the daemon would
has SDK: True
LEAD 1 CONFIRMED, TypeError: RockyMiniApp.run() takes from 1 to 2 positional arguments but 3 were given

$ python probe_settarget.py      # live handle against the mockup daemon
rocky call broken: AttributeError: 'Pose' object has no attribute 'shape'
correct call OK: head 4x4 + antennas rad + body_yaw accepted by mockup daemon
DoA on handle: False | DoA on media: True

$ pytest -q                      # sim/test path untouched by the SDK install
128 passed in 2.02s
```
