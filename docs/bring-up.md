# Rocky bring-up checklist (on the physical Reachy Mini)

Run these in order on the day the robot is assembled and on the LAN. The order puts the
cheapest and most likely failures first, so you find a dead daemon before you chase a
latency number. Each step has a command, what you should see, and what to do if you do not.

Every step is marked:
- **[sim-verified]** the same code path was exercised against the MuJoCo/mockup daemon.
- **[needs-robot]** never run on hardware; the first real test is here.

Do not skip a failing step. A red step early is usually the real cause of a red step later.

## 0. Prerequisites

The brain PC is on and reachable, Ollama is running, and the speech server is up (see
`docs/tutorial.md` steps for the PC side). You know the PC's LAN IP.

## 1. Daemon reachable  [sim-verified]

- Command: on the robot, `curl -s http://localhost:8000/ | head` (the Reachy daemon).
- Expect: an HTTP response, not connection refused.
- Fail: the daemon is not running. Reboot the robot or restart the daemon service. Nothing
  below can work without it; the daemon owns the motors and sensors.

## 2. App launches and stops cleanly  [sim-verified]

- Command: install Rocky to the dashboard (tutorial step), then Start it from the dashboard.
- Expect: the app appears and starts; the settings UI answers on its port. Press Stop.
- Expect on Stop: the app exits within a couple of seconds, antennas ramp to neutral, and
  the log shows the ordered shutdown (audio, asyncio, mixer drain, ramp). This exact path
  was proven in sim: SIGINT to clean exit in about 0.9 s.
- Fail: if Stop hangs, capture the log. The launch contract (`run(reachy_mini, stop_event)`)
  and the settings-webserver ownership were the audit's F1/F2 fixes; a hang means a
  regression there.

## 3. Motors and homing  [needs-robot]

- Command: watch the head after Start.
- Expect: a slow breathing idle (about 0.25 Hz), the head bobbing gently, no snapping.
- Fail: if the head jerks or throws, Stop immediately. Check the pose clamps in
  `motion/manager.py` (pitch/roll, head-vs-body yaw). Do not leave a jerking head running.

## 4. Antennas rest near 10 degrees, never 0  [needs-robot]  (footgun 6)

- Command: observe the antennas at idle and after Stop.
- Expect: they sit around 10 degrees and never drive to 0. Zero looks dead.
- Fail: if they reach 0, stop and check the neutral pose and the shutdown ramp target.

## 5. 100 Hz motion holds on the CM4  [needs-robot]  (the big unknown)

- Command: `ROCKY_AUDIO_BACKEND=fake python scripts/gate_jitter.py` on the robot, then
  again with `--load vad`.
- Expect: mean rate near 100 Hz and a small jitter tail. The PC baseline to beat is in the
  script docstring (mean 98.79 Hz, p99 10.33 ms, 0 late ticks). Every set_target call must
  come from MotionThread only.
- Fail: if the CM4 comes back near 60 Hz with a fat tail, that is the reference app's own
  loop rate and a real constraint, not a bug. Drop the motion loop to 60 Hz and re-run.
  Record the number in `docs/hardware-audit.md` (the "1a addendum" slot). Also run
  `python scripts/profile_budget.py` here for the per-tick core budget (target under 35
  percent of one core, under 300 MB RSS); it is POSIX-only and runs on the CM4.

## 6. Microphone capture  [needs-robot]

- Command: speak near the robot with the app running and watch the log or UI for speech
  events.
- Expect: VAD opens on speech and closes after the 0.6 s hangover; a transcript arrives.
- Fail: no events means the mic read path is wrong. The audit's F5/F6 fixes made the mic
  read and DoA surfaces fail loudly if missing; read the raised error.

## 7. Speaker playback at the correct pitch  [needs-robot]  (lead 4)

- Command: trigger a spoken reply (type in the settings UI chat, or speak a turn).
- Expect: Rocky's voice plays at normal speed and pitch, ring-modulated and Eridian.
- Fail: if it sounds too fast or too slow or chipmunked, the 22050 to 16000 resample at the
  mixer boundary is not firing on hardware. This is exactly the lead-4 fix; confirm the
  device output rate and that `Mixer.push_voice` resamples when src rate differs.

## 8. DoA returns real angles as you move  [needs-robot]  (footgun 9)

- Command: speak from the left, then the right, then in front, and watch the head.
- Expect: the head leads toward your voice (about 15 degrees) with the body following, and
  the aim tracks as you move. This is the only way Rocky knows where you are; he has no eyes.
- Fail: if the head never turns, DoA is returning None or the wrong unit. The seam now
  raises loudly when the surface is missing (audit F5), so read the error. Confirm the
  hardware speech flag AND our VAD both agree before Rocky orients (the plan's AND-gate).

## 9. LAN reach to the brain server  [needs-robot]

- Command: from the robot, `curl http://<pc-ip>:11434/api/tags` (Ollama) and
  `curl http://<pc-ip>:8123/health` (speech server).
- Expect: JSON from both.
- Fail: firewall or wrong IP. Open the ports on the PC and confirm the IP in
  `~/.rocky_mini/.env` (or the settings UI). Rocky is dead without the PC awake on the LAN;
  that is accepted for a personal project and covered by the M9 offline drill below.

## 10. First real conversation turn  [needs-robot]

- Command: say "Hello Rocky" and wait.
- Expect: an ack chord within about 150 ms (the perceived-response mask), then a spoken
  in-character reply. Watch the latency meter in the UI against the 2.5 s budget.
- Fail: if it is silent, walk back up: STT (step 6), brain (step 9), TTS (step 7).

## 11. Honest latency and KV-cache reuse  [needs-robot for the LAN number]

- Command: on the PC, `python scripts/measure_latency.py --turns 8` (full harness) and
  `python scripts/check_kv_reuse.py --model <your model>`.
- Expect: a p50 at or under 2.5 s measured from the last voiced frame including the 0.6 s
  hangover (footgun 7), and a REUSE CONFIRMED verdict from the KV-cache check.
- Fail: if the p50 blows the budget, the harness prints the per-stage breakdown; attack the
  largest stage. If KV reuse is not confirmed, the persona prefix is being re-ingested every
  turn; confirm `keep_alive=-1` and that the byte-stable prefix is unchanged.

## 12. M9 failure drills  [needs-robot]

Run each and confirm Rocky degrades gracefully, never crashes:
- Brain server down (stop Ollama mid-session): expect the canned "Signal bad" path and
  Eridian-only mode (chords plus UI subtitles), then recovery when it returns.
- Wi-Fi pulled mid-turn: same offline fallback, no crash.
- Ollama cold (first turn after idle): expect a slower first turn, then warm.
- SIGINT mid-TTS: facts already learned are intact on restart (fsync memory).
- Rapid barge-in: interrupting mid-sentence stops the old audio (generation flush) and
  starts the new turn.

## 13. If the barge-in AEC test passes, consider full-duplex  [needs-robot]  (decision 9)

- Command: with the robot speaking a loud chord, speak over it and watch VAD.
- Expect (to flip the default): VAD does not trigger on Rocky's own voice.
- Only then flip open-mic barge-in on. Half-duplex stays the default until this passes on
  the real robot; do not flip it from a keyboard (decisions.md 6 and 9).
