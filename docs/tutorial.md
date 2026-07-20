# Rocky tutorial: from an assembled box to a first conversation

Audience: TJ, on the day the Reachy Mini Wireless arrives, with the robot, the PC, and no
memory of this codebase. Run it top to bottom. Every step says what to do, what you should
see, and what to do when you do not.

Confidence markers, because honesty is the point:
- **[sim-verified]** the code path was exercised against the MuJoCo/mockup daemon on the PC.
- **[needs-robot]** first proven here, on hardware.

## 0. Assemble the robot (not covered here)

Follow Pollen Robotics' official assembly guide for the Reachy Mini Wireless and stop when
it is assembled and powered:

  https://docs.pollen-robotics.com/

Do not follow any assembly steps invented here; getting hardware assembly wrong costs
hardware. Come back when the robot is built and powered on.

## 1. First boot and Wi-Fi onboarding  [needs-robot]

- Power the robot. Follow Pollen's onboarding to join it to your Wi-Fi (it exposes a setup
  network or captive portal on first boot; the official guide covers the exact flow).
- Expect: the robot is on your LAN and its dashboard is reachable.
- Fail: if it never joins, re-run onboarding close to the router. Note the robot's IP from
  your router's client list.

## 2. Dashboard and Hugging Face login  [needs-robot]

- Open the Reachy dashboard in a browser (the onboarding step gives you the address).
- Log in to Hugging Face when prompted (needed to install apps from a Space).
- Expect: the dashboard lists installed apps and the daemon shows healthy.
- Fail: if the daemon is unhealthy, reboot and check step 1.

## 3. Bring up the brain on the PC  [sim-verified for the software, needs your GPU for models]

The robot is a thin client. The PC does the thinking. On the PC (Windows 11, RTX 4080):

```bash
# 3a. Ollama and the model
ollama pull qwen2.5:7b-instruct-q5_K_M          # about 5.5 GB, fits 16 GB VRAM
ollama serve                                     # or the Windows service; leave it running

# 3b. The speech server (STT + TTS)
pip install -r server/requirements.txt           # faster-whisper + Piper
#   download the Piper voice en_US-lessac-medium (.onnx + .json) next to where you run it
uvicorn server.app:app --host 0.0.0.0 --port 8123
```

- Expect: `curl http://localhost:11434/api/tags` lists the model, and
  `curl http://localhost:8123/health` returns `{"ok": true, ...}`.
- Fail: if `/health` reports the models not loaded, they load lazily on first `/stt` and
  `/tts` call. Confirm the Piper voice files are present where you launched uvicorn.

## 4. LAN IP and firewall  [needs-robot]

- Find the PC's LAN IP (`ipconfig` on Windows). Open inbound ports 11434 (Ollama) and 8123
  (speech) in the Windows firewall for your local network.
- Expect: from the robot, `curl http://<pc-ip>:11434/api/tags` and
  `curl http://<pc-ip>:8123/health` both answer.
- Fail: closed port or wrong IP. This is the most common first-day snag.

## 5. Install Rocky on the robot  [needs-robot]

Two loops, fastest first. Use loop A for day-one and stable updates, loop B for iterating.

- **Loop A, release path (Hugging Face Space):** publish this repo as a PRIVATE HF Space,
  then install it from the dashboard by its Space URL. The dashboard installs from a Space
  or a local folder, not from a raw GitHub URL. A private Space is fine; your repo is
  already public on GitHub, so a private Space is strictly less exposed.
- **Loop B, on-robot iteration:** SSH into the robot once, `git clone` this repo, and
  `pip install -e .` into the robot's apps venv. After that, each change is `git pull` plus
  restart from the dashboard. No re-publish.

- Expect: "Rocky Mini" appears in the dashboard app list.
- Fail: if it does not appear, the `reachy_mini_apps` entry point did not register; confirm
  the install finished without error.

## 6. Point Rocky at the brain  [needs-robot]

Set the connection either in the settings UI (the app serves it) or in
`~/.rocky_mini/.env` on the robot (the home dir survives app reinstalls; footgun 5):

```
ROCKY_LLM_BACKEND=ollama
ROCKY_LLM_BASE_URL=http://<pc-ip>:11434/v1
ROCKY_SPEECH_BASE_URL=http://<pc-ip>:8123
ROCKY_MODEL=qwen2.5:7b-instruct
ROCKY_AUDIO_BACKEND=reachy
```

- Expect: the settings UI shows the model and the brain URL.
- Fail: if the app still behaves like the sim (rule-based replies), `ROCKY_LLM_BACKEND` is
  not set to `ollama`; the default is the Fake on purpose.

## 7. First conversation  [needs-robot]

- Start Rocky from the dashboard. Say "Hello Rocky."
- Expect: an ack chord within about 150 ms, then a spoken in-character reply. Teach him a
  fact ("A taco is food"), restart the app, and confirm he still knows it (memory persists
  in `~/.rocky_mini/`).
- Fail: silent or wrong-pitch audio sends you to `docs/bring-up.md` steps 6, 7, and 9.

## 8. Run the bring-up checklist  [needs-robot]

Now work through `docs/bring-up.md` in order. That is where the motion, antenna, DoA,
speaker-pitch, CM4 headroom, and failure-drill checks live, each with a fail action.

## 9. Train the Rocky LoRA (optional, later)  [sim-verified pipeline, needs WSL2 + GPU to run]

Once you have had a few conversations, give Rocky his trained-in voice. See
`finetune/README.md` for the full runbook. In short: build the dataset (authored gold plus
off-novel synthetic Q&A plus a neutral slice plus optional paraphrased book calibration),
QLoRA train in WSL2, merge and quantize to GGUF, `ollama create rocky`, then gate it:

```bash
python finetune/eval.py --model rocky:latest --baseline qwen2.5:7b-instruct
```

It ships only if it beats the stock baseline on naivety, tics, and tool validity, passes
the memorization probe, and keeps its general ability. Then set `ROCKY_MODEL=rocky:latest`.
The dataset and model stay on your machines (decisions.md 14).

## 10. Publishing (optional, not the main path)

You may publish Rocky as a Hugging Face Space to share or reinstall easily. Rocky is
personal: a public Space exposes the persona, the settings UI, and whatever is in the repo.
Keep the Space private unless you deliberately want it public. Memory (`~/.rocky_mini/`) is
never part of the app package and is not published.

## 11. Rollback and logs

- Uninstall from the dashboard app list, or `pip uninstall rocky_mini` in the robot venv.
- Memory survives an app reinstall because it lives in `~/.rocky_mini/`. To reset Rocky's
  learning, back up and clear that directory.
- Logs: the app logs to the daemon's app log (visible from the dashboard) and to stdout when
  launched over SSH. Motion, audio, and shutdown steps log at INFO.
- To get back to a known-good state: Stop the app, confirm the daemon is healthy (bring-up
  step 1), and restart.
