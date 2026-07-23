"""One-time dev-PC build step: render Rocky's short in-character emote lines to WAVs.

Each line is POSTed to the PC speech server /tts endpoint (Piper with Rocky's deadpan
prosody applied server-side, returning raw int16 little-endian mono PCM at 22050 Hz).
The bytes are written verbatim to a CLEAN mono 22050 Hz int16 WAV per emote at
rocky_mini/assets/canned/emotes/<name>.wav .

The clips carry NO ring modulation on purpose: the Mixer ring-modulates every VoiceBus
item at playout (audio/output.py _drain_voice), so baking the electronic edge in here
would double it. The robot loads the committed WAVs offline, so this script only runs on
the dev PC and is never imported at runtime.

Run:  python scripts/make_emote_voices.py
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import httpx

BASE_URL = os.environ.get("ROCKY_SPEECH_BASE_URL", "http://localhost:8123")
TIMEOUT_S = 60.0
SAMPLE_RATE = 22050

OUT = Path(__file__).resolve().parents[1] / "rocky_mini" / "assets" / "canned" / "emotes"

# emote name -> short, deadpan, movie-flavored line. Telegraphic but warm; the Mixer
# adds the electronic timbre at playout, so these stay clean voice.
LINES = {
    "amaze": "Amaze. Amaze. Amaze.",
    "happy": "Happy. Happy. Happy.",
    "laugh": "Heh. Heh. Heh.",
    "sad": "Sad. Rocky sad.",
    "downcast": "Not good. Not good.",
    "surprised": "Oh. Question?",
    "scared": "Danger, question?",
    "grateful": "Thank. Thank. Thank.",
    "bored": "Wait. Wait. Wait.",
    "yes": "Yes. Good.",
    "no": "No. No.",
    "welcome": "I fist your bump, question?",
    "sleep": "Sleep now. Rocky rest.",
}


def synth(client: httpx.Client, text: str) -> bytes:
    """POST one line to /tts and return the raw int16 PCM bytes. Raises on non-200."""
    resp = client.post("/tts", json={"text": text})
    resp.raise_for_status()
    return resp.content


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> int:
    """Write raw int16 mono PCM bytes to a proper WAV. Returns the sample count."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return len(pcm_bytes) // 2


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    succeeded: list[str] = []
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT_S) as client:
        for name, line in LINES.items():
            try:
                pcm_bytes = synth(client, line)
            except Exception as exc:  # noqa: BLE001 - report the exact error and stop
                print(f"FAILED at {name!r} via {BASE_URL}: {type(exc).__name__}: {exc}")
                print(f"stopping; no WAV written for {name!r} (voice is not fabricated)")
                break
            n_samples = write_wav(OUT / f"{name}.wav", pcm_bytes, SAMPLE_RATE)
            seconds = n_samples / SAMPLE_RATE
            print(f"{name}.wav: {n_samples} samples, {seconds:.3f} s  ({line})")
            succeeded.append(name)

    print(f"wrote {len(succeeded)}/{len(LINES)} emote WAVs to {OUT}")
    if succeeded:
        print("succeeded: " + ", ".join(succeeded))
    return 0 if len(succeeded) == len(LINES) else 1


if __name__ == "__main__":
    sys.exit(main())
