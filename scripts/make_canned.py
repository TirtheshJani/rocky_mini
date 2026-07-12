"""Generate placeholder canned-fallback WAVs into rocky_mini/assets/canned.

These are the offline "brain server down / Wi-Fi pulled" fallback clips (plan M9). Real
lines should be Piper-synthesized in-character and ring-modulated; until then these are
deterministic Eridian chord stand-ins so the CannedTTS path has real audio to play.

Run:  python scripts/make_canned.py
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocky_mini.audio.chords import ChordBank  # noqa: E402
from rocky_mini.audio.dsp import ring_mod, soft_clip  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "rocky_mini" / "assets" / "canned"

# caption -> the chord that stands in for the (future) spoken line.
CANNED = {
    "signal_bad": "signal_lost",
    "wait": "thinking",
    "pc_asleep": "sleepy",
    "still_here": "still_watching",
    "cannot_reach": "error",
    "reconnecting": "listening",
}


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bank = ChordBank(sample_rate=22050)
    for name, stinger in CANNED.items():
        chord = bank.render(stinger)
        # Two beats + ring-mod for the alien timbre.
        clip = np.concatenate([chord, np.zeros(int(0.08 * bank.sample_rate), dtype=np.float32), chord])
        modded, _ = ring_mod(clip, 140.0, bank.sample_rate)
        out = soft_clip(0.6 * clip + 0.4 * modded)
        write_wav(OUT / f"{name}.wav", out, bank.sample_rate)
    print(f"wrote {len(CANNED)} canned WAVs to {OUT}")


if __name__ == "__main__":
    main()
