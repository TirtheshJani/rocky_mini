"""Per-tick and per-frame CPU cost of Rocky's hot loops, on whatever machine runs it.

This answers "how much core does Rocky's own math need", separate from the SDK and
network. Run it on the PC for the Phase 3 headroom estimate and on the CM4 at
bring-up for the real answer (plan.md budget: <35 percent of one core, <300 MB RSS).

    python scripts/profile_budget.py

Costs measured: MovementManager.tick (100 Hz), the pose->SDK adapter (100 Hz),
ring mod + soft clip per mixer frame (50 Hz at 320 samples), resample per TTS
second, and Silero VAD per 512-sample frame (31.25 Hz). Each line reports the
per-call cost and what fraction of one core the loop's real-time cadence needs.
"""

from __future__ import annotations

import resource
import time

import numpy as np

from rocky_mini.audio.dsp import resample, ring_mod, soft_clip
from rocky_mini.motion.manager import MovementManager
from rocky_mini.motion.pose import Pose, sdk_targets


def bench(label: str, fn, calls_per_s: float, n: int = 2000, warmup: int = 50) -> None:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    per_call = (time.perf_counter() - t0) / n
    core = per_call * calls_per_s * 100.0
    print(f"{label:<38} {per_call * 1e6:9.1f} us/call  @{calls_per_s:7.2f}/s = {core:6.2f}% core")


def main() -> None:
    print(f"machine: {open('/proc/cpuinfo').read().count('processor')} cores")

    mgr = MovementManager()
    mgr.set_doa(25.0)
    mgr.start_emote("jazz_hands")
    t = [0.0]

    def tick():
        t[0] += 0.01
        return mgr.tick(t[0], 0.01)

    bench("MovementManager.tick", tick, 100.0)
    pose = Pose(head_yaw=10, head_pitch=5, head_z=0.005)
    bench("sdk_targets (pose -> matrix)", lambda: sdk_targets(pose), 100.0)

    frame = (np.random.default_rng(0).standard_normal(320) * 0.2).astype(np.float32)
    state = {"phase": 0.0}

    def dsp_frame():
        out, state["phase"] = ring_mod(frame, 140.0, 16000, state["phase"])
        soft_clip(out)

    bench("ring_mod + soft_clip (320 samples)", dsp_frame, 50.0)

    second = (np.random.default_rng(1).standard_normal(22050) * 0.2).astype(np.float32)
    bench("resample 22050->16000 (1 s of TTS)", lambda: resample(second, 22050, 16000), 0.5, n=200)

    try:
        from rocky_mini.audio.vad import SileroVAD

        vad = SileroVAD()
        vframe = (np.random.default_rng(2).standard_normal(512) * 0.1).astype(np.float32)
        bench("SileroVAD.probability (512 samples)", lambda: vad.probability(vframe), 16000 / 512, n=500)
    except Exception as exc:
        print(f"SileroVAD: skipped ({exc})")

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\npeak RSS this process: {rss_mb:.0f} MB  (budget on the CM4: <300 MB for the app)")
    print("sum the '% core' column for the always-on load; STT/TTS/LLM live on the PC, not here.")


if __name__ == "__main__":
    main()
