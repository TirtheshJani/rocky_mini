"""Motion-loop jitter gate: 60 s of MotionThread through the real app path.

This is the exact harness behind the Phase 1a numbers in docs/hardware-audit.md.
Re-run it on the robot's CM4 (the 1a addendum TJ asked for) to check the four cores
hold the 100 Hz loop before Phase 1b's audio load lands on top.

On the CM4 (dev loop 2: SSH, editable install):

    ROCKY_AUDIO_BACKEND=fake python scripts/gate_jitter.py

ROCKY_AUDIO_BACKEND=fake keeps the run motion-only, matching the PC baseline.
On a dev PC, also set ROCKY_MEDIA_BACKEND=no_media and run a sim daemon first:

    reachy-mini-daemon --mockup-sim --headless --no-media --autostart --no-wake-up-on-start

Observation only: ReachyMini.set_target is wrapped to record (timestamp, thread)
and then called unchanged. Numbers to beat (PC baseline, mockup daemon):
mean 98.79 Hz, tick p50 10.11 ms, p99 10.33 ms, max 11.50 ms, 0 ticks > 25 ms.
"""

import argparse
import threading
from time import perf_counter

import numpy as np
from reachy_mini import ReachyMini

DURATION_S = 60.0


def _vad_load(stop: threading.Event) -> None:
    """Real-time Silero inference on synthetic frames: Phase 1b's CPU cost.

    Run the gate once bare and once with --load vad. The delta between the two
    reports is what the voice-in thread costs the motion loop on this machine,
    which answers "did 1b push us off 100 Hz, or was the loop never holding it".
    """
    from rocky_mini.audio.vad import SileroVAD

    vad = SileroVAD()
    frame = (np.random.default_rng(0).standard_normal(512) * 0.1).astype(np.float32)
    period = 512 / 16000  # real-time cadence: 31.25 inferences/s
    inferences = 0
    t_next = perf_counter()
    while not stop.is_set():
        vad.probability(frame)
        inferences += 1
        t_next += period
        delay = t_next - perf_counter()
        if delay > 0:
            stop.wait(delay)
    print(f"vad load thread: {inferences} inferences ({inferences / DURATION_S:.1f}/s)")

records: list[tuple[float, str]] = []
_orig = ReachyMini.set_target


def _observed(self, head=None, antennas=None, body_yaw=None):
    records.append((perf_counter(), threading.current_thread().name))
    return _orig(self, head=head, antennas=antennas, body_yaw=body_yaw)


ReachyMini.set_target = _observed

from rocky_mini.main import RockyMiniApp  # noqa: E402  (after the wrap, on purpose)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load", choices=["vad"], default=None,
        help="add Phase 1b's VAD inference load at real-time cadence during the run",
    )
    args = parser.parse_args()

    app = RockyMiniApp()
    load_thread = None
    if args.load == "vad":
        load_thread = threading.Thread(
            target=_vad_load, args=(app.stop_event,), daemon=True, name="VadLoad"
        )
        load_thread.start()
    threading.Timer(DURATION_S, app.stop_event.set).start()  # = dashboard Stop
    t0 = perf_counter()
    app.wrapped_run()
    if load_thread is not None:
        load_thread.join(timeout=2.0)
    wall = perf_counter() - t0

    ts = np.array([r[0] for r in records])
    threads = {r[1] for r in records}
    dt = np.diff(ts) * 1000.0
    print(f"wall time incl. shutdown: {wall:.2f} s")
    print(f"set_target calls: {len(ts)}  (expected ~{int(DURATION_S * 100)} at 100 Hz + ramp)")
    print(f"calling threads: {threads}")
    print(f"mean rate: {1000.0 / dt.mean():.2f} Hz")
    print(
        f"tick interval ms: p50={np.percentile(dt, 50):.2f} p95={np.percentile(dt, 95):.2f} "
        f"p99={np.percentile(dt, 99):.2f} max={dt.max():.2f}"
    )
    print(f"ticks > 15 ms late (interval > 25 ms): {(dt > 25.0).sum()}")


if __name__ == "__main__":
    main()
