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

import threading
from time import perf_counter

import numpy as np
from reachy_mini import ReachyMini

DURATION_S = 60.0

records: list[tuple[float, str]] = []
_orig = ReachyMini.set_target


def _observed(self, head=None, antennas=None, body_yaw=None):
    records.append((perf_counter(), threading.current_thread().name))
    return _orig(self, head=head, antennas=antennas, body_yaw=body_yaw)


ReachyMini.set_target = _observed

from rocky_mini.main import RockyMiniApp  # noqa: E402  (after the wrap, on purpose)


def main() -> None:
    app = RockyMiniApp()
    threading.Timer(DURATION_S, app.stop_event.set).start()  # = dashboard Stop
    t0 = perf_counter()
    app.wrapped_run()
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
