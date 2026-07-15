"""MovementManager: the single owner of set_target, running at 100 Hz.

Layering:
  primary   = breathing (or sleep-watch) + at most one blended trajectory emote
  secondary = jazz-hands reflex + speech wobble + DoA orient  (additive, not blended)
final = clamp(primary_blended + secondary)

Only run() calls mini.set_target(), exactly once per tick, and the SDK is import-guarded
so this module and its tests never require the robot. tick() is pure and returns the Pose
that run() would send, which is what the tests exercise.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from .emotes import PRIMARY_EMOTES, REFLEX_EMOTES, Trajectory, jazz_hands_delta
from .idle import breathing_delta, sleepwatch_delta
from .pose import Pose, apply_deltas, blend_factor, clamp_pose, lerp_pose, sdk_targets
from .wobble import WobbleTracker

logger = logging.getLogger("rocky_mini.motion")


class MovementManager:
    def __init__(self, blend_tau: float = 0.4, clock: Callable[[], float] = time.perf_counter) -> None:
        self.blend_tau = blend_tau
        self.clock = clock
        self.current_primary = Pose()
        self.sleep_watch = False
        self.doa_deg: float | None = None
        self.wobble = WobbleTracker()
        self._primary_emote: tuple[Trajectory, float] | None = None  # (traj, start_t)
        self._reflex: tuple[str, float] | None = None  # (name, start_t)

    # -- inputs ------------------------------------------------------------
    def start_emote(self, name: str, t: float | None = None) -> None:
        t = self.clock() if t is None else t
        if name in REFLEX_EMOTES:
            self._reflex = (name, t)
        elif name in PRIMARY_EMOTES:
            self._primary_emote = (PRIMARY_EMOTES[name], t)

    def set_doa(self, deg: float | None) -> None:
        self.doa_deg = deg

    def set_sleep_watch(self, on: bool) -> None:
        self.sleep_watch = on

    def feed_audio_rms(self, rms: float, t: float | None = None) -> None:
        self.wobble.feed(rms, self.clock() if t is None else t)

    # -- composition -------------------------------------------------------
    def _primary_target(self, t: float) -> Pose:
        base = Pose()
        idle = sleepwatch_delta(t, doa_deg=self.doa_deg) if self.sleep_watch else breathing_delta(t)
        deltas = [idle]
        if self._primary_emote is not None:
            traj, start = self._primary_emote
            since = t - start
            if traj.finished(since):
                self._primary_emote = None
            else:
                deltas.append(traj.sample(since))
        return apply_deltas(base, deltas)

    def _secondary_deltas(self, t: float) -> list[dict]:
        deltas: list[dict] = [self.wobble.delta(t)]
        if self._reflex is not None:
            name, start = self._reflex
            since = t - start
            reflex_delta = jazz_hands_delta(since) if name == "jazz_hands" else {}
            if not reflex_delta:
                self._reflex = None
            else:
                deltas.append(reflex_delta)
        if self.doa_deg is not None and not self.sleep_watch:
            # Head leads toward the sound; body follows more slowly (handled by the yaw clamp).
            deltas.append({"head_yaw": 0.6 * self.doa_deg, "body_yaw": 0.2 * self.doa_deg})
        return deltas

    def tick(self, t: float, dt: float) -> Pose:
        target = self._primary_target(t)
        self.current_primary = lerp_pose(self.current_primary, target, blend_factor(dt, self.blend_tau))
        final = apply_deltas(self.current_primary, self._secondary_deltas(t))
        return clamp_pose(final)

    # -- live loop ---------------------------------------------------------
    def run(self, mini: object, stop, hz: float = 100.0, ramp_s: float = 0.6) -> None:
        """Drive set_target at hz until stop is set, then ramp to neutral.

        The ramp is part of the ordered shutdown (plan.md): Rocky settles to the
        rest pose (antennas at ~10 deg, footgun 6) instead of freezing mid-emote.
        """
        period = 1.0 / hz
        last = self.clock()
        while not stop.is_set():
            now = self.clock()
            dt = now - last
            last = now
            pose = self.tick(now, dt)
            self._set_target(mini, pose)
            sleep_left = period - (self.clock() - now)
            if sleep_left > 0:
                time.sleep(sleep_left)
        self._ramp_to_neutral(mini, hz=hz, duration=ramp_s)

    def _ramp_to_neutral(self, mini: object, hz: float, duration: float) -> None:
        neutral = Pose()
        steps = max(1, int(duration * hz))
        start = self.current_primary
        for i in range(1, steps + 1):
            pose = clamp_pose(lerp_pose(start, neutral, i / steps))
            self._set_target(mini, pose)
            time.sleep(1.0 / hz)
        self.current_primary = neutral

    def _set_target(self, mini: object, pose: Pose) -> None:
        # The ONE place set_target is called (footgun 1). SDK 1.9.0 contract:
        # head 4x4 matrix, antennas [right, left] radians, body_yaw radians.
        head, antennas, body_yaw = sdk_targets(pose)
        try:
            mini.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            self._set_target_errors = 0
        except AttributeError:
            # A handle without set_target means the SDK contract changed or the
            # wrong object was passed. Never shrug this off (audit F4).
            raise
        except Exception as exc:
            # Transient daemon hiccups happen at 100 Hz; log once per second,
            # not one hundred times (pattern from the official Conversation App).
            self._set_target_errors = getattr(self, "_set_target_errors", 0) + 1
            now = self.clock()
            if now - getattr(self, "_last_error_log", 0.0) >= 1.0:
                logger.error(
                    "set_target failed (%d since last report): %s",
                    self._set_target_errors, exc,
                )
                self._last_error_log = now
                self._set_target_errors = 0
