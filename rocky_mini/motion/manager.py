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
from .idle import IdlePolicy, breathing_delta, sleepwatch_delta
from .pose import Pose, apply_deltas, blend_factor, clamp_pose, lerp_pose, sdk_targets
from .wobble import WobbleTracker

logger = logging.getLogger("rocky_mini.motion")

# Attentive-perk antenna raise (delta over the rest pose) while the user is speaking.
# Rest is ~10 deg, so this settles near 32 deg, well inside the [3, 45] antenna clamp.
LISTEN_ANTENNA_RAISE = 22.0
# RMS floor above which feed_audio_rms counts as Rocky speaking (idle-timer activity).
IDLE_RMS_FLOOR = 0.01
# How long a face gaze stays authoritative after its last update before DoA resumes.
GAZE_FRESH_S = 1.5
# RMS floor above which Rocky counts as speaking (freezes the gaze anchor, decision 18).
SPEAK_RMS_FLOOR = IDLE_RMS_FLOOR
# How long the speaking flag holds after the last loud frame (mixer playout tail).
SPEAK_HOLD_S = 0.3


class MovementManager:
    def __init__(
        self,
        blend_tau: float = 0.4,
        clock: Callable[[], float] = time.perf_counter,
        idle_glance_s: float = 180.0,
        rng=None,
    ) -> None:
        self.blend_tau = blend_tau
        self.clock = clock
        self.current_primary = Pose()
        self.sleep_watch = False
        self.listening = False
        self.doa_deg: float | None = None
        self.wobble = WobbleTracker()
        self.idle = IdlePolicy(idle_glance_s, rng)
        self._primary_emote: tuple[Trajectory, float] | None = None  # (traj, start_t)
        self._reflex: tuple[str, float] | None = None  # (name, start_t)
        # Client-side face gaze (decision 18): a (yaw_deg, pitch_deg) look offset fed
        # by the VisionThread through set_gaze. A fresh gaze wins over DoA; while Rocky
        # speaks the anchor is frozen so his gaze holds on the listener as he talks.
        self._gaze: tuple[float, float] | None = None
        self._gaze_t: float = -1e9
        self.speaking = False
        self._speaking_until = -1e9

    # -- inputs ------------------------------------------------------------
    def start_emote(self, name: str, t: float | None = None) -> None:
        t = self.clock() if t is None else t
        if name in REFLEX_EMOTES:
            self._reflex = (name, t)
        elif name in PRIMARY_EMOTES:
            self._primary_emote = (PRIMARY_EMOTES[name], t)
        self.idle.note_activity(self.clock())

    def set_doa(self, deg: float | None) -> None:
        self.doa_deg = deg
        if deg is not None:
            self.idle.note_activity(self.clock())

    def set_gaze(self, yaw_deg: float | None, pitch_deg: float | None) -> None:
        """Feed a face-gaze look offset from the VisionThread (footgun 1: only setter).

        yaw_deg None clears the anchor, but NOT while Rocky is speaking: the anchor is
        frozen during speech so his gaze holds on the listener even through detection
        dropouts. A live value is likewise ignored while speaking; otherwise it becomes
        the fresh anchor and counts as activity for the idle timer.
        """
        if yaw_deg is None:
            if not self.speaking:
                self._gaze = None
            return
        if self.speaking:
            return
        self._gaze = (yaw_deg, pitch_deg or 0.0)
        self._gaze_t = self.clock()
        self.idle.note_activity(self.clock())

    def set_sleep_watch(self, on: bool) -> None:
        self.sleep_watch = on
        self.idle.note_activity(self.clock())

    def set_listening(self, on: bool) -> None:
        self.listening = bool(on)
        if on:
            self.idle.note_activity(self.clock())

    def feed_audio_rms(self, rms: float, t: float | None = None) -> None:
        now = self.clock() if t is None else t
        self.wobble.feed(rms, now)
        if rms > IDLE_RMS_FLOOR:
            self.idle.note_activity(self.clock())
        if rms > SPEAK_RMS_FLOOR:
            # Rocky is producing sound: freeze the gaze anchor for a short tail.
            self.speaking = True
            self._speaking_until = now + SPEAK_HOLD_S

    def _expire_speaking(self) -> None:
        """Drop the speaking flag once the hold window has elapsed (injected clock)."""
        if self.speaking and self.clock() > self._speaking_until:
            self.speaking = False

    # -- composition -------------------------------------------------------
    def _primary_target(self, t: float) -> Pose:
        base = Pose()
        idle = sleepwatch_delta(t, doa_deg=self.doa_deg) if self.sleep_watch else breathing_delta(t)
        if self.listening:
            # Attentive perk: freeze the breathing antenna sway and raise both antennas
            # to a fixed attentive pose while the user speaks. The head components are
            # left untouched so Rocky still breathes/sweeps and looks alive. The existing
            # current_primary lerp (blend_tau) gives the blend-up and blend-back for free.
            idle["antenna_left"] = LISTEN_ANTENNA_RAISE
            idle["antenna_right"] = LISTEN_ANTENNA_RAISE
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
        self._expire_speaking()
        deltas: list[dict] = [self.wobble.delta(t)]
        if self._reflex is not None:
            name, start = self._reflex
            since = t - start
            reflex_delta = jazz_hands_delta(since) if name == "jazz_hands" else {}
            if not reflex_delta:
                self._reflex = None
            else:
                deltas.append(reflex_delta)
        if not self.sleep_watch:
            # Gaze/DoA reconciliation (decision 18): a fresh face wins over DoA. While
            # Rocky speaks the anchor is always fresh (frozen); otherwise it stays fresh
            # for GAZE_FRESH_S after its last update. With no fresh gaze, fall back to DoA.
            gaze_fresh = self._gaze is not None and (
                self.speaking or (self.clock() - self._gaze_t) < GAZE_FRESH_S
            )
            if gaze_fresh:
                gaze_yaw, gaze_pitch = self._gaze
                deltas.append(
                    {
                        "head_yaw": 0.6 * gaze_yaw,
                        "body_yaw": 0.2 * gaze_yaw,
                        "head_pitch": 0.5 * gaze_pitch,
                    }
                )
            elif self.doa_deg is not None:
                # Head leads toward the sound; body follows more slowly (yaw clamp).
                deltas.append({"head_yaw": 0.6 * self.doa_deg, "body_yaw": 0.2 * self.doa_deg})
        return deltas

    def tick(self, t: float, dt: float) -> Pose:
        target = self._primary_target(t)
        self.current_primary = lerp_pose(self.current_primary, target, blend_factor(dt, self.blend_tau))
        final = apply_deltas(self.current_primary, self._secondary_deltas(t))
        # Rare idle liveliness lives in the motion domain: it must fire during true
        # silence, when the conversation loop is not running. Only consult it when Rocky
        # is idle (not sleep-watching, no primary emote in flight); a started emote takes
        # effect on the following ticks, so this tick's returned pose is unchanged.
        if not self.sleep_watch and self._primary_emote is None:
            action = self.idle.maybe_act(t)
            if action is not None:
                self.start_emote(action, t)
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
