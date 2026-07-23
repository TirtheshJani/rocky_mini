"""Vision tests (Phase 5): gaze mapping, VisionThread step wiring, and the
gaze/DoA + speaking-anchor reconciliation in MovementManager.

Sim-only: these never import cv2 or the reachy_mini SDK. FakeCamera/FakeDetector carry
the whole path, mirroring the AudioIO Fakes.
"""

import pytest

from rocky_mini.motion.manager import GAZE_FRESH_S, SPEAK_HOLD_S, MovementManager
from rocky_mini.vision import FakeCamera, FakeDetector, VisionThread, gaze_from_uv


class _Clock:
    """Controllable clock injected into MovementManager for deterministic tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _RecordingMotion:
    """Minimal motion stand-in: records every set_gaze call (footgun 1 surface)."""

    def __init__(self) -> None:
        self.gazes: list[tuple[float | None, float | None]] = []

    def set_gaze(self, yaw, pitch) -> None:
        self.gazes.append((yaw, pitch))


# -- gaze_from_uv ----------------------------------------------------------
def test_gaze_from_uv_center_is_zero():
    yaw, pitch = gaze_from_uv(0.5, 0.5, 70.0, 42.0)
    assert yaw == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)


def test_gaze_from_uv_right_edge_positive_half_fov():
    yaw, pitch = gaze_from_uv(1.0, 0.5, 70.0, 42.0)
    assert yaw == pytest.approx(35.0)  # +h_fov / 2
    assert pitch == pytest.approx(0.0)


def test_gaze_from_uv_top_edge_negative_half_fov():
    yaw, pitch = gaze_from_uv(0.5, 0.0, 70.0, 42.0)
    assert yaw == pytest.approx(0.0)
    assert pitch == pytest.approx(-21.0)  # -v_fov / 2 (face high -> look up)


# -- VisionThread.step wiring ----------------------------------------------
def test_vision_thread_step_feeds_gaze_and_captures_jpeg():
    motion = _RecordingMotion()
    cam = FakeCamera()
    vt = VisionThread(cam, FakeDetector([(0.8, 0.5)]), motion, h_fov_deg=70.0, v_fov_deg=42.0)

    assert vt.get_jpeg() is None  # nothing captured before the first step
    vt.step()

    assert len(motion.gazes) == 1
    yaw, pitch = motion.gazes[0]
    assert yaw > 0  # u = 0.8 is right of center -> positive yaw
    assert yaw == pytest.approx(21.0)
    assert vt.get_jpeg() == cam.get_frame_jpeg()  # single-slot jpeg populated
    assert vt.latest_jpeg is not None


def test_vision_thread_step_clears_gaze_when_no_face():
    motion = _RecordingMotion()
    vt = VisionThread(FakeCamera(), FakeDetector([None]), motion)
    vt.step()
    assert motion.gazes == [(None, None)]


# -- gaze wins over DoA, then DoA resumes ----------------------------------
def test_vision_gaze_overrides_doa_and_doa_resumes():
    clock = _Clock()
    mgr = MovementManager(clock=clock)
    detector = FakeDetector([(0.2, 0.5), (0.8, 0.5), None])
    vt = VisionThread(FakeCamera(), detector, mgr, h_fov_deg=70.0, v_fov_deg=42.0)

    # Sound is set to the negative side; a fresh face must override it while fresh.
    mgr.set_doa(-40.0)

    vt.step()  # face 1: u = 0.2 -> yaw negative (consume, no assertion)
    vt.step()  # face 2: u = 0.8 -> yaw = +21 (positive)
    pose = mgr.tick(clock.t, dt=0.01)
    assert pose.head_yaw > 0  # follows the face (right), NOT the DoA (which is negative)

    vt.step()  # face 3: None -> gaze cleared (not speaking)
    clock.t += GAZE_FRESH_S + 0.5  # well past the freshness window
    pose_doa = mgr.tick(clock.t, dt=0.01)
    assert pose_doa.head_yaw < 0  # DoA (-40) resumes once gaze is gone/stale


# -- speaking anchor freezes gaze ------------------------------------------
def test_speaking_anchor_freezes_then_resumes():
    clock = _Clock()
    mgr = MovementManager(clock=clock)

    # Anchor gaze to the right; it steers the head positive.
    mgr.set_gaze(20.0, 0.0)
    pose_anchor = mgr.tick(clock.t, dt=0.01)
    assert pose_anchor.head_yaw > 0

    # Rocky starts talking: a loud frame sets the speaking flag.
    mgr.feed_audio_rms(0.5, t=clock.t)
    assert mgr.speaking is True

    # A live update to the opposite side is ignored while speaking (anchor frozen).
    mgr.set_gaze(-20.0, 0.0)
    pose_speaking = mgr.tick(clock.t, dt=0.01)
    assert pose_speaking.head_yaw > 0  # still anchored to +20, not -20

    # After SPEAK_HOLD_S of quiet, the next tick expires the flag; gaze updates resume.
    clock.t += SPEAK_HOLD_S + 0.1
    mgr.feed_audio_rms(0.0, t=clock.t)  # quiet: does not refresh the speaking window
    mgr.tick(clock.t, dt=0.01)  # this tick clears the speaking flag
    assert mgr.speaking is False

    mgr.set_gaze(-20.0, 0.0)  # now accepted
    pose_after = mgr.tick(clock.t, dt=0.01)
    assert pose_after.head_yaw < 0  # follows the new gaze to the left
