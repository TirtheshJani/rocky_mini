"""Hardware seam tests (audit findings F1-F8).

Everything here runs without the SDK: the adapter math is pure, and ReachyMediaIO is
exercised against an in-memory fake of the SDK's mini.media surface. The contracts
being tested (signatures, units, rates, pipeline calls) come from reachy-mini 1.9.0,
see docs/hardware-audit.md.
"""

from __future__ import annotations

import inspect
import math
import threading

import numpy as np
import pytest

from rocky_mini.audio.io import FakeAudioIO, ReachyMediaIO
from rocky_mini.audio.output import Mixer
from rocky_mini.motion.manager import MovementManager
from rocky_mini.motion.pose import ANTENNA_REST, Pose, sdk_targets


# -- F1: run() must accept the daemon's (reachy_mini, stop_event) call ------

def test_run_signature_matches_daemon_contract():
    from rocky_mini.main import RockyMiniApp

    params = list(inspect.signature(RockyMiniApp.run).parameters)
    assert params[1] == "reachy_mini"
    assert params[2] == "stop_event"


def test_module_has_daemon_entrypoint():
    import rocky_mini.main as m

    assert callable(getattr(m, "main", None))


# -- F4: Pose -> SDK set_target arguments ------------------------------------

def test_sdk_targets_neutral_pose():
    head, antennas, body_yaw = sdk_targets(Pose())
    assert head.shape == (4, 4)
    np.testing.assert_allclose(head[:3, :3], np.eye(3), atol=1e-9)
    # antennas are [right, left] in radians; rest is ~10 deg, never 0 (footgun 6)
    assert antennas == pytest.approx(
        [math.radians(ANTENNA_REST), math.radians(ANTENNA_REST)]
    )
    assert body_yaw == 0.0


def test_sdk_targets_yaw_rotation_and_z():
    pose = Pose(head_yaw=90.0, head_z=0.008)
    head, _, _ = sdk_targets(pose)
    # yaw 90 deg about z: x-axis maps to y-axis
    np.testing.assert_allclose(head[:3, 0], [0.0, 1.0, 0.0], atol=1e-9)
    # head_z is meters and lands in the translation column
    assert head[2, 3] == pytest.approx(0.008)


def test_sdk_targets_antenna_order_and_units():
    pose = Pose(antenna_right=20.0, antenna_left=5.0)
    _, antennas, _ = sdk_targets(pose)
    assert antennas[0] == pytest.approx(math.radians(20.0))  # right first
    assert antennas[1] == pytest.approx(math.radians(5.0))


def test_sdk_targets_body_yaw_radians():
    _, _, body_yaw = sdk_targets(Pose(body_yaw=30.0))
    assert body_yaw == pytest.approx(math.radians(30.0))


# -- F4: the one set_target call site fails loudly, not silently -------------

class _RecordingMini:
    """Stands in for a live ReachyMini handle."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def set_target(self, head=None, antennas=None, body_yaw=None) -> None:
        assert head is not None and head.shape == (4, 4)
        self.calls.append({"head": head, "antennas": antennas, "body_yaw": body_yaw})


def test_manager_set_target_uses_sdk_call_form():
    mgr = MovementManager()
    mini = _RecordingMini()
    mgr._set_target(mini, Pose())
    call = mini.calls[0]
    assert call["antennas"] == pytest.approx([math.radians(ANTENNA_REST)] * 2)


def test_manager_rejects_handle_without_set_target():
    mgr = MovementManager()
    with pytest.raises((AttributeError, RuntimeError)):
        mgr._set_target(object(), Pose())


def test_manager_ramp_to_neutral_never_zeroes_antennas():
    mgr = MovementManager()
    mgr.current_primary = Pose(head_yaw=30.0, antenna_left=40.0, antenna_right=40.0)
    mini = _RecordingMini()
    stop = threading.Event()
    stop.set()  # run() must still ramp to neutral after stop
    mgr.run(mini, stop, hz=100.0, ramp_s=0.05)
    assert mini.calls, "ramp must issue set_target calls after stop"
    for call in mini.calls:
        for a in call["antennas"]:
            assert a >= math.radians(1.0), "antennas commanded to ~0 (footgun 6)"
    final = mini.calls[-1]
    assert final["antennas"] == pytest.approx([math.radians(ANTENNA_REST)] * 2, abs=0.02)


# -- F5/F6/F7/F8: ReachyMediaIO against the SDK 1.9.0 media surface ----------

class _FakeAudio:
    def __init__(self) -> None:
        self.cleared = 0

    def clear_player(self) -> None:
        self.cleared += 1


class _FakeMedia:
    """Mimics reachy_mini.media.media_manager.MediaManager (1.9.0)."""

    def __init__(self, doa=None) -> None:
        self.audio = _FakeAudio()
        self.recording = False
        self.playing = False
        self.pushed: list[np.ndarray] = []
        self._in: list = []
        self._doa = doa

    def start_recording(self) -> None:
        self.recording = True

    def start_playing(self) -> None:
        self.playing = True

    def stop_recording(self) -> None:
        self.recording = False

    def stop_playing(self) -> None:
        self.playing = False

    def get_audio_sample(self):
        return self._in.pop(0) if self._in else None

    def push_audio_sample(self, data) -> None:
        self.pushed.append(np.asarray(data))

    def get_input_audio_samplerate(self) -> int:
        return 16000

    def get_output_audio_samplerate(self) -> int:
        return 16000

    def get_DoA(self):
        return self._doa


class _FakeMini:
    def __init__(self, media) -> None:
        self.media = media


def test_reachy_media_io_starts_pipelines_on_init():
    media = _FakeMedia()
    ReachyMediaIO(_FakeMini(media))
    assert media.recording and media.playing


def test_reachy_media_io_validates_surface_loudly():
    class NoMedia:
        pass

    with pytest.raises(RuntimeError, match="media"):
        ReachyMediaIO(NoMedia())


def test_reachy_media_io_none_sample_is_empty_frame():
    io = ReachyMediaIO(_FakeMini(_FakeMedia()))
    out = io.get_audio_sample()
    assert out.size == 0 and out.dtype == np.float32


def test_reachy_media_io_downmixes_stereo():
    media = _FakeMedia()
    media._in.append(np.ones((160, 2), dtype=np.float32) * np.array([0.2, 0.4]))
    io = ReachyMediaIO(_FakeMini(media))
    out = io.get_audio_sample()
    assert out.shape == (160,)
    assert out[0] == pytest.approx(0.3)


def test_reachy_media_io_doa_maps_radians_to_signed_degrees():
    # SDK convention (1.9.0 docstring): 0 rad = left, pi/2 = front, pi = right.
    io = ReachyMediaIO(_FakeMini(_FakeMedia(doa=(math.pi / 2, True))))
    reading = io.get_DoA()
    assert reading is not None
    deg, speech = reading
    assert deg == pytest.approx(0.0)  # straight ahead
    assert speech is True

    io2 = ReachyMediaIO(_FakeMini(_FakeMedia(doa=(0.0, False))))
    deg2, speech2 = io2.get_DoA()
    assert deg2 == pytest.approx(90.0)  # left of the robot
    assert speech2 is False


def test_reachy_media_io_doa_none_passthrough():
    io = ReachyMediaIO(_FakeMini(_FakeMedia(doa=None)))
    assert io.get_DoA() is None


def test_reachy_media_io_flush_calls_clear_player():
    media = _FakeMedia()
    io = ReachyMediaIO(_FakeMini(media))
    io.flush()
    assert media.audio.cleared == 1


def test_reachy_media_io_reports_output_rate():
    io = ReachyMediaIO(_FakeMini(_FakeMedia()))
    assert io.output_sample_rate == 16000


# -- F5: voice resampling at the Mixer boundary ------------------------------

def test_mixer_resamples_voice_to_output_rate():
    io = FakeAudioIO()
    mixer = Mixer(io, sample_rate=16000, ring_mod_enabled=False)
    one_second = np.ones(22050, dtype=np.float32) * 0.1
    mixer.push_voice(one_second, generation=0, src_rate=22050)
    mixed = mixer.render()
    assert len(mixed) == pytest.approx(16000, rel=0.01)


def test_mixer_no_resample_when_rates_match():
    io = FakeAudioIO()
    mixer = Mixer(io, sample_rate=16000, ring_mod_enabled=False)
    pcm = np.ones(1600, dtype=np.float32) * 0.1
    mixer.push_voice(pcm, generation=0, src_rate=16000)
    assert len(mixer.render()) == 1600


# -- F8: barge-in reaches the device, not just the queue ----------------------

def test_generation_bump_flushes_device():
    io = FakeAudioIO()
    mixer = Mixer(io, sample_rate=16000)
    mixer.push_voice(np.ones(320, dtype=np.float32), generation=0)
    mixer.set_generation(1)
    assert io.flushed == 1
    assert len(mixer.render()) == 0  # stale voice dropped too


def test_set_generation_same_value_does_not_flush():
    io = FakeAudioIO()
    mixer = Mixer(io, sample_rate=16000)
    mixer.set_generation(0)
    assert io.flushed == 0
