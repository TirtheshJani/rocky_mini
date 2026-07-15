"""Voice-in tests (Phase 1b): ring buffer, VAD gate, hangover, DoA poll.

Everything runs against FakeAudioIO and a scripted FakeVAD with an injected clock:
no robot, no onnxruntime model, no threads (the thread loop is exercised via step()).
The 0.6 s hangover and the last-voiced-frame timestamp are footgun 7's latency anchor.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocky_mini.audio.input import AudioInThread, RingBuffer, SpeechSegment, VadGate
from rocky_mini.audio.io import FakeAudioIO
from rocky_mini.audio.vad import FakeVAD
from rocky_mini.motion.manager import MovementManager

RATE = 16000
FRAME = 512  # silero's 16 kHz frame
FRAME_S = FRAME / RATE


def _voiced(n=FRAME):
    return np.ones(n, dtype=np.float32) * 0.3


def _silence(n=FRAME):
    return np.zeros(n, dtype=np.float32)


# -- RingBuffer ---------------------------------------------------------------

def test_ring_buffer_returns_most_recent():
    rb = RingBuffer(seconds=1.0, rate=RATE)
    rb.append(np.full(RATE, 0.1, dtype=np.float32))
    rb.append(np.full(FRAME, 0.5, dtype=np.float32))
    tail = rb.last(FRAME)
    assert tail.shape == (FRAME,)
    assert np.all(tail == pytest.approx(0.5))


def test_ring_buffer_wraps_without_growing():
    rb = RingBuffer(seconds=0.1, rate=RATE)  # 1600 samples capacity
    for _ in range(100):
        rb.append(np.ones(FRAME, dtype=np.float32))
    assert rb.last(RATE).shape[0] <= 1600


# -- VadGate state machine ----------------------------------------------------

def _gate(**kw):
    defaults = dict(rate=RATE, hangover_s=0.6, preroll_s=0.0, min_speech_s=0.0)
    defaults.update(kw)
    return VadGate(**defaults)


def _run_frames(gate, vad, frames, t0=0.0):
    events = []
    t = t0
    for f in frames:
        events += gate.feed(f, vad.probability(f), t)
        t += FRAME_S
    return events, t


def test_speech_end_fires_only_after_hangover():
    gate = _gate()
    vad = FakeVAD()
    # 10 voiced frames (~0.32 s), then silence
    events, t = _run_frames(gate, vad, [_voiced()] * 10)
    assert [e for e in events if isinstance(e, SpeechSegment)] == []
    # 0.5 s of silence: still inside the 0.6 s hangover
    events2, t = _run_frames(gate, vad, [_silence()] * 15, t0=t)
    assert [e for e in events2 if isinstance(e, SpeechSegment)] == []
    # past the hangover: exactly one utterance
    events3, _ = _run_frames(gate, vad, [_silence()] * 10, t0=t)
    segs = [e for e in events3 if isinstance(e, SpeechSegment)]
    assert len(segs) == 1


def test_segment_timestamps_anchor_last_voiced_frame():
    gate = _gate()
    vad = FakeVAD()
    events, t_after_voice = _run_frames(gate, vad, [_voiced()] * 10)
    events2, _ = _run_frames(gate, vad, [_silence()] * 25, t0=t_after_voice)
    seg = [e for e in events2 if isinstance(e, SpeechSegment)][0]
    # last voiced frame was fed at t_after_voice - FRAME_S
    assert seg.t_last_voiced == pytest.approx(t_after_voice - FRAME_S, abs=1e-9)
    # honest latency anchor (footgun 7): the segment closes hangover_s later
    assert seg.t_end == pytest.approx(seg.t_last_voiced + 0.6, abs=FRAME_S)


def test_short_pause_does_not_split_utterance():
    gate = _gate()
    vad = FakeVAD()
    frames = [_voiced()] * 5 + [_silence()] * 10 + [_voiced()] * 5  # 0.32 s pause < 0.6 s
    events, t = _run_frames(gate, vad, frames)
    assert [e for e in events if isinstance(e, SpeechSegment)] == []
    events2, _ = _run_frames(gate, vad, [_silence()] * 25, t0=t)
    segs = [e for e in events2 if isinstance(e, SpeechSegment)]
    assert len(segs) == 1
    # one utterance spanning both bursts, pause included
    assert seg_samples(segs[0]) == pytest.approx(20 * FRAME, abs=FRAME)


def seg_samples(seg: SpeechSegment) -> int:
    return len(seg.pcm)


def test_min_speech_filters_clicks():
    gate = _gate(min_speech_s=0.2)
    vad = FakeVAD()
    frames = [_voiced()] * 2 + [_silence()] * 25  # 64 ms of "speech": a click
    events, _ = _run_frames(gate, vad, frames)
    assert [e for e in events if isinstance(e, SpeechSegment)] == []


def test_preroll_included_in_segment():
    gate = _gate(preroll_s=0.096)  # 3 frames
    vad = FakeVAD()
    frames = [_silence()] * 5 + [_voiced()] * 10 + [_silence()] * 25
    events, _ = _run_frames(gate, vad, frames)
    seg = [e for e in events if isinstance(e, SpeechSegment)][0]
    assert len(seg.pcm) >= 13 * FRAME  # 10 voiced + 3 preroll


# -- AudioInThread: rebuffering, events, DoA poll, half-duplex ----------------

class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _thread(io, motion=None, on_speech=None, is_speaking=None, **kw):
    return AudioInThread(
        io=io,
        vad=FakeVAD(),
        motion=motion or MovementManager(),
        on_speech=on_speech or (lambda seg: None),
        is_speaking=is_speaking or (lambda: False),
        clock=kw.pop("clock", _Clock()),
        **kw,
    )


def test_thread_rebuffers_odd_block_sizes():
    """io blocks of 320 samples must still reach the VAD as 512-sample frames."""
    io = FakeAudioIO()
    segs: list[SpeechSegment] = []
    clock = _Clock()
    th = _thread(io, on_speech=segs.append, clock=clock)
    for _ in range(60):  # ~1.2 s of voiced audio in 320-sample blocks
        io.feed_input(_voiced(320))
    for _ in range(120):
        th.step()
        clock.t += 0.02
    for _ in range(80):  # silence past the hangover
        io.feed_input(_silence(320))
        th.step()
        clock.t += 0.02
    assert len(segs) == 1
    assert len(segs[0].pcm) >= RATE  # at least the ~1.2 s utterance


def test_doa_reading_updates_motion_when_speech():
    io = FakeAudioIO()
    motion = MovementManager()
    clock = _Clock()
    th = _thread(io, motion=motion, clock=clock)
    io.set_DoA(25.0, speech=True)
    # get into speech state first (AND-gate needs our VAD in speech too)
    for _ in range(6):
        io.feed_input(_voiced())
        th.step()
        clock.t += FRAME_S
    assert motion.doa_deg == pytest.approx(25.0)


def test_doa_ignored_without_hardware_speech_flag():
    io = FakeAudioIO()
    motion = MovementManager()
    clock = _Clock()
    th = _thread(io, motion=motion, clock=clock)
    io.set_DoA(25.0, speech=False)
    for _ in range(6):
        io.feed_input(_voiced())
        th.step()
        clock.t += FRAME_S
    assert motion.doa_deg is None


def test_doa_ignored_when_vad_silent():
    io = FakeAudioIO()
    motion = MovementManager()
    clock = _Clock()
    th = _thread(io, motion=motion, clock=clock)
    io.set_DoA(25.0, speech=True)
    for _ in range(6):
        io.feed_input(_silence())
        th.step()
        clock.t += FRAME_S
    assert motion.doa_deg is None


def test_doa_reading_reaches_the_pose():
    """TJ's whole-path check: a DoA reading must change what set_target would send."""
    io = FakeAudioIO()
    motion = MovementManager()
    clock = _Clock()
    th = _thread(io, motion=motion, clock=clock)
    io.set_DoA(25.0, speech=True)
    for _ in range(6):
        io.feed_input(_voiced())
        th.step()
        clock.t += FRAME_S
    pose = motion.tick(t=0.0, dt=0.01)
    # DoA orient: head leads 0.6 * 25 = 15 deg, body follows 0.2 * 25 = 5 deg
    assert pose.head_yaw == pytest.approx(15.0, abs=1.0)
    assert pose.body_yaw == pytest.approx(5.0, abs=1.0)


def test_half_duplex_suppresses_events_while_speaking():
    io = FakeAudioIO()
    segs: list[SpeechSegment] = []
    motion = MovementManager()
    clock = _Clock()
    th = _thread(
        io, motion=motion, on_speech=segs.append,
        is_speaking=lambda: True, half_duplex=True, clock=clock,
    )
    io.set_DoA(25.0, speech=True)
    for _ in range(40):
        io.feed_input(_voiced())
        th.step()
        clock.t += FRAME_S
    for _ in range(40):
        io.feed_input(_silence())
        th.step()
        clock.t += FRAME_S
    assert segs == []  # Rocky's own voice must not become a turn
    assert motion.doa_deg is None  # and must not drag his gaze around


def test_silero_vad_runs_the_vendored_onnx_model():
    """Structural check only: the model loads and emits sane probabilities.

    Real speech discrimination is validated at bring-up with a live mic; synthetic
    PCM cannot honestly test it (Silero scores synthetic tones near zero).
    """
    pytest.importorskip("onnxruntime")
    from rocky_mini.audio.vad import MODEL_PATH, SileroVAD

    if not MODEL_PATH.exists():
        pytest.skip("vendored silero model not present")
    vad = SileroVAD()
    p = vad.probability(np.zeros(512, dtype=np.float32))
    assert 0.0 <= p <= 1.0
    assert p < 0.3, "silence must not read as speech"
    vad.reset()
    assert vad.probability(np.zeros(512, dtype=np.float32)) == pytest.approx(p, abs=1e-6)


def test_open_mic_emits_barge_in_while_speaking():
    io = FakeAudioIO()
    barged = []
    clock = _Clock()
    th = _thread(
        io, on_speech=lambda s: None, is_speaking=lambda: True,
        half_duplex=False, on_barge_in=lambda: barged.append(True), clock=clock,
    )
    for _ in range(8):
        io.feed_input(_voiced())
        th.step()
        clock.t += FRAME_S
    assert barged, "open-mic mode: speech during playback must fire barge-in"
