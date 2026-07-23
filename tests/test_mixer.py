"""Mixer tests: generation flush (barge-in), frame pushing, chord underlay."""

import numpy as np

from rocky_mini.audio.io import FakeAudioIO
from rocky_mini.audio.output import Mixer


def make_mixer(ring=False):
    io = FakeAudioIO()
    return io, Mixer(io, frame=320, ring_mod_enabled=ring)


def test_render_pushes_frames_of_correct_size():
    io, mixer = make_mixer()
    mixer.push_voice(np.ones(640, dtype=np.float32) * 0.2, generation=0)
    mixer.render()
    assert len(io.pushed) == 2
    assert all(len(f) == 320 for f in io.pushed)


def test_generation_flush_drops_stale_voice():
    io, mixer = make_mixer()
    # Old speech queued under generation 0.
    mixer.push_voice(np.ones(320, dtype=np.float32) * 0.5, generation=0)
    # Barge-in: user interrupts, generation advances.
    mixer.set_generation(1)
    # New speech under generation 1.
    mixer.push_voice(np.ones(320, dtype=np.float32) * 0.1, generation=1)

    out = mixer.render()
    assert mixer.dropped_generations == 1
    # Only the new (quieter) chunk survives; old 0.5 chunk never played.
    assert len(out) == 320
    assert np.max(np.abs(out)) < 0.2


def test_current_generation_voice_survives():
    io, mixer = make_mixer()
    mixer.set_generation(3)
    mixer.push_voice(np.ones(320, dtype=np.float32) * 0.3, generation=3)
    out = mixer.render()
    assert mixer.dropped_generations == 0
    assert len(out) == 320


def test_voice_and_chord_mix_and_clip():
    io, mixer = make_mixer()
    mixer.push_voice(np.ones(320, dtype=np.float32) * 0.9, generation=0)
    mixer.push_chord(np.ones(320, dtype=np.float32) * 0.9)
    out = mixer.render()
    # Sum would be 1.8 but soft-clip keeps it below 1.0.
    assert np.all(np.abs(out) < 1.0)
    assert np.max(out) > 0.7  # still loud, just clipped


def test_render_empty_is_noop():
    io, mixer = make_mixer()
    out = mixer.render()
    assert len(out) == 0
    assert io.pushed == []


def test_ring_mod_path_runs():
    io, mixer = make_mixer(ring=True)
    sr = mixer.sample_rate
    t = np.arange(sr // 10, dtype=np.float64) / sr
    tone = np.sin(2 * np.pi * 300 * t).astype(np.float32) * 0.5
    mixer.push_voice(tone, generation=0)
    out = mixer.render()
    assert len(out) == len(tone)
    # Ring modulation changed the signal.
    assert not np.allclose(out, tone)


def test_ring_mix_blends_wet_and_dry():
    import numpy as np

    from rocky_mini.audio.io import FakeAudioIO
    from rocky_mini.audio.output import Mixer

    pcm = np.sin(2 * np.pi * 440 * np.arange(2205) / 22050).astype(np.float32) * 0.5
    outs = {}
    for mix in (0.0, 0.2, 1.0):
        io = FakeAudioIO()
        m = Mixer(io, frame=320, sample_rate=22050, ring_mix=mix)
        m.push_voice(pcm.copy(), generation=0)
        outs[mix] = m.render()
    # mix=0.0 passes the voice through untouched; mix=1.0 is the old full ring mod;
    # mix=0.2 must be the exact linear blend of the two (soft clip is monotonic but
    # near-linear at this level, so compare pre-clip identity via the blend of outputs
    # being closer to dry than wet).
    assert np.allclose(outs[0.0], np.tanh(pcm), atol=1e-6)
    assert not np.allclose(outs[1.0], outs[0.0])
    d_dry = float(np.mean(np.abs(outs[0.2] - outs[0.0])))
    d_wet = float(np.mean(np.abs(outs[0.2] - outs[1.0])))
    assert d_dry < d_wet
