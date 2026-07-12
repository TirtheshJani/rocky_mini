"""DSP tests: ring-mod sidebands, carrier phase continuity, clipping, resample."""

import numpy as np

from rocky_mini.audio import dsp


def _dominant_freqs(signal: np.ndarray, sr: int, n: int = 3) -> list[float]:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    top = np.argsort(spectrum)[::-1][:n]
    return sorted(round(freqs[i]) for i in top)


def test_ring_mod_creates_sidebands_and_suppresses_carrier():
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr  # 1 second
    f0, fc = 3000.0, 1000.0
    x = np.sin(2 * np.pi * f0 * t).astype(np.float32)

    out, _ = dsp.ring_mod(x, fc, sr)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1.0 / sr)

    def mag_at(f):
        return spectrum[np.argmin(np.abs(freqs - f))]

    lower, upper, original = mag_at(f0 - fc), mag_at(f0 + fc), mag_at(f0)
    # Energy moves to the two sidebands; the original tone is suppressed.
    assert lower > 100 * original
    assert upper > 100 * original
    assert {2000, 4000}.issubset(set(_dominant_freqs(out, sr, n=2)))


def test_ring_mod_phase_is_continuous_across_chunks():
    sr = 16000
    x = np.ones(1000, dtype=np.float32)
    whole, _ = dsp.ring_mod(x, 140.0, sr)

    first, phase = dsp.ring_mod(x[:400], 140.0, sr)
    second, _ = dsp.ring_mod(x[400:], 140.0, sr, phase0=phase)
    chunked = np.concatenate([first, second])

    assert np.allclose(whole, chunked, atol=1e-5)


def test_soft_clip_bounds_output():
    x = np.linspace(-10, 10, 1000).astype(np.float32)
    y = dsp.soft_clip(x)
    # tanh saturates at +/-1 (float32 rounds tanh(10) to exactly 1.0), so bound is inclusive.
    assert np.all(y <= 1.0) and np.all(y >= -1.0)
    # A mid-range input stays strictly inside the bound.
    mid = dsp.soft_clip(np.array([0.5, -0.5], dtype=np.float32))
    assert np.all(np.abs(mid) < 1.0)
    # Monotonic non-decreasing.
    assert np.all(np.diff(y) >= -1e-6)


def test_resample_changes_length_proportionally():
    x = np.random.default_rng(0).standard_normal(22050).astype(np.float32)
    y = dsp.resample(x, 22050, 16000)
    assert abs(len(y) - 16000) <= 2


def test_resample_noop_same_rate():
    x = np.arange(10, dtype=np.float32)
    assert np.array_equal(dsp.resample(x, 16000, 16000), x)


def test_rms_envelope_matches_manual():
    sig = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float32)
    env = dsp.rms_envelope(sig, frame=4)
    assert np.allclose(env, [1.0, 2.0])


def test_normalize_peak_and_silence():
    x = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    y = dsp.normalize_peak(x, 1.0)
    assert np.isclose(np.max(np.abs(y)), 1.0)
    silence = np.zeros(5, dtype=np.float32)
    assert np.array_equal(dsp.normalize_peak(silence), silence)
