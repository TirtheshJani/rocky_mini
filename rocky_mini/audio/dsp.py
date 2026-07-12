"""Pure DSP functions for Rocky's Eridian voice.

Everything here is a pure function: no state, no I/O, deterministic. That makes the
FFT-sideband and phase-continuity tests trivial and keeps the Mixer thread simple.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

TWO_PI = 2.0 * np.pi


def ring_mod(
    signal: np.ndarray,
    carrier_hz: float,
    sample_rate: int,
    phase0: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Ring-modulate a signal with a cosine carrier.

    Multiplying by a carrier at fc shifts each input component f to sidebands at
    |f - fc| and f + fc and suppresses the original f. That metallic, inharmonic
    result is the core of Rocky's alien timbre.

    Phase is carried across calls: pass the returned phase back in as phase0 for the
    next chunk so the carrier is continuous (no clicks at chunk boundaries).

    Returns (modulated_signal, next_phase).
    """
    n = signal.shape[0]
    if n == 0:
        return signal.astype(np.float32, copy=False), phase0
    step = TWO_PI * carrier_hz / sample_rate
    phase = phase0 + step * np.arange(n, dtype=np.float64)
    carrier = np.cos(phase)
    out = (signal.astype(np.float64) * carrier).astype(np.float32)
    next_phase = float((phase0 + step * n) % TWO_PI)
    return out, next_phase


def soft_clip(signal: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Tanh soft clipper. Bounds output to (-1, 1) without hard-edge distortion."""
    return np.tanh(drive * signal.astype(np.float64)).astype(np.float32)


def resample(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resample from src_rate to dst_rate (e.g. 22050 -> 16000)."""
    if src_rate == dst_rate:
        return signal.astype(np.float32, copy=False)
    g = np.gcd(src_rate, dst_rate)
    up = dst_rate // g
    down = src_rate // g
    return resample_poly(signal, up, down).astype(np.float32)


def rms_envelope(signal: np.ndarray, frame: int) -> np.ndarray:
    """Per-frame RMS envelope, one value per non-overlapping frame.

    Drives the head wobble amplitude. Delayed to the playout clock by the caller.
    """
    if frame <= 0:
        raise ValueError("frame must be positive")
    n_frames = len(signal) // frame
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)
    trimmed = signal[: n_frames * frame].astype(np.float64).reshape(n_frames, frame)
    return np.sqrt(np.mean(trimmed * trimmed, axis=1)).astype(np.float32)


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Collapse a (n, channels) or (n,) buffer to mono float32."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 2:
        return arr.mean(axis=1).astype(np.float32)
    return arr


def normalize_peak(signal: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Scale so the peak magnitude equals target. No-op on silence."""
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak < 1e-9:
        return signal.astype(np.float32, copy=False)
    return (signal.astype(np.float64) * (target / peak)).astype(np.float32)
