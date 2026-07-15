"""Voice activity detection behind a Protocol, per decision 4.

SileroVAD runs the vendored Silero ONNX model (assets/models/silero_vad.onnx, MIT,
see assets/models/README.md) directly with onnxruntime. The silero-vad PyPI package
is deliberately NOT used: since v6 it depends on torch + the CUDA stack, which the
CM4 can neither fit nor needs (decisions.md #11). FakeVAD keeps tests and sim pure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "assets" / "models" / "silero_vad.onnx"


@runtime_checkable
class VAD(Protocol):
    """Per-frame speech probability. Frames are mono float32 at 16 kHz."""

    frame_samples: int

    def probability(self, frame: np.ndarray) -> float:
        ...

    def reset(self) -> None:
        ...


class FakeVAD:
    """Deterministic VAD for tests and sim: scripted probabilities, else energy."""

    frame_samples = 512

    def __init__(self, probs: list[float] | None = None, energy_threshold: float = 0.05) -> None:
        self._probs = list(probs) if probs else []
        self._threshold = energy_threshold

    def probability(self, frame: np.ndarray) -> float:
        if self._probs:
            return self._probs.pop(0)
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        return 0.95 if rms > self._threshold else 0.02

    def reset(self) -> None:
        pass


class SileroVAD:
    """Silero VAD via onnxruntime. 512-sample frames at 16 kHz only."""

    frame_samples = 512

    def __init__(self, rate: int = 16000, model_path: Path | None = None) -> None:
        if rate != 16000:
            raise ValueError(f"SileroVAD supports 16 kHz only, got {rate}")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "onnxruntime is not installed; it is a core dependency, reinstall the app"
            ) from exc
        path = model_path or MODEL_PATH
        if not path.exists():
            raise RuntimeError(f"Silero VAD model missing at {path}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1  # keep the CM4 cores for motion and mixing
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def probability(self, frame: np.ndarray) -> float:
        frame = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        prob, self._state = self._session.run(
            None, {"input": frame, "state": self._state, "sr": self._sr}
        )
        return float(np.squeeze(prob))

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
