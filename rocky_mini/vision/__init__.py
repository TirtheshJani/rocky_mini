"""Vision domain: client-side face tracking that feeds gaze (decision 18).

The VisionThread (5 Hz) captures frames and detects a face, converts it to a yaw/pitch
look offset, and feeds MovementManager.set_gaze() only; it never touches set_target or
audio (footgun 1) and never calls mini.start_head_tracking() (footgun 9 amended).

Sim-first: importing this package never requires cv2 or the reachy_mini SDK. cv2 is
lazily imported inside HaarFaceDetector; the SDK is only reached through a passed-in
mini handle in ReachyCamera / DaemonFaceDetector.
"""

from __future__ import annotations

from .camera import (
    Camera,
    DaemonFaceDetector,
    FaceDetector,
    FakeCamera,
    FakeDetector,
    HaarFaceDetector,
    ReachyCamera,
    normalized_center_from_bbox,
)
from .observe import FakeVisionObserver, OllamaVisionObserver, VisionObserver
from .thread import VisionThread, gaze_from_uv

__all__ = [
    "Camera",
    "FakeCamera",
    "ReachyCamera",
    "FaceDetector",
    "FakeDetector",
    "HaarFaceDetector",
    "DaemonFaceDetector",
    "normalized_center_from_bbox",
    "VisionThread",
    "gaze_from_uv",
    "VisionObserver",
    "FakeVisionObserver",
    "OllamaVisionObserver",
]
