"""Vision seam: one Camera Protocol and one FaceDetector Protocol, each with a Fake.

Rocky's eyes are client-side only (decision 18, footgun 9 amended): the VisionThread
captures frames and detects a face here, then feeds MovementManager.set_gaze(). We
NEVER call mini.start_head_tracking(): that hands the head to the daemon and would
fight our single 100 Hz set_target owner (footgun 1).

Sim-first: importing this module never requires cv2 or the reachy_mini SDK. numpy is
a core dep so it lives at module top; cv2 is imported lazily inside HaarFaceDetector,
and the SDK is only ever reached through a passed-in mini handle (never imported).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("rocky_mini.vision")


@runtime_checkable
class Camera(Protocol):
    """A source of BGR frames plus a JPEG snapshot for the Phase 6 look tool."""

    def get_frame(self) -> "np.ndarray | None":
        """Return a BGR (H, W, 3) uint8 frame, or None if none is ready."""
        ...

    def get_frame_jpeg(self) -> "bytes | None":
        """Return a JPEG-encoded snapshot, or None if none is ready."""
        ...


def _default_frame() -> np.ndarray:
    """A deterministic 64x48 (WxH) BGR frame with one bright central blob."""
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[18:30, 26:38] = 255
    return frame


# A tiny stand-in JPEG payload: SOI + EOI markers. Non-empty and JPEG-shaped so the
# Phase 6 look-tool plumbing has something real to carry, with no encoder involved.
_DEFAULT_JPEG = b"\xff\xd8\xff\xd9"


class FakeCamera:
    """In-memory Camera for tests and sim. Returns fixed frame/jpeg, no I/O."""

    def __init__(
        self, frame: "np.ndarray | None" = None, jpeg: bytes = _DEFAULT_JPEG
    ) -> None:
        self._frame = _default_frame() if frame is None else np.asarray(frame, dtype=np.uint8)
        self._jpeg = jpeg

    def get_frame(self) -> "np.ndarray | None":
        return self._frame

    def get_frame_jpeg(self) -> "bytes | None":
        return self._jpeg


class ReachyCamera:
    """Physical-robot Camera wrapping mini.media (reachy-mini contract).

    No heavy imports at module top: it only calls through the passed handle, so the
    sim path never pulls cv2 or the SDK. A failed grab degrades to None (the
    VisionThread just skips that tick) rather than killing the thread.
    """

    def __init__(self, mini: object) -> None:
        self._mini = mini

    def get_frame(self) -> "np.ndarray | None":  # pragma: no cover - hardware path
        try:
            return self._mini.media.get_frame()
        except Exception:  # noqa: BLE001 - a bad grab must never kill the vision thread
            return None

    def get_frame_jpeg(self) -> "bytes | None":  # pragma: no cover - hardware path
        try:
            return self._mini.media.get_frame_jpeg()
        except Exception:  # noqa: BLE001
            return None


@runtime_checkable
class FaceDetector(Protocol):
    """Detects a single face and reports its normalized center."""

    def detect(self, frame: "np.ndarray") -> "tuple[float, float] | None":
        """Return the face center as normalized (u, v) in [0, 1] from the image
        top-left, or None if no face is present."""
        ...


class FakeDetector:
    """Scripted FaceDetector for tests. Ignores the frame.

    Returns each (u, v) | None in sequence, clamping at the last entry once the
    script is exhausted. With no script it always returns a centered (0.5, 0.5).
    """

    def __init__(
        self, script: "Iterable[tuple[float, float] | None] | None" = None
    ) -> None:
        self._script = list(script) if script is not None else [(0.5, 0.5)]
        self._i = 0

    def detect(self, frame: "np.ndarray") -> "tuple[float, float] | None":
        if not self._script:
            return None
        i = min(self._i, len(self._script) - 1)
        self._i += 1
        return self._script[i]


def normalized_center_from_bbox(
    x: float, y: float, w: float, h: float, img_w: float, img_h: float
) -> tuple[float, float]:
    """Center of a pixel bbox as normalized (u, v) in [0, 1] from the top-left."""
    u = (x + w / 2.0) / float(img_w)
    v = (y + h / 2.0) / float(img_h)
    return (u, v)


class HaarFaceDetector:
    """OpenCV Haar-cascade face detector. Lazy cv2 import.

    detect() downscales to a small grayscale width, picks the largest face, and
    normalizes its center. An EMA smooths the reported (u, v); a dropout hysteresis
    keeps returning the last smoothed value for up to hold_s after detection loss so a
    single dropped frame does not jerk Rocky's gaze back to center.
    """

    def __init__(
        self,
        scale_width: int = 320,
        ema_alpha: float = 0.5,
        hold_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "opencv-python is not installed. Install with: pip install 'rocky_mini[vision]'"
            ) from exc
        self._cv2 = cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_width = scale_width
        self.ema_alpha = ema_alpha
        self.hold_s = hold_s
        self.clock = clock
        self._smoothed: tuple[float, float] | None = None
        self._last_seen_t = -1e9

    def detect(self, frame: "np.ndarray") -> "tuple[float, float] | None":  # pragma: no cover - needs cv2
        cv2 = self._cv2
        h, w = frame.shape[:2]
        scale = self.scale_width / float(w) if w > self.scale_width else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        now = self.clock()
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            sh, sw = gray.shape[:2]
            u, v = normalized_center_from_bbox(x, y, fw, fh, sw, sh)
            if self._smoothed is None:
                self._smoothed = (u, v)
            else:
                a = self.ema_alpha
                su, sv = self._smoothed
                self._smoothed = (a * u + (1.0 - a) * su, a * v + (1.0 - a) * sv)
            self._last_seen_t = now
            return self._smoothed
        if self._smoothed is not None and (now - self._last_seen_t) <= self.hold_s:
            return self._smoothed
        self._smoothed = None
        return None


class DaemonFaceDetector:
    """Reads the daemon-side face detector via mini.get_tracked_face(wait=False).

    PENDING PROBE P1: the exact payload shape of get_tracked_face is UNVERIFIED, so the
    extraction below is defensive and tries several common shapes. This path MUST NOT
    call mini.start_head_tracking() (footgun 9, decision 18): that hands the head to the
    daemon and fights our single set_target owner. We only READ the detector's result
    and feed set_gaze. detect() ignores the passed frame; the daemon owns the camera.
    """

    def __init__(self, mini: object, img_w: int = 1280, img_h: int = 720) -> None:
        self._mini = mini
        self.img_w = img_w
        self.img_h = img_h
        self._warned = False

    def detect(self, frame: "np.ndarray") -> "tuple[float, float] | None":
        try:
            face = self._mini.get_tracked_face(wait=False)
        except Exception:  # noqa: BLE001 - any SDK fault degrades to "no face"
            return None
        if not getattr(face, "detected", False):
            return None
        uv = None
        try:
            uv = self._extract_uv(face)
        except Exception:  # noqa: BLE001
            uv = None
        if uv is None and not self._warned:
            logger.warning(
                "DaemonFaceDetector could not read a usable face position (probe P1 "
                "pending); gaze via daemon face is off. Payload attrs: %s",
                sorted(a for a in dir(face) if not a.startswith("_")),
            )
            self._warned = True
        return uv

    def _extract_uv(self, face: object) -> "tuple[float, float] | None":
        # Shape 1: already-normalized u/v attributes.
        u = getattr(face, "u", None)
        v = getattr(face, "v", None)
        if u is not None and v is not None:
            return (float(u), float(v))
        # Shape 2: a center (x, y), normalized already or in pixels needing division.
        center = getattr(face, "center", None)
        if center is not None and len(center) >= 2:
            cx, cy = float(center[0]), float(center[1])
            if cx <= 1.0 and cy <= 1.0:
                return (cx, cy)
            return (cx / float(self.img_w), cy / float(self.img_h))
        # Shape 3: a pixel bbox (x, y, w, h).
        bbox = getattr(face, "bbox", None)
        if bbox is not None and len(bbox) >= 4:
            x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            return normalized_center_from_bbox(x, y, w, h, self.img_w, self.img_h)
        return None
