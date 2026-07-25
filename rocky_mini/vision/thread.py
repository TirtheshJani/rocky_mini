"""VisionThread: the vision latency domain (plan.md), 5 Hz client-side gaze.

step() grabs a frame, detects a face, converts it to yaw/pitch, and feeds
MovementManager.set_gaze(). It NEVER touches set_target or audio (footgun 1): gaze is
just another input the 100 Hz manager reconciles against DoA. It also keeps the latest
JPEG in a single-slot behind a lock, for the Phase 6 look tool.

Sim-first: no cv2 or SDK import here or at module top. The Camera and FaceDetector
seams (camera.py) carry the only hardware-specific code, behind Fakes for tests.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .camera import Camera, FaceDetector

logger = logging.getLogger("rocky_mini.vision")


def gaze_from_uv(u: float, v: float, h_fov_deg: float, v_fov_deg: float) -> tuple[float, float]:
    """Map a normalized face center (u, v) to a (yaw_deg, pitch_deg) look offset.

    A centered face (0.5, 0.5) yields (0, 0). u = 1 (face at the right image edge)
    yields yaw = +h_fov/2; v = 0 (face at the top) yields pitch = -v_fov/2.

    Sign convention: positive pitch nods the head DOWN (matching the 'understand'
    emote), so a face high in the frame (v < 0.5) gives a negative pitch and Rocky
    looks up. The physical sign of BOTH yaw and pitch is a HARDWARE-verification item
    (bring-up): if Rocky turns away from the face, negate the factor at the manager.
    """
    yaw = (u - 0.5) * h_fov_deg
    pitch = (v - 0.5) * v_fov_deg
    return (yaw, pitch)


class VisionThread:
    """Owns vision. step() is the unit of work; run() loops it at hz on hardware."""

    def __init__(
        self,
        camera: Camera,
        detector: FaceDetector,
        motion,  # MovementManager; only set_gaze is used (footgun 1)
        hz: float = 5.0,
        h_fov_deg: float = 70.0,
        v_fov_deg: float = 42.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.motion = motion
        self.hz = hz
        self.h_fov_deg = h_fov_deg
        self.v_fov_deg = v_fov_deg
        self.clock = clock
        self._lock = threading.Lock()
        self.latest_jpeg: bytes | None = None

    def step(self) -> None:
        """One unit of work: refresh the JPEG slot and steer gaze from a detected face.

        Wrapped so one bad frame never kills the thread; a failure logs at debug and
        the next tick tries again.
        """
        try:
            frame = self.camera.get_frame()
            try:
                jpeg = self.camera.get_frame_jpeg()
            except Exception:  # noqa: BLE001 - jpeg is best-effort, never blocks gaze
                jpeg = None
            if jpeg is not None:
                with self._lock:
                    self.latest_jpeg = jpeg
            if frame is None:
                return
            uv = self.detector.detect(frame)
            if uv is None:
                self.motion.set_gaze(None, None)
            else:
                yaw, pitch = gaze_from_uv(uv[0], uv[1], self.h_fov_deg, self.v_fov_deg)
                self.motion.set_gaze(yaw, pitch)
        except Exception:  # noqa: BLE001 - a dead vision thread must never crash the app
            logger.debug("VisionThread.step failed; continuing", exc_info=True)

    def run(self, stop) -> None:  # pragma: no cover - thread loop, logic is step()
        logger.info("VisionThread up (%.1f Hz)", self.hz)
        period = 1.0 / self.hz
        while not stop.is_set():
            self.step()
            stop.wait(period)

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self.latest_jpeg
