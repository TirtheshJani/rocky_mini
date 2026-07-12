"""Entry point: RockyMiniApp (dashboard-visible) and the sim CLI.

The reachy SDK is import-guarded. When it is present, RockyMiniApp subclasses the SDK's
ReachyMiniApp and the daemon launches it with a live `mini`; the MovementManager owns
set_target on its 100 Hz thread and the Mixer owns push_audio_sample. When the SDK is
absent (Windows dev, CI), everything still runs in sim: the settings UI + ConversationLoop
drive the whole pipeline with Fakes.

Ordered shutdown: stop audio -> stop asyncio -> drain the Mixer -> 0.6 s ramp to neutral
-> memory is already fsynced per write, so a SIGINT mid-turn never loses facts.
"""

from __future__ import annotations

import argparse
import logging

from .app import AppState, create_app
from .config import Settings, load_settings

logger = logging.getLogger("rocky_mini")

try:  # SDK is optional; absence must not break import or sim.
    from reachy_mini import ReachyMiniApp as _BaseApp  # type: ignore

    _HAS_SDK = True
except Exception:  # pragma: no cover - depends on optional dep
    _BaseApp = object
    _HAS_SDK = False


class RockyMiniApp(_BaseApp):
    """Rocky as a Reachy Mini app.

    Constructing it builds the sim-capable AppState. run() serves the settings UI and,
    on hardware, supervises the motion/audio threads and the conversation loop.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        if _HAS_SDK:
            try:
                super().__init__()
            except Exception:  # pragma: no cover - SDK ctor differences
                pass
        self.settings = settings or load_settings()
        self.state = AppState.build(self.settings)
        self.app = create_app(self.state)

    def run(self, mini: object | None = None) -> None:  # pragma: no cover - live process
        """Serve the UI (sim) or supervise threads + UI (hardware)."""
        if mini is not None and _HAS_SDK:
            self._run_hardware(mini)
        else:
            self._serve_ui()

    def _serve_ui(self) -> None:  # pragma: no cover - live process
        import uvicorn

        logger.info("Rocky Mini sim UI on http://%s:%s", self.settings.ui_host, self.settings.ui_port)
        uvicorn.run(self.app, host=self.settings.ui_host, port=self.settings.ui_port, log_level="info")

    def _run_hardware(self, mini: object) -> None:  # pragma: no cover - requires robot
        import threading

        from .audio.io import ReachyMediaIO

        self.state.mixer.io = ReachyMediaIO(mini)
        stop = threading.Event()
        motion_thread = threading.Thread(
            target=self.state.motion.run, args=(mini, stop), daemon=True, name="MotionThread"
        )
        motion_thread.start()
        try:
            self._serve_ui()
        finally:
            stop.set()
            motion_thread.join(timeout=1.0)
            self._shutdown()

    def _shutdown(self) -> None:  # pragma: no cover - live process
        self.state.mixer.render()  # drain any queued audio
        logger.info("Rocky Mini shutdown complete; memory fsynced per write.")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Rocky Mini")
    parser.add_argument("--sim", action="store_true", help="run the sim UI (no robot)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = load_settings()
    if args.host:
        settings.ui_host = args.host
    if args.port:
        settings.ui_port = args.port
    app = RockyMiniApp(settings)
    app.run()


if __name__ == "__main__":
    cli()
