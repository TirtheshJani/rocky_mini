"""Entry point: RockyMiniApp (dashboard-visible) and the sim CLI.

Launch contract (reachy-mini 1.9.0, see docs/hardware-audit.md F1/F2): the daemon
spawns `python -u -m rocky_mini.main` as a subprocess, the module main() calls
wrapped_run(), the base class connects the ReachyMini handle and serves the settings
UI (custom_app_url), and calls run(reachy_mini, stop_event). Stop from the dashboard
is SIGINT; run() must poll stop_event and return within the daemon's 20 s timeout.

The SDK is import-guarded. When it is absent (Windows dev, CI), everything still runs
in sim: the settings UI + ConversationLoop drive the whole pipeline with Fakes.

Ordered shutdown: stop the mixer pump -> drain the Mixer -> MotionThread ramps 0.6 s
to neutral (antennas at rest ~10 deg, never 0) -> close audio. Memory is fsynced per
write, so a SIGINT mid-turn never loses facts.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading

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

    Constructing it builds the sim-capable AppState. On hardware the SDK base class
    serves the settings UI and run() supervises the motion and audio threads.
    """

    # The dashboard discovers the settings link by regex-scanning main.py for this
    # exact assignment shape; keep it a literal string (audit F3).
    custom_app_url: str | None = "http://0.0.0.0:8042"
    # "default" auto-resolves to LOCAL when the app runs on the robot itself.
    # ROCKY_MEDIA_BACKEND overrides it for dev runs against a local sim daemon
    # (e.g. "no_media"); this selects the SDK's media transport, while
    # ROCKY_AUDIO_BACKEND selects which AudioIO the Mixer uses. See config.py.
    request_media_backend: str | None = "default"

    def __init__(self, settings: Settings | None = None) -> None:
        if _HAS_SDK:
            super().__init__()
            override = os.environ.get("ROCKY_MEDIA_BACKEND")
            if override:
                self.media_backend = override
        else:
            self.stop_event = threading.Event()
        self.settings = settings or load_settings()
        self.state = AppState.build(self.settings)
        # When the base class owns the webserver, attach our API routes to its app;
        # otherwise (sim, no SDK) build the standalone app.
        base_app = getattr(self, "settings_app", None)
        self.app = create_app(self.state, app=base_app)

    def run(self, reachy_mini: object | None = None, stop_event: threading.Event | None = None) -> None:
        """Daemon entry (via wrapped_run) or sim UI (via cli, no arguments)."""
        if reachy_mini is not None and _HAS_SDK:
            self._run_hardware(reachy_mini, stop_event or self.stop_event)
        else:
            self._serve_ui()

    def _serve_ui(self) -> None:  # pragma: no cover - live process
        import uvicorn

        logger.info("Rocky Mini sim UI on http://%s:%s", self.settings.ui_host, self.settings.ui_port)
        uvicorn.run(self.app, host=self.settings.ui_host, port=self.settings.ui_port, log_level="info")

    def _run_hardware(self, mini: object, stop_event: threading.Event) -> None:
        """Supervise the hardware threads until the daemon asks us to stop."""
        import asyncio

        from .audio.chords import ChordBank
        from .audio.input import AudioInThread, SpeechSegment
        from .audio.io import ReachyMediaIO, make_audio_io
        from .audio.vad import FakeVAD, SileroVAD
        from .turn import SPEAKING

        if self.settings.audio_backend == "reachy":
            io = ReachyMediaIO(mini)  # raises if the SDK surface is missing
            vad = SileroVAD(rate=self.settings.sample_rate_in)
        else:  # dev against a sim daemon: fake/sounddevice audio, real motion
            io = make_audio_io(self.settings.audio_backend)
            vad = FakeVAD()
        out_rate = getattr(io, "output_sample_rate", self.settings.sample_rate_out)
        self.state.mixer.io = io
        self.state.mixer.sample_rate = out_rate
        self.state.loop.chords = ChordBank(sample_rate=out_rate)

        # The asyncio loop is its own latency domain (plan.md): voice turns are
        # submitted onto it from AudioInThread without blocking the mic reads.
        aio = asyncio.new_event_loop()
        aio_thread = threading.Thread(target=aio.run_forever, daemon=True, name="ConversationLoop")

        def on_speech(seg: SpeechSegment) -> None:
            # t_start = last voiced frame, so the 0.6 s hangover counts (footgun 7).
            asyncio.run_coroutine_threadsafe(
                self.state.loop.handle_voice_turn(
                    seg.pcm, seg.sample_rate, t_start=seg.t_last_voiced
                ),
                aio,
            )

        audio_in = AudioInThread(
            io=io,
            vad=vad,
            motion=self.state.motion,
            on_speech=on_speech,
            is_speaking=lambda: self.state.loop.state == SPEAKING or self.state.mixer.pending(),
            on_barge_in=self.state.loop.barge_in,
            half_duplex=self.settings.half_duplex,
            rate=self.settings.sample_rate_in,
            hangover_s=self.settings.vad_hangover_s,
        )
        motion_thread = threading.Thread(
            target=self.state.motion.run, args=(mini, stop_event),
            daemon=True, name="MotionThread",
        )
        pump_thread = threading.Thread(
            target=self._mixer_pump, args=(stop_event,),
            daemon=True, name="MixerPump",
        )
        audio_in_thread = threading.Thread(
            target=audio_in.run, args=(stop_event,),
            daemon=True, name="AudioInThread",
        )
        aio_thread.start()
        motion_thread.start()
        pump_thread.start()
        audio_in_thread.start()
        try:
            stop_event.wait()
        finally:
            # Ordered shutdown (plan.md): audio in -> asyncio -> mixer drain ->
            # motion neutral ramp -> close the device.
            stop_event.set()
            audio_in_thread.join(timeout=1.0)
            aio.call_soon_threadsafe(aio.stop)
            aio_thread.join(timeout=2.0)
            pump_thread.join(timeout=1.0)
            self.state.mixer.render()  # drain anything still queued
            motion_thread.join(timeout=2.0)  # includes the 0.6 s neutral ramp
            io.close()
            logger.info("Rocky Mini shutdown complete; memory fsynced per write.")

    def _mixer_pump(self, stop_event: threading.Event) -> None:
        """AudioOut cadence: render queued PCM to the device until stopped."""
        period = self.state.mixer.frame / float(self.state.mixer.sample_rate)
        while not stop_event.is_set():
            self.state.mixer.render()
            stop_event.wait(period)


def main() -> None:  # pragma: no cover - live process
    """Daemon subprocess entry: python -u -m rocky_mini.main (no arguments)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = RockyMiniApp()
    if _HAS_SDK:
        try:
            app.wrapped_run()  # SIGINT from the daemon -> KeyboardInterrupt
        except KeyboardInterrupt:
            app.stop()
    else:
        logger.warning("reachy-mini SDK not installed; serving the sim UI instead.")
        app.run()


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
    import sys

    # `python -m rocky_mini.main` with no arguments is the daemon's launch shape;
    # any argument (e.g. --sim) means a human at a terminal wants the CLI.
    if len(sys.argv) > 1:
        cli()
    else:
        main()
