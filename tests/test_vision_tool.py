"""Phase 6 look-around tool tests: the FakeVisionObserver and the AppState wiring.

Sim-only: these never import cv2, the reachy_mini SDK, or httpx. The FakeVisionObserver
carries the whole path, and AppState.build defaults to the sim backend where
state.vision is None (so the None-jpeg branch returns the canned in-character reading).
"""

from rocky_mini.app import AppState
from rocky_mini.config import load_settings
from rocky_mini.vision import FakeVisionObserver


def test_fake_observer_reads_dummy_jpeg():
    obs = FakeVisionObserver()
    reading = obs.observe(b"\xff\xd8\xff\xd9", "what do you see")
    assert isinstance(reading, str) and reading.strip()


def test_fake_observer_no_reading_for_none():
    obs = FakeVisionObserver()
    assert obs.observe(None, "what do you see") == "Camera instrument returns no reading."


def test_look_around_tool_executes_in_sim(tmp_path):
    settings = load_settings(home_dir=tmp_path)
    state = AppState.build(settings)
    # Sim backend -> FakeVisionObserver; state.vision is None -> None-jpeg branch.
    assert isinstance(state.vision_observer, FakeVisionObserver)
    assert state.vision is None
    out = state.loop.tools.execute("look_around", {"question": "what do you see"})
    assert isinstance(out, str) and out.strip()
