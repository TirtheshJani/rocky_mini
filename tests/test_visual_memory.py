"""Visual object memory tests (decision 21): the embedder seam, the ObjectStore, and
the teach/recognize path through AppState.

Sim-only: these never import cv2, httpx, or torch. FakeImageEmbedder carries the whole
embedding path, and AppState.build defaults to the sim backend (FakeImageEmbedder + a
None camera) so teach degrades to a plain fact unless a stub frame is supplied.
"""

import pytest

from rocky_mini.app import AppState
from rocky_mini.config import load_settings
from rocky_mini.memory.objects import MAX_SHOTS, ObjectStore
from rocky_mini.vision.embed import FakeImageEmbedder, HttpImageEmbedder, cosine


class _StubVision:
    """Minimal VisionThread stand-in: get_jpeg() returns a fixed frame (or None)."""

    def __init__(self, jpeg: bytes | None) -> None:
        self._jpeg = jpeg

    def get_jpeg(self) -> bytes | None:
        return self._jpeg


# -- FakeImageEmbedder -----------------------------------------------------
def test_fake_embedder_is_deterministic_and_unit_norm():
    e = FakeImageEmbedder()
    v1 = e.embed(b"mug-bytes")
    v2 = e.embed(b"mug-bytes")
    assert v1 == v2
    assert sum(x * x for x in v1) == pytest.approx(1.0)


def test_fake_embedder_none_for_none_frame():
    assert FakeImageEmbedder().embed(None) is None
    assert FakeImageEmbedder().embed(b"") is None


def test_fake_embedder_different_bytes_land_far_below_threshold():
    e = FakeImageEmbedder()
    a = e.embed(b"coffee mug frame")
    b = e.embed(b"a totally different object")
    assert cosine(a, b) < 0.60


# -- HttpImageEmbedder -----------------------------------------------------
def test_http_embedder_none_frame_needs_no_httpx():
    # A None frame must short-circuit before the lazy httpx import, so this passes
    # even in an environment without httpx installed.
    assert HttpImageEmbedder("http://x").embed(None) is None


# -- cosine ----------------------------------------------------------------
def test_cosine_identical_orthogonal_and_mismatch():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0  # dim mismatch -> 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero norm -> 0.0
    assert cosine(None, [1.0]) == 0.0


# -- ObjectStore -----------------------------------------------------------
def test_enroll_and_reload_persists(tmp_path):
    e = FakeImageEmbedder()
    store = ObjectStore(tmp_path)
    oid = store.enroll("coffee mug", e.embed(b"mug"))
    assert store.names() == ["coffee mug"]

    reloaded = ObjectStore(tmp_path)
    assert reloaded.names() == ["coffee mug"]
    hit = reloaded.match(e.embed(b"mug"), threshold=0.60, margin=0.08)
    assert hit is not None and hit[0] == "coffee mug"
    # id survives the reload (event-sourced replay), so re-teaching appends, not forks.
    assert reloaded.enroll("coffee mug", e.embed(b"mug angle 2")) == oid


def test_enroll_caps_shots_at_three(tmp_path):
    e = FakeImageEmbedder()
    store = ObjectStore(tmp_path)
    for i in range(5):
        store.enroll("mug", e.embed(f"mug-{i}".encode()))
    obj = store.active_objects()[0]
    assert len(obj.embeddings) == MAX_SHOTS
    # The oldest shots were dropped; the last MAX_SHOTS are kept.
    assert obj.embeddings[-1] == e.embed(b"mug-4")


def test_match_rejects_below_threshold(tmp_path):
    e = FakeImageEmbedder()
    store = ObjectStore(tmp_path)
    store.enroll("mug", e.embed(b"mug"))
    # An unrelated frame lands near cosine 0, well under the threshold.
    assert store.match(e.embed(b"unrelated thing"), threshold=0.60, margin=0.08) is None


def test_match_rejects_when_margin_too_small(tmp_path):
    # Two objects whose stored shots are near-identical vectors: the top beats the
    # runner-up by less than the margin, so the ambiguous pick is rejected.
    store = ObjectStore(tmp_path)
    base = [1.0] + [0.0] * 63
    near = [0.999] + [0.001] * 63
    store.enroll("mug", base)
    store.enroll("cup", near)
    probe = [1.0] + [0.0] * 63
    # Both score ~1.0; gap is tiny -> rejected on margin.
    assert store.match(probe, threshold=0.60, margin=0.08) is None


def test_single_object_skips_margin(tmp_path):
    # With one enrolled object there is nothing to confuse it with, so an above-threshold
    # score matches regardless of margin.
    store = ObjectStore(tmp_path)
    store.enroll("mug", [1.0] + [0.0] * 63)
    hit = store.match([1.0] + [0.0] * 63, threshold=0.60, margin=0.99)
    assert hit is not None and hit[0] == "mug"


def test_tombstone_survives_reload(tmp_path):
    store = ObjectStore(tmp_path)
    oid = store.enroll("mug", [1.0] + [0.0] * 63)
    store.delete_object(oid)
    assert store.names() == []
    assert ObjectStore(tmp_path).names() == []


# -- teach_object through AppState -----------------------------------------
def test_teach_in_sim_degrades_to_fact_only(tmp_path):
    # No camera in sim (state.vision is None): teach records the fact but enrolls no shot.
    state = AppState.build(load_settings(home_dir=tmp_path))
    out = state.loop.tools.execute("teach_object", {"name": "coffee mug"})
    assert isinstance(out, str) and "coffee mug" in out
    assert any(f.category == "object" for f in state.memory.active_facts())
    assert state.objects.names() == []


def test_teach_then_recognize_round_trips(tmp_path):
    state = AppState.build(load_settings(home_dir=tmp_path))
    state.vision = _StubVision(b"\xff\xd8coffee-mug-frame\xff\xd9")

    teach = state.loop.tools.execute("teach_object", {"name": "coffee mug"})
    assert "coffee mug" in teach
    assert state.objects.names() == ["coffee mug"]

    look = state.loop.tools.execute("look_around", {"question": "what do you see"})
    assert "recognize" in look and "coffee mug" in look

    # A different frame must NOT recognize the mug (false-positive guard).
    state.vision = _StubVision(b"\xff\xd8some-other-frame\xff\xd9")
    look2 = state.loop.tools.execute("look_around", {"question": "what do you see"})
    assert "recognize" not in look2


def test_teach_object_in_tool_specs(tmp_path):
    from rocky_mini.brain.tools import tool_specs

    spec = next(s for s in tool_specs() if s["function"]["name"] == "teach_object")
    props = spec["function"]["parameters"]["properties"]
    assert "name" in props


def test_kill_switch_makes_teach_fact_only_even_with_frame(tmp_path):
    state = AppState.build(load_settings(home_dir=tmp_path, object_memory_enabled=False))
    state.vision = _StubVision(b"\xff\xd8coffee-mug-frame\xff\xd9")

    teach = state.loop.tools.execute("teach_object", {"name": "coffee mug"})
    assert "coffee mug" in teach
    # Fact recorded, but no visual shot enrolled and no recognition on look_around.
    assert any(f.category == "object" for f in state.memory.active_facts())
    assert state.objects.names() == []
    look = state.loop.tools.execute("look_around", {"question": "x"})
    assert "recognize" not in look
