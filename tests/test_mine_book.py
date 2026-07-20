"""mine_book pure helpers, on SYNTHETIC sample text and a fake generator.

No real book and no Ollama are touched here: the sample lines below are invented for the
test. Verifies normalization, speaker-attributed extraction, paraphrase QC, and that the
memorization probes are short.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))

import mine_book as mb  # noqa: E402
from rocky_mini.brain.persona import STAGE_FLUENT, build_system_prompt  # noqa: E402

# Invented sample. Straight quotes (post-normalization). Not from any source.
SAMPLE = (
    'Rocky tapped the wall. "Good. Good. Good," he chirped. '
    'Later, Rocky said, "You are good friend, question?" '
    'Rocky lifted a tool. "Rocky fix." '
    "Then a very long stretch of ordinary human narration with nobody named speaking at all, "
    "padding the distance well past sixty characters on each side so the next quote is alone, "
    '"This sentence has no alien speaker near it in the window here." '
    "and afterward still more plain narration continues for a good while with no named speaker "
    "so nothing attributes that lonely quote to anyone in particular at all across this passage."
)


def test_normalize_text_maps_curly_and_ligatures():
    raw = "“Hello” — oﬀice ‘hi’"
    out = mb.normalize_text(raw)
    assert '"Hello"' in out
    assert "office" in out
    assert "'hi'" in out
    assert "—" not in out


def test_find_rocky_lines_uses_attribution_window():
    lines = mb.find_rocky_lines(SAMPLE)
    assert "Good. Good. Good," in lines
    assert "You are good friend, question?" in lines
    assert "Rocky fix." in lines
    assert "This sentence has no alien anywhere near it in the window at all." not in lines


def test_find_rocky_lines_skips_overlong():
    long_line = " ".join(["word"] * 40)
    text = f'Rocky said, "{long_line}"'
    assert mb.find_rocky_lines(text, max_words=25) == []


def test_paraphrase_pair_drops_leak_keeps_clean():
    leak = mb.paraphrase_pair("x", lambda line, seed: "The capital of France is Paris.", 0, 0)
    assert leak is None
    ok = mb.paraphrase_pair("x", lambda line, seed: "You are good friend, question?", 0, 0)
    assert ok is not None
    assert ok["messages"][0]["content"] == build_system_prompt(STAGE_FLUENT)
    assert ok["messages"][1]["content"] in mb.USER_ROTATION
    assert ok["messages"][-1]["content"].endswith("question?")


def test_holdout_probes_are_short():
    lines = ["one two three four five six seven", "a b c"]
    probes = mb.holdout_probes(lines, n=5, words=3)
    assert len(probes) == 2
    assert probes[0]["probe"] == "one two three"
    assert probes[1]["probe"] == "a b c"
