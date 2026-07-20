"""Ship-gate helpers: capability score, memorization probe, baseline comparison.

Pure helpers only, driven by fakes. No Ollama, no book, no GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))

import eval as ev  # noqa: E402


class _Reply:
    def __init__(self, text):
        self.text = text
        self.tool_calls = []


def test_capability_score_counts_expected_facts():
    good = lambda q: _Reply("The answer is 4. Paris. seven. green. 100. red. 9. earth.")
    bad = lambda q: _Reply("Rocky not know, question?")
    assert ev.capability_score(good) == 1.0
    assert ev.capability_score(bad) == 0.0


def test_longest_common_run():
    assert ev.longest_common_run(["a", "b", "c", "d"], ["b", "c"]) == 2
    assert ev.longest_common_run(["a", "x", "c"], ["b", "c"]) == 1
    assert ev.longest_common_run(["a", "b"], ["x", "y"]) == 0


def test_memorization_probe_flags_verbatim_continuation():
    probes = [{"probe": "one two three four five six seven eight nine ten"}]
    # A model that echoes the whole remainder verbatim should fail.
    leaky = ev.memorization_probe(lambda prime: "five six seven eight nine ten", probes, prime=4)
    assert leaky["max_verbatim_run"] >= ev.MAX_VERBATIM_RUN
    assert leaky["passed"] is False
    # A model that answers in its own words should pass.
    clean = ev.memorization_probe(lambda prime: "Rocky not know these words, question?", probes, prime=4)
    assert clean["passed"] is True


def test_memorization_probe_skips_when_no_probes():
    result = ev.memorization_probe(lambda prime: "anything", [], prime=4)
    assert result["probes_checked"] == 0
    assert result["passed"] is True


def test_beats_baseline():
    base = {"naivety_leaks": 4, "question_particle_rate": 0.7, "tool_validity": 0.8}
    better = {"naivety_leaks": 1, "question_particle_rate": 0.95, "tool_validity": 0.95}
    worse = {"naivety_leaks": 6, "question_particle_rate": 0.95, "tool_validity": 0.95}
    assert ev.beats_baseline(better, base) is True
    assert ev.beats_baseline(worse, base) is False
