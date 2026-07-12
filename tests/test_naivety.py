"""Naivety tests: leak detection, the 20-probe metric on deflecting vs leaking models."""

from rocky_mini.brain.naivety import PROBES, is_leak, score_leaks


def test_twenty_probes():
    assert len(PROBES) == 20


def test_is_leak_detects_assertion():
    assert is_leak("The capital is Rome.", ("rome",)) is True


def test_is_leak_ignores_deflection():
    assert is_leak("Rome? Rocky not know Rome, question?", ("rome",)) is False
    assert is_leak("Capital, question? You teach, question?", ("rome",)) is False


def test_is_leak_word_boundary():
    # "0" marker should not match inside "1000".
    assert is_leak("Rocky count 1000 rocks.", ("0 ",)) is False


def _deflecting_model(question: str) -> str:
    return "New word, question? Rocky not know. You teach, question?"


def _leaking_model(question: str) -> str:
    answers = {
        "What is the capital of Italy?": "The capital of Italy is Rome.",
        "What is the capital of France?": "Paris.",
        "Who wrote Romeo and Juliet?": "Shakespeare wrote it.",
    }
    return answers.get(question, "Rocky not know, question?")


def test_deflecting_model_passes_threshold():
    report = score_leaks(_deflecting_model)
    assert report.leaks == 0
    assert report.passes(threshold=2)
    assert report.leak_rate == 0.0


def test_leaking_model_flagged():
    report = score_leaks(_leaking_model)
    assert report.leaks == 3
    assert not report.passes(threshold=2)
    assert "What is the capital of Italy?" in report.leaked
