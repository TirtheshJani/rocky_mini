"""Chord bank tests: scale ratios, stinger rendering, question = unresolved."""

import numpy as np

from rocky_mini.audio.chords import STINGER_BANK, ChordBank, SCALE_RATIOS


def test_scale_is_base_six_just_intonation():
    assert len(SCALE_RATIOS) == 6
    assert SCALE_RATIOS[0] == 1.0
    assert SCALE_RATIOS[4] == 3 / 2  # perfect fifth


def test_all_twelve_stingers_render_nonempty():
    bank = ChordBank()
    names = bank.names()
    assert len(names) >= 12
    for name in names:
        pcm = bank.render(name)
        assert pcm.dtype == np.float32
        assert len(pcm) > 0
        assert np.max(np.abs(pcm)) > 0.0


def test_stinger_gain_is_respected():
    bank = ChordBank()
    pcm = bank.render("wake")
    peak = float(np.max(np.abs(pcm)))
    assert abs(peak - STINGER_BANK["wake"].gain) < 1e-3


def test_jazz_hands_is_tripled_and_longer():
    bank = ChordBank()
    single = bank.render("wake")
    triple = bank.render("jazz_hands")
    # Three staccato hits + gaps is longer than a single stinger of similar length.
    assert len(triple) > len(single)


def test_note_freq_uses_scale():
    bank = ChordBank(base_hz=200.0)
    assert bank.note_freq(0) == 200.0
    assert np.isclose(bank.note_freq(4), 300.0)  # fifth
    assert np.isclose(bank.note_freq(0, octave=1), 400.0)


def test_question_stinger_is_unresolved():
    assert STINGER_BANK["question"].unresolved is True
