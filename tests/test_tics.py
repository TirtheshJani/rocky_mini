"""TicLinter tests: question particle, assistant-ism strip, tripling/article metrics."""

from rocky_mini.brain.tics import lint


def test_question_particle_inserted():
    r = lint("Sun is hot?")
    assert r.text == "Sun is hot, question?"
    assert r.metrics.question_particle_hits == 1
    assert "question" in r.sfx


def test_existing_particle_not_doubled():
    r = lint("Sun is hot, question?")
    assert r.text == "Sun is hot, question?"
    assert r.text.count("question") == 1


def test_non_question_untouched():
    r = lint("Rocky fix.")
    assert r.text == "Rocky fix."
    assert r.metrics.question_sentences == 0


def test_assistant_isms_stripped():
    r = lint("As an AI language model, I would be happy to help. Rocky here.")
    assert "AI" not in r.text
    assert "happy to" not in r.text
    assert "Rocky here." in r.text


def test_tripling_triggers_jazz_hands():
    r = lint("Good. Good. Good.")
    assert r.metrics.tripling_count == 1
    assert "jazz_hands" in r.emotes


def test_no_tripling_no_jazz_hands():
    r = lint("Good. Good.")
    assert r.metrics.tripling_count == 0
    assert "jazz_hands" not in r.emotes


def test_article_rate_measured():
    r = lint("The dog and a cat.")
    # "the" and "a" out of 5 words.
    assert r.metrics.article_count == 2
    assert 0.0 < r.metrics.article_rate < 0.5


def test_question_particle_rate_full_when_no_questions():
    r = lint("Rocky fix. Good.")
    assert r.metrics.question_particle_rate == 1.0
