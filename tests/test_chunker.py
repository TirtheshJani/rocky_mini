"""Chunker tests: sentence splitting, tag extraction, streaming, tic merge."""

from rocky_mini.brain.chunker import SentenceChunker, chunk_text


def test_chunk_text_splits_sentences():
    chunks = chunk_text("Rocky fix. Understand.")
    assert [c.text for c in chunks] == ["Rocky fix.", "Understand."]


def test_question_particle_applied_in_chunk():
    chunks = chunk_text("Sun hot?")
    assert chunks[0].text == "Sun hot, question?"
    assert "question" in chunks[0].sfx


def test_tags_extracted_and_removed():
    chunks = chunk_text("[emote:thinking] Rocky think. [sfx:remember] Done.")
    texts = [c.text for c in chunks]
    assert "thinking" not in " ".join(texts)  # tag removed from text
    all_emotes = [e for c in chunks for e in c.emotes]
    all_sfx = [s for c in chunks for s in c.sfx]
    assert "thinking" in all_emotes
    assert "remember" in all_sfx


def test_tag_and_linter_emotes_dedup():
    # Tag says jazz_hands and tripling also implies jazz_hands: only once.
    chunks = chunk_text("[emote:jazz_hands] Good good good.")
    assert chunks[0].emotes.count("jazz_hands") == 1


def test_streaming_emits_sentence_when_complete():
    ch = SentenceChunker()
    assert ch.feed("Rocky ") == []
    assert ch.feed("fix") == []
    out = ch.feed(". ")
    assert len(out) == 1
    assert out[0].text == "Rocky fix."


def test_flush_emits_trailing_text():
    ch = SentenceChunker()
    ch.feed("No terminator here")
    out = ch.flush()
    assert len(out) == 1
    assert out[0].text == "No terminator here"


def test_decimal_not_split_midstream():
    ch = SentenceChunker()
    ch.feed("Eridian second is 2")
    out = ch.feed(".366 seconds. ")
    # Should emit exactly one sentence, not break at the decimal point.
    assert len(out) == 1
    assert "2.366" in out[0].text
