"""Speech Fake tests: FakeSTT scripted transcripts, FakeTTS deterministic audio."""

import numpy as np
import pytest

from rocky_mini.speech.stt import FakeSTT
from rocky_mini.speech.tts import FakeTTS


@pytest.mark.asyncio
async def test_fake_stt_returns_scripted():
    stt = FakeSTT(["hello rocky", "teach you"])
    assert await stt.transcribe(np.zeros(10), 16000) == "hello rocky"
    assert await stt.transcribe(np.zeros(10), 16000) == "teach you"
    assert await stt.transcribe(np.zeros(10), 16000) == ""  # exhausted -> default


@pytest.mark.asyncio
async def test_fake_tts_length_tracks_text():
    tts = FakeTTS(sample_rate=22050, per_char_s=0.04)
    short = await tts.synthesize("hi")
    long = await tts.synthesize("a much longer sentence here")
    assert short.dtype == np.float32
    assert len(long) > len(short)
    assert np.max(np.abs(long)) > 0.0


@pytest.mark.asyncio
async def test_fake_tts_is_deterministic():
    tts = FakeTTS()
    a = await tts.synthesize("Rocky fix")
    b = await tts.synthesize("Rocky fix")
    assert np.array_equal(a, b)
