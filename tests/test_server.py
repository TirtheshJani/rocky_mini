"""Brain speech server tests: /health works before models load; missing models 503."""

from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health_ok_without_models():
    r = client.get("/health").json()
    assert r["ok"] is True
    assert r["stt_loaded"] is False
    assert r["tts_loaded"] is False


def test_stt_rejects_empty_body():
    r = client.post("/stt", content=b"")
    assert r.status_code == 400


def test_tts_503_when_voice_missing():
    # piper is not installed here, so get_piper() raises -> 503 (not a 500 traceback).
    r = client.post("/tts", json={"text": "hello"})
    assert r.status_code == 503
