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


def test_tts_503_when_voice_missing(monkeypatch):
    # Force the missing-voice condition regardless of what this machine has
    # installed: a bogus voice name makes get_piper() raise -> 503 (not a 500).
    import server.app as server_app

    monkeypatch.setattr(server_app, "PIPER_VOICE", "no-such-voice-xyz")
    monkeypatch.setattr(server_app, "_PIPER", None)
    r = client.post("/tts", json={"text": "hello"})
    assert r.status_code == 503
