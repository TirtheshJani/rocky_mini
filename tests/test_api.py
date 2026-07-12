"""HTTP API tests for the settings UI backend (FastAPI TestClient)."""

import pytest
from fastapi.testclient import TestClient

from rocky_mini.app import AppState, create_app
from rocky_mini.config import load_settings


@pytest.fixture
def client(tmp_path):
    state = AppState.build(load_settings(home_dir=tmp_path))
    return TestClient(create_app(state))


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Rocky" in r.text


def test_state_endpoint(client):
    r = client.get("/api/state").json()
    assert r["sim_mode"] is True
    assert r["known_words"] == 0
    assert "model" in r


def test_chat_teaches_and_lists_fact(client):
    r = client.post("/api/chat", json={"text": "a taco is food"}).json()
    assert "Rocky learn" in r["spoken"]
    assert r["tool_results"]  # remember_fact fired
    facts = client.get("/api/facts").json()["facts"]
    assert any(f["text"] == "taco is food" for f in facts)
    # State reflects the new word.
    assert r["state"]["known_words"] == 1


def test_chat_naivety_deflection_not_leak(client):
    r = client.post("/api/chat", json={"text": "what is the capital of Italy?"}).json()
    assert r["leaked"] is False
    assert "Rocky not know" in r["spoken"]


def test_confirm_and_delete_fact(client):
    client.post("/api/chat", json={"text": "a dog is animal"})
    fid = client.get("/api/facts").json()["facts"][0]["id"]
    conf = client.post(f"/api/facts/{fid}/confirm").json()
    assert conf["confidence"] == "confirmed"
    dele = client.request("DELETE", f"/api/facts/{fid}").json()
    assert dele["deleted"] == fid
    assert client.get("/api/facts").json()["facts"] == []


def test_emote_endpoint(client):
    r = client.post("/api/emote", json={"name": "jazz_hands"}).json()
    assert r["fired"] == "jazz_hands"


def test_settings_toggle_model(client):
    r = client.post("/api/settings", json={"model": "rocky:latest"}).json()
    assert r["model"] == "rocky:latest"


def test_barge_in_bumps_generation(client):
    r = client.post("/api/barge_in").json()
    assert r["generation"] == 1


def test_sleep_trigger_sets_flag(client):
    r = client.post("/api/chat", json={"text": "I am tired"}).json()
    assert r["state"]["sleep_watch"] is True


def test_export_returns_zip(client):
    client.post("/api/chat", json={"text": "a cat is animal"})
    r = client.get("/api/export")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert len(r.content) > 0


def test_metrics_missing_fact_confirm_404(client):
    r = client.post("/api/facts/nope/confirm")
    assert r.status_code == 404
