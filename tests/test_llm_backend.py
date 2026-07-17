"""LLM backend selector: defaults to the Fake, uses OllamaLLM only when flagged.

The sim/test seam (decision 4) requires the default path to construct no real client,
so pytest stays green with no Ollama, no openai package, and no GPU. This test never
imports the openai client: the ollama branch is exercised with a stub.
"""

import rocky_mini.app as app_module
from rocky_mini.app import AppState, _build_llm
from rocky_mini.brain.llm import FakeLLM
from rocky_mini.config import load_settings


def test_default_backend_is_fake(tmp_path):
    settings = load_settings(home_dir=tmp_path)
    assert settings.llm_backend == "fake"
    assert isinstance(_build_llm(settings), FakeLLM)


def test_build_uses_fake_by_default(tmp_path):
    state = AppState.build(load_settings(home_dir=tmp_path))
    assert isinstance(state.loop.llm, FakeLLM)


def test_ollama_backend_selected_without_importing_openai(tmp_path, monkeypatch):
    constructed: dict = {}

    class StubOllama:
        def __init__(self, base_url, model, api_key="ollama", keep_alive=-1):
            constructed.update(
                base_url=base_url, model=model, api_key=api_key, keep_alive=keep_alive
            )

        async def stream(self, messages, tools=None):  # pragma: no cover - not called here
            yield None

    monkeypatch.setattr(app_module, "OllamaLLM", StubOllama)
    settings = load_settings(
        home_dir=tmp_path,
        llm_backend="ollama",
        llm_base_url="http://localhost:11434/v1",
        model="rocky:latest",
        keep_alive=-1,
    )
    llm = _build_llm(settings)
    assert isinstance(llm, StubOllama)
    assert constructed["base_url"] == "http://localhost:11434/v1"
    assert constructed["model"] == "rocky:latest"
    assert constructed["keep_alive"] == -1
