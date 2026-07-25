"""Configuration for rocky_mini (pydantic-settings).

Values come from environment variables (prefix ROCKY_) and ~/.rocky_mini/.env.
Nothing here is secret except the optional LAN token; there are no paid-API keys.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def rocky_home() -> Path:
    """Return ~/.rocky_mini, creating it if needed.

    Memory and the runtime .env live here so they survive app reinstalls
    (an app-scoped dir is wiped by pip --force-reinstall; the home dir is not).
    """
    home = Path.home() / ".rocky_mini"
    home.mkdir(parents=True, exist_ok=True)
    return home


class Settings(BaseSettings):
    """Runtime settings. Immutable-ish: constructed once at startup."""

    model_config = SettingsConfigDict(
        env_prefix="ROCKY_",
        env_file=str(rocky_home() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Brain server (Ollama, OpenAI-compatible).
    llm_backend: str = "fake"  # "fake" (sim/test default) | "ollama" (real local brain).
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # Ollama ignores the value but the client requires one.
    model: str = "qwen2.5:7b-instruct"  # toggle: rocky:latest for the LoRA build.
    keep_alive: int = -1  # keep the model resident in VRAM.

    # Speech brain service (faster-whisper + Piper).
    speech_base_url: str = "http://localhost:8123"

    # Visual object memory (decision 21): frozen DINOv2 embeddings, few-shot cosine match.
    # The image embedder lives on the speech server (/embed_image), because Ollama has no
    # image-embedding API. Kill switch: ROCKY_OBJECT_MEMORY_ENABLED=0 makes the jsonl
    # sidecar inert and recognition a no-op.
    object_memory_enabled: bool = True
    embed_url: str = ""  # empty -> use speech_base_url (the speech server hosts /embed_image).
    object_match_threshold: float = 0.60  # min cosine to accept a match (pending hardware tuning).
    object_match_margin: float = 0.08  # min top1-top2 gap; skipped with one enrolled object.

    # LAN.
    lan_token: str = ""

    # Audio.
    audio_backend: str = "sounddevice"  # "reachy" | "sounddevice" | "fake"
    sample_rate_in: int = 16000
    sample_rate_out: int = 22050  # Piper en_US-lessac-medium native rate.
    mixer_frame: int = 320  # 50 Hz at 16 kHz control cadence.

    # Voice DSP (Eridian ring modulation).
    ring_carrier_hz: float = 140.0
    # Movie Rocky: mostly natural deadpan computer voice with a light metallic
    # edge, not a fully ring-modulated robot. 0.0 = clean voice, 1.0 = full alien.
    ring_mix: float = 0.2
    underlay_db: float = -12.0

    # Conversation timing.
    vad_hangover_s: float = 0.6
    ack_deadline_ms: int = 150
    latency_p50_budget_s: float = 2.5
    wobble_playout_delay_s: float = 0.2

    # Behavior.
    half_duplex: bool = True  # open-mic barge-in off by default (no AEC on PC).
    max_proactive_per_n_turns: int = 3
    idle_curiosity_s: float = 20.0
    idle_glance_s: float = 180.0  # motion: mean gap between idle glances/emotes during silence.

    # Vision (Phase 5, decision 18): client-side face tracking that feeds gaze only.
    vision_enabled: bool = True  # ROCKY_VISION_ENABLED=0 -> DoA-only (reversal path).
    # Rocky greets first on wake (official conversation app pattern). ROCKY_WAKE_GREETING=0 disables.
    wake_greeting_enabled: bool = True
    vision_backend: str = "fake"  # "reachy" | "fake"
    face_detector: str = "fake"  # "daemon" | "haar" | "fake"
    vision_model: str = "qwen2.5vl:7b"  # look-around tool model (consumed in Phase 6).
    gaze_h_fov_deg: float = 70.0  # camera horizontal FoV mapped to yaw offset.
    gaze_v_fov_deg: float = 42.0  # camera vertical FoV mapped to pitch offset.

    # UI.
    ui_host: str = "0.0.0.0"
    ui_port: int = 8042

    # Storage.
    home_dir: Path = Field(default_factory=rocky_home)

    @property
    def facts_path(self) -> Path:
        return self.home_dir / "facts.jsonl"

    @property
    def open_questions_path(self) -> Path:
        return self.home_dir / "open_questions.jsonl"

    @property
    def sessions_path(self) -> Path:
        return self.home_dir / "sessions.jsonl"

    def resolved_embed_url(self) -> str:
        """Base URL for the image embedder: ROCKY_EMBED_URL if set, else the speech server."""
        return self.embed_url or self.speech_base_url


def load_settings(**overrides: object) -> Settings:
    """Construct Settings, applying explicit overrides (used by tests).

    A home_dir override also redirects the .env file to that home, so tests are
    hermetic against the real user's ~/.rocky_mini/.env (which may select the
    ollama backend on a machine where the LoRA has shipped).
    """
    if "home_dir" in overrides and "_env_file" not in overrides:
        overrides["_env_file"] = str(Path(str(overrides["home_dir"])) / ".env")
    return Settings(**overrides)
