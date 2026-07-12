"""rocky_mini - Rocky (Project Hail Mary) conversation app for Reachy Mini.

A curious Eridian engineer who learns about Earth, remembers across sessions, and
speaks with chords and a ring-modulated alien voice. Fully local, zero paid-API cost.

The package is layered so the software core runs and tests green with no robot, no
Ollama, and no GPU. External services (LLM, STT, TTS, robot SDK) sit behind Protocols
with Fakes; real clients are imported lazily and only on the hardware path.
"""

__version__ = "0.1.0"
