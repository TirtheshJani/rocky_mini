"""The four tools Rocky can call, pydantic-validated.

A 7B is a flaky tool-caller, so every tool-call payload is validated against a pydantic
model before it runs. Malformed calls raise ToolCallError; the LLM layer retries once
with format=json. Handlers are injected (not imported) so this module stays free of the
memory/motion layers and is unit-testable on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ValidationError


class RememberFactArgs(BaseModel):
    category: str
    fact: str
    source_quote: str


class NoteOpenQuestionArgs(BaseModel):
    question: str


class ConfirmFactArgs(BaseModel):
    id: str


class SetSleepWatchArgs(BaseModel):
    on: bool


# Sorted by name (the plan requires deterministic tool ordering).
TOOL_MODELS: dict[str, type[BaseModel]] = {
    "confirm_fact": ConfirmFactArgs,
    "note_open_question": NoteOpenQuestionArgs,
    "remember_fact": RememberFactArgs,
    "set_sleep_watch": SetSleepWatchArgs,
}

_TOOL_DESCRIPTIONS = {
    "confirm_fact": "Mark a previously-heard fact as confirmed once TJ restates it.",
    "note_open_question": "Record something Rocky is curious about and wants to ask later.",
    "remember_fact": "Store a durable Earth fact TJ just taught, with the source quote.",
    "set_sleep_watch": "Enter or leave sleep-watch mode (watch over TJ while he rests).",
}


def tool_specs() -> list[dict]:
    """OpenAI-compatible tools list for Ollama's `tools` param. Sorted by name."""
    specs = []
    for name in sorted(TOOL_MODELS):
        model = TOOL_MODELS[name]
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _TOOL_DESCRIPTIONS[name],
                    "parameters": model.model_json_schema(),
                },
            }
        )
    return specs


class ToolCallError(ValueError):
    """Raised when a tool call is unknown or its arguments fail validation."""


def parse_tool_call(name: str, arguments: str | dict) -> BaseModel:
    """Validate a tool call. Raises ToolCallError on unknown tool or bad args."""
    if name not in TOOL_MODELS:
        raise ToolCallError(f"unknown tool: {name!r}")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolCallError(f"tool {name} arguments are not valid JSON: {exc}") from exc
    try:
        return TOOL_MODELS[name].model_validate(arguments)
    except ValidationError as exc:
        raise ToolCallError(f"tool {name} arguments failed validation: {exc}") from exc


Handler = Callable[[BaseModel], str]


@dataclass
class ToolExecutor:
    """Dispatches validated tool calls to injected handlers.

    Each handler takes the validated args model and returns a short result string
    (e.g. a new fact id) that is fed back to the model as the tool result.
    """

    handlers: dict[str, Handler]

    def execute(self, name: str, arguments: str | dict) -> str:
        args = parse_tool_call(name, arguments)
        handler = self.handlers.get(name)
        if handler is None:
            raise ToolCallError(f"no handler registered for tool {name!r}")
        return handler(args)
