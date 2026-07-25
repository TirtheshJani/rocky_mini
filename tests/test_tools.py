"""Tool tests: pydantic validation, unknown-tool + bad-args errors, dispatch, specs."""

import pytest

from rocky_mini.brain.tools import (
    LookAroundArgs,
    RememberFactArgs,
    ToolCallError,
    ToolExecutor,
    parse_tool_call,
    tool_specs,
)


def test_parse_valid_remember_fact_dict():
    args = parse_tool_call(
        "remember_fact",
        {"category": "food", "fact": "taco is food", "source_quote": "a taco is food"},
    )
    assert isinstance(args, RememberFactArgs)
    assert args.category == "food"


def test_parse_valid_from_json_string():
    args = parse_tool_call("set_sleep_watch", '{"on": true}')
    assert args.on is True


def test_unknown_tool_raises():
    with pytest.raises(ToolCallError):
        parse_tool_call("delete_everything", {})


def test_missing_field_raises():
    with pytest.raises(ToolCallError):
        parse_tool_call("remember_fact", {"category": "food"})


def test_bad_json_raises():
    with pytest.raises(ToolCallError):
        parse_tool_call("note_open_question", "{not json")


def test_tool_specs_are_the_six_sorted():
    specs = tool_specs()
    names = [s["function"]["name"] for s in specs]
    assert names == sorted(names)
    assert names == [
        "confirm_fact",
        "look_around",
        "note_open_question",
        "remember_fact",
        "set_sleep_watch",
        "teach_object",
    ]


def test_parse_teach_object_requires_name():
    args = parse_tool_call("teach_object", {"name": "coffee mug"})
    assert args.name == "coffee mug"
    with pytest.raises(ToolCallError):
        parse_tool_call("teach_object", {})


def test_parse_look_around_empty_ok():
    # Default question is "" so an empty payload validates.
    args = parse_tool_call("look_around", "{}")
    assert isinstance(args, LookAroundArgs)
    assert args.question == ""


def test_parse_look_around_with_question():
    args = parse_tool_call("look_around", {"question": "what do you see"})
    assert args.question == "what do you see"


def test_look_around_bad_payload_type_raises():
    with pytest.raises(ToolCallError):
        parse_tool_call("look_around", {"question": ["not", "a", "string"]})


def test_executor_dispatches_to_handler():
    calls = []

    def handle_remember(args):
        calls.append(args.fact)
        return "fact-1"

    ex = ToolExecutor(handlers={"remember_fact": handle_remember})
    result = ex.execute(
        "remember_fact",
        {"category": "food", "fact": "taco is food", "source_quote": "q"},
    )
    assert result == "fact-1"
    assert calls == ["taco is food"]


def test_executor_missing_handler_raises():
    ex = ToolExecutor(handlers={})
    with pytest.raises(ToolCallError):
        ex.execute("confirm_fact", {"id": "x"})
