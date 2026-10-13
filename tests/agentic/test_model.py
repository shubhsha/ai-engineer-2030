import pytest

from ai_engineer.llm.model import (
    Reply,
    ScriptedModel,
    ToolCall,
    Usage,
    ask,
    assistant,
    estimate_tokens,
    tool_result,
    tool_use,
)


def test_scripted_model_replays_in_order_then_uses_the_default():
    model = ScriptedModel(["one", "two"], default="again")
    assert [ask(model, "q").text for _ in range(4)] == [
        "one",
        "two",
        "again",
        "again",
    ]
    assert len(model.requests) == 4


def test_scripted_model_fails_loudly_when_the_script_runs_out():
    model = ScriptedModel(["only"])
    ask(model, "first")
    with pytest.raises(AssertionError, match="ran out"):
        ask(model, "second")


def test_string_replies_get_estimated_usage():
    reply = ask(ScriptedModel(["x" * 40]), "hello", system="Be brief.")
    assert reply.usage.output_tokens == estimate_tokens("x" * 40) == 10
    assert reply.usage.input_tokens > 0


def test_assistant_turn_keeps_text_and_tool_calls_in_wire_format():
    reply = Reply("Checking.", (ToolCall("t1", "lookup", {"id": 7}),))
    assert assistant(reply) == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Checking."},
            {
                "type": "tool_use",
                "id": "t1",
                "name": "lookup",
                "input": {"id": 7},
            },
        ],
    }


def test_tool_results_flag_errors_only_when_they_happen():
    assert "is_error" not in tool_result("t1", "42")
    assert tool_result("t1", "boom", is_error=True)["is_error"] is True


def test_tool_use_helper_asks_for_one_call():
    reply = tool_use("t1", "lookup", id=7)
    assert reply.stop_reason == "tool_use"
    assert reply.tool_calls == (ToolCall("t1", "lookup", {"id": 7}),)


def test_usage_adds_field_by_field():
    total = Usage(10, 2, 3, 4) + Usage(1, 1, 1, 1)
    assert total == Usage(11, 3, 4, 5)
    assert total.total == 23
