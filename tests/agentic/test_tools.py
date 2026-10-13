import logging

from ai_engineer.llm.model import ToolCall
from ai_engineer.llm.tools import Tool, ToolError, ToolRegistry

ORDER = {
    "type": "object",
    "properties": {"id": {"type": "integer"}},
    "required": ["id"],
    "additionalProperties": False,
}


def lookup_order(id):
    if id == 99:
        raise ToolError("No order 99. Check the id with the customer.")
    if id == 13:
        raise RuntimeError("connection reset by peer")
    return {"id": id, "status": "shipped"}


REGISTRY = ToolRegistry(
    [
        Tool("lookup_order", "Look up an order by id.", ORDER, lookup_order),
        Tool("cancel_order", "Cancel an order.", ORDER, print, False),
        Tool("export", "Export everything.", {}, lambda: "x" * 500),
    ],
    max_chars=200,
)


def call(name, **arguments):
    return REGISTRY.call(ToolCall("t1", name, arguments))


def test_specs_come_in_a_stable_order():
    names = [spec["name"] for spec in REGISTRY.specs()]
    assert names == ["cancel_order", "export", "lookup_order"]


def test_a_successful_call_returns_json_text():
    result = call("lookup_order", id=7)
    assert result["content"] == '{"id": 7, "status": "shipped"}'
    assert "is_error" not in result


def test_an_unknown_tool_lists_the_real_ones():
    result = call("delete_everything")
    assert result["is_error"]
    assert "Available: cancel_order, export, lookup_order" in result["content"]


def test_invalid_input_is_explained():
    result = call("lookup_order", id="seven")
    assert result["content"] == (
        "Invalid input: $.id: expected integer, got str"
    )


def test_expected_failures_pass_their_message_through():
    assert call("lookup_order", id=99)["content"].startswith("No order 99.")


def test_unexpected_failures_are_logged_not_leaked(caplog):
    with caplog.at_level(logging.ERROR):
        result = call("lookup_order", id=13)
    assert result["content"] == (
        "lookup_order failed internally. Do not retry it."
    )
    assert "connection reset" not in result["content"]
    assert "connection reset" in caplog.text


def test_long_results_are_cut_with_advice():
    content = call("export")["content"]
    assert content.startswith("x" * 200)
    assert "[300 more characters cut. Narrow the request" in content
