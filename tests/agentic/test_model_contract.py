"""One contract for every ``Model`` adapter, the scripted fake included.

The rest of the suite trusts ``ScriptedModel`` to behave like a real
adapter, and these tests check that it does: each adapter is built to
give the same answer and must turn it into the same ``Reply``. A new
adapter, such as one for a local runtime, is one more fixture param.
"""

from types import SimpleNamespace

import pytest

from ai_engineer.claude.adapter import ClaudeModel
from ai_engineer.llm.model import (
    Model,
    ScriptedModel,
    ToolCall,
    ask,
    tool_use,
)


def scripted(answer: str) -> Model:
    if answer == "text":
        return ScriptedModel(["Paris."])
    return ScriptedModel([tool_use("t1", "lookup", id=7)])


def claude(answer: str) -> Model:
    """The Claude adapter, with a fake client that returns ``answer``."""
    if answer == "text":
        block = SimpleNamespace(type="text", text="Paris.")
    else:
        block = SimpleNamespace(
            type="tool_use", id="t1", name="lookup", input={"id": 7}
        )
    message = SimpleNamespace(
        content=[block],
        stop_reason="end_turn" if answer == "text" else "tool_use",
        usage=SimpleNamespace(
            input_tokens=50,
            output_tokens=20,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        ),
    )

    def create(**params: object) -> SimpleNamespace:
        return message

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return ClaudeModel("claude-haiku-4-5-20251001", client=client)


@pytest.fixture(params=[scripted, claude], ids=["scripted", "claude"])
def model_for(request):
    """Every test that asks for ``model_for`` runs once per adapter."""
    return request.param


def test_a_text_answer_ends_the_turn(model_for):
    reply = ask(model_for("text"), "What is the capital of France?")
    assert reply.text == "Paris."
    assert reply.tool_calls == ()
    assert reply.stop_reason == "end_turn"


def test_a_tool_request_names_the_call_and_stops_for_it(model_for):
    reply = ask(model_for("tool"), "Where is order 7?")
    assert reply.tool_calls == (ToolCall("t1", "lookup", {"id": 7}),)
    assert reply.stop_reason == "tool_use"


@pytest.mark.parametrize("answer", ["text", "tool"])
def test_every_reply_reports_the_tokens_it_used(model_for, answer):
    usage = ask(model_for(answer), "Hello").usage
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
