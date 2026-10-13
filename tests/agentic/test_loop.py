from decimal import Decimal
from functools import partial

from ai_engineer.agents.loop import run_agent
from ai_engineer.claude.catalog import ModelInfo
from ai_engineer.claude.cost import call_cost
from ai_engineer.llm.budget import Budget
from ai_engineer.llm.model import ScriptedModel, tool_use
from ai_engineer.llm.tools import Tool, ToolError, ToolRegistry

ORDER = {
    "type": "object",
    "properties": {"id": {"type": "integer"}},
    "required": ["id"],
}


def lookup_order(id):
    if id == 99:
        raise ToolError("No order 99. Ask the customer for the order id.")
    return {"id": id, "status": "shipped"}


TOOLS = ToolRegistry(
    [Tool("lookup_order", "Look up an order by id.", ORDER, lookup_order)]
)


def test_the_agent_calls_a_tool_then_answers():
    model = ScriptedModel(
        [tool_use("t1", "lookup_order", id=7), "Order 7 has shipped."]
    )
    result = run_agent(model, "Where is order 7?", TOOLS)
    assert result.answer == "Order 7 has shipped."
    assert (result.stop, result.steps) == ("end_turn", 2)
    [block] = result.messages[2]["content"]
    assert block["content"] == '{"id": 7, "status": "shipped"}'
    assert model.requests[0].tools == TOOLS.specs()


def test_a_tool_error_goes_back_to_the_model():
    model = ScriptedModel(
        [tool_use("t1", "lookup_order", id=99), "I could not find order 99."]
    )
    result = run_agent(model, "Where is order 99?", TOOLS)
    [block] = result.messages[2]["content"]
    assert block["is_error"] is True
    assert block["content"].startswith("No order 99.")
    assert result.answer == "I could not find order 99."


def test_a_step_limit_stops_a_loop_that_never_answers():
    model = ScriptedModel(default=tool_use("t1", "lookup_order", id=7))
    result = run_agent(model, "Where is order 7?", TOOLS, max_steps=3)
    assert (result.stop, result.steps, result.answer) == ("max_steps", 3, "")
    assert len(model.requests) == 3


def test_a_budget_stops_the_loop_when_the_money_runs_out():
    haiku = ModelInfo("haiku", "small", 200_000, Decimal(1), Decimal(5))
    budget = Budget(Decimal("0.001"), partial(call_cost, haiku))
    model = ScriptedModel(default=tool_use("t1", "lookup_order", id=7))
    result = run_agent(model, "Where is order 7?", TOOLS, budget=budget)
    # Each step costs $0.00035 (200 tokens in, 30 out): the third one
    # takes the total past the $0.001 limit.
    assert (result.stop, result.steps) == ("budget", 3)
    assert budget.spent == Decimal("0.00105")
