"""Chapter 30: the agent loop, with a step limit and a budget."""

from __future__ import annotations

from dataclasses import dataclass

from ai_engineer.llm.budget import Budget, BudgetExceeded
from ai_engineer.llm.model import (
    JSON,
    Model,
    Request,
    Usage,
    assistant,
    user,
)
from ai_engineer.llm.tools import Executor, ToolRegistry


@dataclass(frozen=True, slots=True)
class AgentResult:
    answer: str
    stop: str  # end_turn, max_tokens, refusal, max_steps or budget
    steps: int
    usage: Usage
    messages: tuple[JSON, ...]


def run_agent(
    model: Model,
    task: str,
    tools: ToolRegistry,
    *,
    system: str = "",
    max_steps: int = 8,
    budget: Budget | None = None,
    execute: Executor | None = None,
) -> AgentResult:
    """Call the model, run the tools it asks for, repeat until it answers.

    Stops at a final answer, after ``max_steps`` model calls or when the
    budget runs out, and always reports which. Tool failures go back to
    the model as error results so it can recover. ``execute`` wraps the
    registry, for example with the policy hooks of Chapter 34.
    """
    execute = execute or tools.call
    messages: list[JSON] = [user(task)]
    usage = Usage()
    for step in range(1, max_steps + 1):
        request = Request(system, tuple(messages), tools.specs())
        reply = model.complete(request)
        usage += reply.usage
        messages.append(assistant(reply))
        try:
            if budget is not None:
                budget.charge(reply.usage)
        except BudgetExceeded:
            return AgentResult("", "budget", step, usage, tuple(messages))
        if reply.stop_reason != "tool_use":
            stop = reply.stop_reason
            return AgentResult(reply.text, stop, step, usage, tuple(messages))
        messages.append(user([execute(call) for call in reply.tool_calls]))
    return AgentResult("", "max_steps", max_steps, usage, tuple(messages))
