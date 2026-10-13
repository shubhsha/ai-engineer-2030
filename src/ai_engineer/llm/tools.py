"""Chapter 30: tools a model can call, with schemas and honest errors."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ai_engineer.llm.model import JSON, ToolCall, tool_result
from ai_engineer.llm.structured import validate

log = logging.getLogger(__name__)
Executor = Callable[[ToolCall], JSON]


class ToolError(Exception):
    """An expected failure, with a message the model can act on."""


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    input_schema: JSON
    fn: Callable[..., Any]
    read_only: bool = True

    def spec(self) -> JSON:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """The tools an agent may use, and the only way it can use them."""

    def __init__(self, tools: Iterable[Tool], *, max_chars: int = 4000):
        self._tools = {tool.name: tool for tool in tools}
        self.max_chars = max_chars

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def specs(self) -> tuple[JSON, ...]:
        """Definitions in a fixed order. They sit at the front of every
        prompt, so a stable order keeps the prompt cache valid."""
        return tuple(self._tools[name].spec() for name in sorted(self._tools))

    def call(self, call: ToolCall) -> JSON:
        """Run one call and return a ``tool_result`` block. Never raises:
        every failure goes back to the model as an error it can read."""
        tool = self._tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self._tools))
            message = f"Unknown tool {call.name!r}. Available: {known}."
            return tool_result(call.id, message, is_error=True)
        if errors := validate(call.input, tool.input_schema):
            message = "Invalid input: " + "; ".join(errors)
            return tool_result(call.id, message, is_error=True)
        try:
            output = tool.fn(**call.input)
        except ToolError as exc:
            return tool_result(call.id, str(exc), is_error=True)
        except Exception:
            log.exception("tool %s failed", call.name)
            message = f"{call.name} failed internally. Do not retry it."
            return tool_result(call.id, message, is_error=True)
        return tool_result(call.id, self._render(output))

    def _render(self, output: Any) -> str:
        text = output if isinstance(output, str) else json.dumps(output)
        if len(text) <= self.max_chars:
            return text
        cut = len(text) - self.max_chars
        return (
            f"{text[: self.max_chars]}\n[{cut} more characters cut. "
            "Narrow the request, for example with a filter or a limit.]"
        )
