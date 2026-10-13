"""Chapter 30: the model boundary, one small protocol and a fake.

Everything in Volume III talks to a language model through ``Model``.
Messages use the Messages API wire format (a role, and either a string
or a list of content blocks), so the adapter for the real API stays
thin. ``ScriptedModel`` replays canned replies and records requests,
which keeps agent tests offline, fast and deterministic.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

JSON = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: JSON


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts as the API reports them. Cache reads and writes are
    counted apart from ``input_tokens``, so the four never overlap."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(frozen=True, slots=True)
class Reply:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True, slots=True)
class Request:
    system: str
    messages: tuple[JSON, ...]
    tools: tuple[JSON, ...] = ()
    max_tokens: int = 1024
    output_schema: JSON | None = None


class Model(Protocol):
    """Anything that turns a request into a reply."""

    def complete(self, request: Request) -> Reply: ...


def user(content: str | list[JSON]) -> JSON:
    return {"role": "user", "content": content}


def assistant(reply: Reply) -> JSON:
    """The assistant turn to append to the history, tool calls included."""
    blocks: list[JSON] = []
    if reply.text:
        blocks.append({"type": "text", "text": reply.text})
    blocks += [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
        for c in reply.tool_calls
    ]
    return {"role": "assistant", "content": blocks}


def tool_result(call_id: str, content: str, *, is_error: bool = False) -> JSON:
    block: JSON = {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


def ask(
    model: Model, prompt: str, *, system: str = "", max_tokens: int = 1024
) -> Reply:
    """One user turn, one reply: the simplest call there is."""
    return model.complete(
        Request(system, (user(prompt),), max_tokens=max_tokens)
    )


def estimate_tokens(text: str) -> int:
    """About four characters per token for English prose.

    Good enough for planning a budget, never for billing: bill from the
    usage the API reports, and count with the provider's token counter
    when a limit is tight.
    """
    return max(1, -(-len(text) // 4))


Script = Reply | str | Callable[[Request], Reply | str]


class ScriptedModel:
    """A test double for models: replays ``script`` in order.

    Each entry is a ``Reply``, a string (an ``end_turn`` reply) or a
    function of the request. ``default`` answers once the script runs
    out. Requests are recorded, and usage is estimated from the text,
    so cost and budget code can be tested too. Safe to share between
    threads.
    """

    def __init__(
        self, script: Iterable[Script] = (), *, default: Script | None = None
    ) -> None:
        self._script = list(script)
        self._default = default
        self._lock = threading.Lock()
        self.requests: list[Request] = []

    def complete(self, request: Request) -> Reply:
        with self._lock:
            self.requests.append(request)
            if self._script:
                entry = self._script.pop(0)
            elif self._default is not None:
                entry = self._default
            else:
                raise AssertionError("ScriptedModel ran out of replies")
        out = entry(request) if callable(entry) else entry
        if isinstance(out, Reply):
            return out
        sent = request.system + json.dumps(request.messages + request.tools)
        usage = Usage(estimate_tokens(sent), estimate_tokens(out))
        return Reply(out, usage=usage)


def tool_use(call_id: str, name: str, **arguments: Any) -> Reply:
    """A scripted reply that asks for one tool call."""
    return Reply(
        tool_calls=(ToolCall(call_id, name, arguments),),
        stop_reason="tool_use",
        usage=Usage(200, 30),
    )
