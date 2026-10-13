"""Chapter 37: the Claude API behind the ``Model`` protocol.

The SDK (the ``claude`` extra) is imported only when a real client is
created, so the package and every test run without it or a network.
``to_params`` and ``to_reply`` are pure functions, tested with fakes.
"""

from __future__ import annotations

from typing import Any

from ai_engineer.llm.model import JSON, Reply, Request, ToolCall, Usage


class ClaudeModel:
    """A ``Model`` backed by the Messages API.

    ``cache_ttl`` turns on automatic prompt caching ("5m" or "1h"), and
    ``effort`` trades depth of thinking for speed and cost ("low" to
    "max"). The SDK retries rate limits and server errors with backoff.
    """

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        cache_ttl: str | None = "5m",
        effort: str | None = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model, self.cache_ttl, self.effort = model, cache_ttl, effort
        self._client = client

    def complete(self, request: Request) -> Reply:
        params = to_params(
            self.model, request, cache_ttl=self.cache_ttl, effort=self.effort
        )
        return to_reply(self._client.messages.create(**params))


def to_params(
    model: str,
    request: Request,
    *,
    cache_ttl: str | None = "5m",
    effort: str | None = None,
) -> JSON:
    """Keyword arguments for ``client.messages.create``."""
    params: JSON = {
        "model": model,
        "max_tokens": request.max_tokens,
        "messages": list(request.messages),
    }
    if request.system:
        params["system"] = request.system
    if request.tools:
        params["tools"] = list(request.tools)
    if cache_ttl:
        params["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}
    output: JSON = {}
    if effort:
        output["effort"] = effort
    if request.output_schema is not None:
        schema = request.output_schema
        output["format"] = {"type": "json_schema", "schema": schema}
    if output:
        params["output_config"] = output
    return params


def to_reply(message: Any) -> Reply:
    """Convert an SDK ``Message`` into a ``Reply``."""
    texts, calls = [], []
    for block in message.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(block.id, block.name, dict(block.input)))
    u = message.usage
    usage = Usage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=u.cache_read_input_tokens or 0,
        cache_write_tokens=u.cache_creation_input_tokens or 0,
    )
    stop = message.stop_reason or "end_turn"
    return Reply("".join(texts), tuple(calls), stop, usage)
