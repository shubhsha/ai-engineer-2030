"""Chapter 28: assemble a context window to a token budget."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ai_engineer.llm.model import JSON, estimate_tokens, user
from ai_engineer.llm.prompt import tagged

Counter = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class Item:
    name: str
    text: str
    priority: int = 0
    pinned: bool = False


@dataclass(frozen=True, slots=True)
class Assembled:
    text: str
    tokens: int
    included: tuple[str, ...]
    dropped: tuple[str, ...]


class ContextOverflow(ValueError):
    """The pinned items alone do not fit the budget."""


def assemble(
    items: Sequence[Item], budget: int, *, count: Counter = estimate_tokens
) -> Assembled:
    """Fit ``items`` into ``budget`` tokens, highest priority first.

    Pinned items always go in. The rest are admitted by priority while
    they fit, then emitted in their original order, so the stable parts
    stay at the front where a prompt cache can reuse them. Whatever was
    left out is reported, never silently lost.
    """
    if len({item.name for item in items}) != len(items):
        raise ValueError("item names must be unique")
    cost = {item.name: count(tagged(item.name, item.text)) for item in items}
    used = sum(cost[item.name] for item in items if item.pinned)
    if used > budget:
        raise ContextOverflow(f"pinned items need {used} of {budget} tokens")
    keep = {item.name for item in items if item.pinned}
    optional = [item for item in items if not item.pinned]
    for item in sorted(optional, key=lambda item: -item.priority):
        if used + cost[item.name] <= budget:
            keep.add(item.name)
            used += cost[item.name]
    chosen = [item for item in items if item.name in keep]
    return Assembled(
        text="\n\n".join(tagged(item.name, item.text) for item in chosen),
        tokens=used,
        included=tuple(item.name for item in chosen),
        dropped=tuple(item.name for item in items if item.name not in keep),
    )


def _opens_turn(message: JSON) -> bool:
    """A user message that is not carrying tool results."""
    content = message["content"]
    return message["role"] == "user" and (
        isinstance(content, str)
        or all(block.get("type") != "tool_result" for block in content)
    )


def compact_history(
    messages: Sequence[JSON],
    *,
    keep_last: int,
    summarize: Callable[[Sequence[JSON]], str],
) -> list[JSON]:
    """Replace all but the last ``keep_last`` messages with a summary.

    The cut moves earlier until the kept part starts with a fresh user
    turn: a tool result separated from the call it answers is rejected
    by the API, and meaningless to the model.
    """
    cut = len(messages) - keep_last
    while cut > 0 and not _opens_turn(messages[cut]):
        cut -= 1
    if cut <= 0:
        return list(messages)
    summary = summarize(messages[:cut])
    return [user(tagged("summary", summary)), *messages[cut:]]
