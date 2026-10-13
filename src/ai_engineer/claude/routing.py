"""Chapter 37: send each call to the cheapest model that can do it."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from ai_engineer.claude.catalog import TIERS, Catalog, ModelInfo
from ai_engineer.llm.model import Model, Reply, Request, Usage


def cheapest_fit(
    catalog: Catalog, *, prompt_tokens: int, min_tier: str = "small"
) -> ModelInfo:
    """The cheapest model at or above ``min_tier`` whose window fits."""
    floor = TIERS.index(min_tier)
    fits = [
        m
        for m in catalog.models
        if TIERS.index(m.tier) >= floor and m.context_window >= prompt_tokens
    ]
    if not fits:
        raise LookupError(f"no model fits {prompt_tokens} prompt tokens")
    return min(fits, key=lambda m: m.prices(prompt_tokens))


class NoAcceptableReply(RuntimeError):
    def __init__(self, tried: Sequence[str]) -> None:
        super().__init__(f"no acceptable reply from {', '.join(tried)}")
        self.tried = tuple(tried)


@dataclass(frozen=True, slots=True)
class Cascaded:
    reply: Reply
    model: str
    tried: tuple[str, ...]
    usage: Usage = field(default_factory=Usage)


def cascade(
    models: Sequence[tuple[str, Model]],
    request: Request,
    accept: Callable[[Reply], bool],
) -> Cascaded:
    """Try models cheapest first; stop at the first acceptable reply.

    Every attempt is paid for, so a cascade saves money only when the
    cheap model usually succeeds and ``accept`` is cheap and reliable,
    such as a schema check or a test run.
    """
    tried: list[str] = []
    usage = Usage()
    for name, model in models:
        reply = model.complete(request)
        tried.append(name)
        usage += reply.usage
        if accept(reply):
            return Cascaded(reply, name, tuple(tried), usage)
    raise NoAcceptableReply(tried)


# region expected-cascade-cost
def expected_cascade_cost(
    small: Decimal, large: Decimal, pass_rate: Decimal
) -> Decimal:
    """Dollars per task: the small model always runs, the large one only
    when the check fails. ``pass_rate`` is the share the small model passes."""
    return small + (1 - pass_rate) * large


def cascade_pays(small: Decimal, large: Decimal, pass_rate: Decimal) -> bool:
    """True when the cascade is cheaper than calling the large model."""
    return expected_cascade_cost(small, large, pass_rate) < large


# endregion
