"""Chapter 37: what a call costs, and when a cache pays for itself.

Multipliers on the input price, from the provider's pricing page
(checked 2026-10-08): writing a prompt-cache entry costs 1.25 times the
input price with a five-minute lifetime and 2 times with one hour;
reading it costs the model's ``cache_read_multiplier``. The Batch API
halves everything, caching included. A model priced by prompt length
charges its higher prices for the whole call once the prompt passes the
step; cached tokens count toward the length, because a cached prefix is
still part of the prompt.
"""

from __future__ import annotations

from decimal import Decimal

from ai_engineer.claude.catalog import ModelInfo
from ai_engineer.llm.model import Usage

MTOK = Decimal(1_000_000)
CACHE_WRITE = {"5m": Decimal("1.25"), "1h": Decimal(2)}
BATCH = Decimal("0.5")


def call_cost(
    model: ModelInfo, usage: Usage, *, ttl: str = "5m", batch: bool = False
) -> Decimal:
    """Dollars for one call, from the usage the API reported."""
    prompt = (
        usage.input_tokens + usage.cache_write_tokens + usage.cache_read_tokens
    )
    input_price, output_price = model.prices(prompt)
    per_input = input_price / MTOK
    cost = (
        usage.input_tokens * per_input
        + usage.cache_write_tokens * per_input * CACHE_WRITE[ttl]
        + usage.cache_read_tokens * per_input * model.cache_read_multiplier
        + usage.output_tokens * output_price / MTOK
    )
    return cost * BATCH if batch else cost


def cache_break_even(model: ModelInfo, ttl: str = "5m") -> int:
    """The fewest calls sharing a prefix for which caching it is cheaper.

    Caching costs one write and then a read per call; not caching costs
    the full input price every time.
    """
    write, read = CACHE_WRITE[ttl], model.cache_read_multiplier
    calls = 1
    while write + (calls - 1) * read >= calls:
        calls += 1
    return calls
