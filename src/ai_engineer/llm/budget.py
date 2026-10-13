"""Chapter 30: a spending limit that stops a runaway loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from ai_engineer.llm.model import Usage


class BudgetExceeded(RuntimeError):
    def __init__(self, spent: Decimal, limit: Decimal) -> None:
        super().__init__(f"spent ${spent:.4f} of a ${limit:.4f} budget")
        self.spent, self.limit = spent, limit


@dataclass
class Budget:
    """Dollars a task may spend. ``price`` turns usage into dollars,
    for example ``functools.partial(call_cost, model_info)``."""

    limit: Decimal
    price: Callable[[Usage], Decimal]
    spent: Decimal = Decimal(0)
    calls: int = 0

    def charge(self, usage: Usage) -> Decimal:
        """Record one call and return what is left; raise once the
        total passes the limit."""
        self.calls += 1
        self.spent += self.price(usage)
        if self.spent > self.limit:
            raise BudgetExceeded(self.spent, self.limit)
        return self.limit - self.spent
