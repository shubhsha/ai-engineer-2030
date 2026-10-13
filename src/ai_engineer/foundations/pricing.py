"""Chapter 12: a pricing rule, after refactoring.

The legacy version lives in ``tests/foundations/legacy_pricing.py``; a
characterization test pins the two together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from ai_engineer.foundations.abstractions import Money


@dataclass(frozen=True, slots=True)
class Line:
    sku: str
    unit_price: Money
    quantity: int


@dataclass(frozen=True, slots=True)
class Policy:
    bulk_threshold: int = 10  # units of one SKU on one line
    bulk_discount: Decimal = Decimal("0.05")
    loyalty_discount: Decimal = Decimal("0.03")
    tax_rate: Decimal = Decimal("0.08")


DEFAULT_POLICY = Policy()


def line_total(line: Line, policy: Policy = DEFAULT_POLICY) -> Money:
    gross = line.unit_price.times(line.quantity)
    if line.quantity >= policy.bulk_threshold:
        return gross.times(1 - policy.bulk_discount)
    return gross


def quote_total(
    lines: Sequence[Line], *, loyal: bool, policy: Policy = DEFAULT_POLICY
) -> Money:
    """Subtotal, then loyalty discount, then tax. Each step is rounded."""
    currency = lines[0].unit_price.currency if lines else "USD"
    subtotal = sum(
        (line_total(ln, policy) for ln in lines), Money.zero(currency)
    )
    if loyal:
        subtotal = subtotal.times(1 - policy.loyalty_discount)
    return subtotal.times(1 + policy.tax_rate)
