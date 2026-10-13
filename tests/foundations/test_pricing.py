from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai_engineer.foundations.abstractions import Money
from ai_engineer.foundations.pricing import Line, quote_total
from tests.foundations.legacy_pricing import calc

prices = st.decimals(
    min_value=Decimal("0.01"), max_value=Decimal(500), places=2
)
lines = st.lists(
    st.tuples(prices, st.integers(min_value=1, max_value=40)),
    min_size=1,
    max_size=8,
)


def test_known_quote():
    order = [
        Line("a", Money(Decimal("19.99")), 2),
        Line("b", Money(Decimal("5.00")), 10),
    ]
    # (39.98 + 47.50) * 0.97 = 84.8556 -> 84.86; * 1.08 = 91.6488 -> 91.65
    assert quote_total(order, loyal=True).amount == Decimal("91.65")


# region characterization
@given(lines, st.booleans())
def test_refactor_matches_legacy_within_rounding(raw, loyal):
    legacy = calc([("sku", float(p), q) for p, q in raw], {"loyal": loyal})
    order = [Line("sku", Money(p), q) for p, q in raw]
    new = quote_total(order, loyal=loyal).amount
    # The legacy code rounds once, at the end, in binary floating point.
    # The new code rounds each line to cents, as an invoice prints it.
    # The drift is bounded by one cent per rounding step.
    steps = len(raw) + 3
    assert abs(new - Decimal(str(legacy))) <= Decimal("0.01") * steps


# endregion


# region basis_table
BASIS = [  # (sku, price, quantity) lines, loyal, total worked by hand
    ([], False, "0.00"),
    ([("a", "10.00", 2)], False, "21.60"),
    ([("a", "10.00", 10)], False, "102.60"),
    ([("a", "10.00", 2)], True, "20.95"),
]


@pytest.mark.parametrize(("items", "loyal", "expected"), BASIS)
def test_basis_paths(items, loyal, expected):
    """One order per independent path through the legacy function."""
    legacy = calc([(s, float(p), q) for s, p, q in items], {"loyal": loyal})
    order = [Line(s, Money(Decimal(p)), q) for s, p, q in items]
    new = quote_total(order, loyal=loyal).amount
    assert Decimal(str(legacy)) == new == Decimal(expected)


# endregion


# region documented_difference
DECIDED = [  # lines, loyal, legacy total, new total
    ([("a", "19.99", 2), ("b", "5.00", 10)], True, 91.64, "91.65"),
    ([("a", "0.99", 10)], False, 10.16, "10.15"),
]


@pytest.mark.parametrize(("items", "loyal", "old", "new"), DECIDED)
def test_decided_difference(items, loyal, old, new):
    """The decision to round each step, pinned in both directions."""
    legacy = calc([(s, float(p), q) for s, p, q in items], {"loyal": loyal})
    order = [Line(s, Money(Decimal(p)), q) for s, p, q in items]
    assert legacy == old
    assert quote_total(order, loyal=loyal).amount == Decimal(new)


# endregion
