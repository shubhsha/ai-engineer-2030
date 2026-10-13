"""Chapter 11: pin down what the old pricing code does, and the golden file.

The old answers were run, not reasoned out. Where the rewrite differs, the
test says so, to the cent.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from ai_engineer.foundations.abstractions import Money
from ai_engineer.foundations.pricing import Line, quote_total
from tests.foundations import legacy_pricing

# The old module stays untyped, as found; this says what calc() is.
calc = cast("Callable[..., float]", legacy_pricing.calc)
Order = list[tuple[str, str, int]]  # sku, unit price, units


# region pinned-answers
# (order, loyal, answer of calc(), answer of the rewrite). The old code
# rounds once, at the end; the rewrite rounds at every step.
RECORDED = [
    ([("a", "19.99", 2), ("b", "5.00", 10)], True, "91.64", "91.65"),
    ([("a", "2.01", 10)], False, "20.62", "20.63"),
    ([("a", "2.03", 10)], False, "20.83", "20.82"),
    ([("a", "20.00", 1)], False, "21.60", "21.60"),
]


@pytest.mark.parametrize(("raw", "loyal", "old", "new"), RECORDED)
def test_old_and_new_answers_are_pinned(
    raw: Order, loyal: bool, old: str, new: str
) -> None:
    legacy = calc([(s, float(p), q) for s, p, q in raw], {"loyal": loyal})
    order = [Line(s, Money(Decimal(p)), q) for s, p, q in raw]
    assert Decimal(str(legacy)) == Decimal(old)
    assert quote_total(order, loyal=loyal).amount == Decimal(new)


# endregion


# region golden-file
def check_golden(path: Path, actual: str) -> None:
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.write_text(actual)  # then read the diff before you commit it
    assert actual == path.read_text(), f"{path.name} changed"


def test_a_changed_output_fails_until_the_file_is_updated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = tmp_path / "quotes.txt"
    monkeypatch.setenv("UPDATE_GOLDEN", "1")
    check_golden(golden, "20.62\n")  # records the file
    monkeypatch.delenv("UPDATE_GOLDEN")
    check_golden(golden, "20.62\n")  # same output: passes
    with pytest.raises(AssertionError, match=r"quotes\.txt changed"):
        check_golden(golden, "20.63\n")  # a one-cent change is caught


# endregion


def test_a_missing_golden_file_is_a_failure_not_a_pass(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        check_golden(tmp_path / "absent.txt", "anything\n")
