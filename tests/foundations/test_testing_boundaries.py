"""Chapter 11: boundary tests, placed by hand on the edges.

``line_total`` changes its rule at ten units, and the commit-message
check changes its verdict at 72 characters. Each case below sits on an
edge, because that is where a ``>`` written as ``>=`` hides.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

from ai_engineer.foundations.abstractions import Money
from ai_engineer.foundations.commitmsg import check_commit_message, main
from ai_engineer.foundations.pricing import Line, line_total


# region bulk-boundary
@pytest.mark.parametrize(
    ("quantity", "total"),
    [
        (1, "20.00"),  # the smallest order: 1 x 20.00
        (9, "180.00"),  # one below the threshold: 9 x 20.00
        (10, "190.00"),  # on it: 200.00 less 5 percent
        (11, "209.00"),  # one above: 220.00 less 5 percent
    ],
)
def test_the_bulk_discount_starts_at_exactly_ten_units(
    quantity: int, total: str
) -> None:
    line = Line("sku", Money(Decimal("20.00")), quantity)
    assert line_total(line) == Money(Decimal(total))


# endregion


# region header-limit
def header_of(length: int) -> str:
    return "fix: " + "x" * (length - len("fix: "))


def test_a_header_of_exactly_72_characters_is_accepted() -> None:
    assert check_commit_message(header_of(72)) == []


def test_a_header_of_73_characters_is_rejected() -> None:
    problems = check_commit_message(header_of(73))
    assert problems == ["header has 73 characters; the limit is 72"]


# endregion


def test_main_reads_its_path_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text("wip\n")
    monkeypatch.setattr(sys, "argv", ["commit-msg", str(msg)])
    assert main() == 1


def test_main_prints_each_problem_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text("fix: add x.\n")
    main([str(msg)])
    err = capsys.readouterr().err
    assert err == "commit-msg: subject should not end with a period\n"
