"""Chapter 61: the decision matrix, its flip margins, the order check."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal as D

import pytest

from ai_engineer.enterprise.asof import OPEN_END, as_of_totals
from ai_engineer.enterprise.warehouse import SqliteWarehouse
from ai_engineer.leadership.decisions import (
    Criterion,
    Option,
    flip_margin,
    order_problems,
    score_matrix,
)

PORT, NAMED, KEEP = "Port line by line", "Named functions", "Keep in BW"
OPTIONS = [Option(PORT), Option(NAMED), Option(KEEP)]
CRITERIA = [
    Criterion("need", D("0.30")),
    Criterion("numbers", D("0.25")),
    Criterion("ci", D("0.20")),
    Criterion("speed", D("0.15")),
    Criterion("cost", D("0.10")),
]


# region composite_decision
def composite_scores() -> dict[tuple[str, str], int]:
    """Three options, five criteria, scores 1 to 5."""
    rows = {
        PORT: (3, 4, 1, 3, 3),
        NAMED: (4, 4, 4, 3, 4),
        KEEP: (2, 5, 3, 5, 2),
    }
    return {
        (name, c.name): s
        for name, row in rows.items()
        for c, s in zip(CRITERIA, row, strict=True)
    }


def test_the_named_functions_option_wins() -> None:
    ranked = score_matrix(OPTIONS, CRITERIA, composite_scores())
    assert [(r.option, r.total) for r in ranked] == [
        (NAMED, D("3.85")),
        (KEEP, D("3.40")),
        (PORT, D("2.85")),
    ]


def test_flip_margins_name_the_weights_that_matter() -> None:
    margins = flip_margin(OPTIONS, CRITERIA, composite_scores())
    assert margins == {
        "need": D("-0.225"),
        "numbers": D("0.45"),
        "ci": None,
        "speed": D("0.225"),
        "cost": None,
    }


# endregion composite_decision


def test_a_flip_margin_really_changes_the_winner() -> None:
    scores = composite_scores()
    margin = flip_margin(OPTIONS, CRITERIA, scores)["speed"]
    assert margin is not None
    bumped = [
        Criterion(c.name, c.weight + margin + D("0.01"))
        if c.name == "speed"
        else c
        for c in CRITERIA
    ]
    assert score_matrix(OPTIONS, bumped, scores)[0].option == KEEP


def test_a_missing_score_is_refused() -> None:
    scores = composite_scores()
    del scores[(KEEP, "cost")]
    with pytest.raises(ValueError, match="missing scores"):
        score_matrix(OPTIONS, CRITERIA, scores)


# region order_check
def test_scores_before_the_criteria_are_flagged() -> None:
    fixed = datetime(2030, 3, 4, 10, 0)
    scored = [datetime(2030, 3, 4, 9, 30), datetime(2030, 3, 4, 11, 0)]
    assert order_problems(fixed, scored) == [
        "score 1 at 2030-03-04T09:30:00 is not after the criteria"
    ]
    assert order_problems(fixed, scored[1:]) == []


# endregion order_check


def test_time_dependent_constraint_as_of_join() -> None:
    """Valid-from and valid-to dates become a type-2 dimension."""
    w = SqliteWarehouse()
    c = w.connection
    c.execute("CREATE TABLE fact (plant TEXT, posted TEXT, amount TEXT)")
    c.execute(
        "CREATE TABLE dim (plant TEXT, region TEXT, "
        "valid_from TEXT, valid_to TEXT)"
    )
    c.executemany(
        "INSERT INTO dim VALUES (?, ?, ?, ?)",
        [
            ("P1", "NORTH", "2020-01-01", "2030-04-01"),
            ("P1", "SOUTH", "2030-04-01", OPEN_END),
        ],
    )
    c.executemany(
        "INSERT INTO fact VALUES ('P1', ?, ?)",
        [("2030-03-31", "10.00"), ("2030-04-01", "5.00")],
    )
    got = as_of_totals(
        w,
        "fact",
        "dim",
        key="plant",
        date="posted",
        attr="region",
        measure="amount",
    )
    assert got == {"NORTH": D("10.00"), "SOUTH": D("5.00")}
