"""Chapter 41: the as-of join and the validity-chain check."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from ai_engineer.enterprise.asof import (
    OPEN_END,
    as_of_sql,
    as_of_totals,
    validity_problems,
)
from ai_engineer.enterprise.warehouse import SqliteWarehouse


@pytest.fixture
def wh(rng: random.Random) -> SqliteWarehouse:
    w = SqliteWarehouse()
    c = w.connection
    c.execute("CREATE TABLE fact (material TEXT, posted TEXT, amount TEXT)")
    c.execute(
        "CREATE TABLE dim (material TEXT, grp TEXT, "
        "valid_from TEXT, valid_to TEXT)"
    )
    c.executemany(
        "INSERT INTO dim VALUES (?, ?, ?, ?)",
        [
            ("M1", "PUMPS", "2020-01-01", "2030-04-01"),
            ("M1", "VALVES", "2030-04-01", OPEN_END),
        ],
    )
    c.executemany(
        "INSERT INTO fact VALUES ('M1', ?, ?)",
        [
            ("2030-02-10", "100.00"),
            ("2030-04-01", "50.00"),
            ("2030-05-02", "25.00"),
        ],
    )
    return w


KW = {"key": "material", "date": "posted", "attr": "grp", "measure": "amount"}


def test_as_of_uses_the_row_valid_on_the_posting_date(
    wh: SqliteWarehouse,
) -> None:
    got = as_of_totals(wh, "fact", "dim", **KW)
    assert got == {"PUMPS": Decimal("100.00"), "VALVES": Decimal("75.00")}


def test_current_rewrites_history(wh: SqliteWarehouse) -> None:
    got = as_of_totals(wh, "fact", "dim", mode="current", **KW)
    assert got == {"VALVES": Decimal("175.00")}


def test_unmatched_facts_surface_as_question_mark(
    wh: SqliteWarehouse,
) -> None:
    wh.connection.execute(
        "UPDATE dim SET valid_from = '2030-03-01' WHERE grp = 'PUMPS'"
    )
    assert as_of_totals(wh, "fact", "dim", **KW)["?"] == Decimal("100.00")


def test_sql_rejects_unsafe_identifiers_and_modes() -> None:
    with pytest.raises(ValueError, match="not an identifier"):
        as_of_sql("fact; DROP TABLE x", "dim", **KW)
    with pytest.raises(ValueError, match="mode must be"):
        as_of_sql("fact", "dim", mode="never", **KW)


def test_validity_problems(wh: SqliteWarehouse) -> None:
    rows = wh.query("SELECT * FROM dim")
    assert validity_problems(rows, key="material") == []
    rows[0]["valid_to"] = "2030-04-15"
    assert [p.problem for p in validity_problems(rows, key="material")] == [
        "overlap"
    ]
    rows[0]["valid_to"] = "2030-03-01"
    assert [p.problem for p in validity_problems(rows, key="material")] == [
        "gap"
    ]
    rows[0]["valid_to"] = "2019-01-01"
    assert "inverted" in [
        p.problem for p in validity_problems(rows, key="material")
    ]
