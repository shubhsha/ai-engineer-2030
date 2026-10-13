"""Chapter 41: the layered checks catch what each layer should."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from ai_engineer.enterprise.reconcile import (
    Diff,
    collision_probability,
    compare_counts,
    compare_totals,
    control_totals,
    diff_evidence,
    key_diff,
    load,
    row_fingerprint,
)
from ai_engineer.enterprise.warehouse import Column, InMemoryWarehouse, Row

NOW = 1_700_000_000.0
COLS = ("order_id", "line", "amount", "currency")


def make_rows(rng: random.Random, n: int = 40) -> list[Row]:
    return [
        {
            "order_id": 1000 + i // 4,
            "line": i % 4 + 1,
            "amount": str(Decimal(rng.randrange(100, 99999)) / 100),
            "currency": rng.choice(["EUR", "USD"]),
        }
        for i in range(n)
    ]


def test_fingerprint_ignores_formatting_not_content() -> None:
    a = {"x": "12.5", "y": " EUR "}
    b = {"x": Decimal("12.50"), "y": "EUR"}
    assert row_fingerprint(a, ["x", "y"]) == row_fingerprint(b, ["x", "y"])
    assert row_fingerprint(a, ["x", "y"]) != row_fingerprint(
        {"x": "12.6", "y": "EUR"}, ["x", "y"]
    )
    assert len(row_fingerprint(a, ["x"], bits=128)) == 32
    with pytest.raises(ValueError, match="multiple of 8"):
        row_fingerprint(a, ["x"], bits=12)


def test_fingerprint_separates_null_from_empty_and_neighbors() -> None:
    assert row_fingerprint({"a": None}, ["a"]) != row_fingerprint(
        {"a": ""}, ["a"]
    )
    assert row_fingerprint({"a": "ab", "b": "c"}, ["a", "b"]) != (
        row_fingerprint({"a": "a", "b": "bc"}, ["a", "b"])
    )


def test_counts_pass_and_fail(rng: random.Random) -> None:
    src = make_rows(rng)
    ok = compare_counts(src, src, scope="t", run_id="r", recorded_at=NOW)
    bad = compare_counts(src, src[:-3], scope="t", run_id="r", recorded_at=NOW)
    assert (ok.verdict, bad.verdict) == ("pass", "fail")
    assert bad.expected == "40"
    assert bad.observed == "37"


def test_totals_catch_a_cent_that_counts_miss(rng: random.Random) -> None:
    src = make_rows(rng)
    tgt = [dict(r) for r in src]
    tgt[5]["amount"] = str(Decimal(tgt[5]["amount"]) + Decimal("0.01"))
    assert (
        compare_counts(
            src, tgt, scope="t", run_id="r", recorded_at=NOW
        ).verdict
        == "pass"
    )
    ev = compare_totals(
        src,
        tgt,
        group_by=["currency"],
        measure="amount",
        scope="t",
        run_id="r",
        recorded_at=NOW,
    )
    assert ev.verdict == "fail"
    assert "0.01" not in ev.expected
    loose = compare_totals(
        src,
        tgt,
        group_by=["currency"],
        measure="amount",
        tolerance=Decimal("0.01"),
        scope="t",
        run_id="r",
        recorded_at=NOW,
    )
    assert loose.verdict == "pass"


def test_decimal_totals_are_exact() -> None:
    rows = [{"g": "a", "v": "0.1"}] * 10
    assert control_totals(rows, ["g"], "v") == {("a",): Decimal("1.0")}


def test_key_diff_finds_each_kind(rng: random.Random) -> None:
    src = make_rows(rng)
    tgt = [dict(r) for r in src]
    gone = tgt.pop(0)
    tgt[3]["amount"] = "1.00"
    tgt.append({"order_id": 9999, "line": 1, "amount": "5", "currency": "EUR"})
    tgt.append(dict(tgt[10]))
    d = key_diff(src, tgt, key=["order_id", "line"], columns=COLS)
    assert d.missing == ((gone["order_id"], gone["line"]),)
    assert d.extra == ((9999, 1),)
    assert len(d.changed) == 1
    assert len(d.duplicate) == 1
    assert not d.clean
    ev = diff_evidence(d, scope="t", run_id="r", recorded_at=NOW)
    assert ev.verdict == "fail"
    assert "duplicate=1" in ev.observed


def test_clean_diff_and_warehouse_load(rng: random.Random) -> None:
    src = make_rows(rng)
    cols = [Column(c, "TEXT", True) for c in COLS]
    wh = InMemoryWarehouse({"a": cols, "b": cols}, {"a": src, "b": src})
    d = key_diff(
        load(wh, "a"), load(wh, "b"), key=["order_id", "line"], columns=COLS
    )
    assert d == Diff()
    assert d.clean


def test_collision_probability_matches_known_values() -> None:
    assert collision_probability(10_000_000, 64) == pytest.approx(
        2.71e-6, rel=0.01
    )
    assert collision_probability(10_000_000, 128) < 1e-24
    assert collision_probability(1, 64) == 0.0
