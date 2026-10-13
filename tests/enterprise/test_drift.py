"""Chapter 41: schema drift classes."""

from __future__ import annotations

from ai_engineer.enterprise.drift import drift_evidence, schema_drift
from ai_engineer.enterprise.warehouse import Column

BW = [
    Column("MATERIAL", "VARCHAR(18)", False),
    Column("QTY", "INT", True),
    Column("NET_VALUE", "DECIMAL(15,2)", True),
    Column("PLANT", "VARCHAR(4)", True),
]


def kinds(target: list[Column]) -> list[tuple[str, str]]:
    return [(d.kind, d.column) for d in schema_drift(BW, target)]


def test_identical_schemas_have_no_drift() -> None:
    assert schema_drift(BW, list(BW)) == []


def test_rename_widen_and_drop() -> None:
    target = [
        Column("MATERIAL_ID", "VARCHAR(18)", False),
        Column("QTY", "BIGINT", True),
        Column("NET_VALUE", "DECIMAL(15,2)", True),
    ]
    assert kinds(target) == [
        ("widened", "QTY"),
        ("renamed", "MATERIAL"),
        ("dropped", "PLANT"),
    ]


def test_narrowing_retyping_nullability_and_added() -> None:
    target = [
        Column("MATERIAL", "VARCHAR(10)", False),
        Column("QTY", "VARCHAR(5)", True),
        Column("NET_VALUE", "DECIMAL(15,2)", False),
        Column("PLANT", "VARCHAR(4)", True),
        Column("LOADED_AT", "TIMESTAMP", True),
    ]
    assert kinds(target) == [
        ("narrowed", "MATERIAL"),
        ("retyped", "QTY"),
        ("nullable", "NET_VALUE"),
        ("added", "LOADED_AT"),
    ]


def test_evidence_fails_only_on_blocking_drift() -> None:
    widened = [Column("QTY", "BIGINT", True)]
    ok = drift_evidence(
        schema_drift([Column("QTY", "INT", True)], widened),
        scope="t",
        run_id="r",
        recorded_at=1.0,
    )
    bad = drift_evidence(
        schema_drift(BW, widened), scope="t", run_id="r", recorded_at=1.0
    )
    assert ok.verdict == "pass"
    assert "widened QTY" in ok.observed
    assert bad.verdict == "fail"
