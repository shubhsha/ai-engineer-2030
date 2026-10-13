"""Tests for chapter 44: manifests, stable baselines, paired tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ai_engineer.enterprise.baseline import (
    ChangeRule,
    Manifest,
    Snapshot,
    compare,
    discordant,
    mcnemar_exact,
    paired_evidence,
    round_to,
)
from ai_engineer.enterprise.evidence import release_gate

D = Decimal


def rows(
    vat_de: str = "100.00", loaded: str = "t1"
) -> list[dict[str, object]]:
    return [
        {
            "id": 1,
            "country": "DE",
            "net": "100.00",
            "vat": vat_de,
            "loaded_at": loaded,
        },
        {
            "id": 2,
            "country": "FR",
            "net": "50.00",
            "vat": "10.00",
            "loaded_at": loaded,
        },
    ]


def snap(data: list[dict[str, object]]) -> Snapshot:
    return Snapshot.build(
        data,
        ["id"],
        ignore=["loaded_at"],
        normalizers={"vat": round_to(2), "net": round_to(2)},
    )


RULE = ChangeRule("vat", "exact", D("19.00"), where=(("country", "DE"),))
MANIFEST = Manifest("finance", (RULE,))


def test_unchanged_snapshot_has_no_differences() -> None:
    base = snap(rows("19.00"))
    d = compare(base, snap(list(reversed(rows("19.00", "t2")))), MANIFEST)
    assert d.plain_count == 0
    assert not d.missing


def test_planned_change_that_did_not_happen_is_missing() -> None:
    d = compare(snap(rows("100.00")), snap(rows("100.00")), MANIFEST)
    assert len(d.missing) == 1
    assert not d.clean


def test_expected_change_passes() -> None:
    d = compare(snap(rows("100.00")), snap(rows("19.00")), MANIFEST)
    assert len(d.expected) == 1
    assert d.clean
    ev = d.to_evidence(run_id="r", recorded_at=1.0, scope="vat")
    assert ev.verdict == "pass"


def test_accident_is_unexpected_and_fails() -> None:
    after = rows("19.00")
    after[1]["vat"] = "10.01"
    d = compare(snap(rows("100.00")), snap(after), MANIFEST)
    assert [f.key for f in d.unexpected] == [(2,)]
    assert d.plain_count == 2
    assert not d.clean
    ev = d.to_evidence(run_id="r", recorded_at=1.0, scope="vat")
    assert ev.verdict == "fail"
    assert "(2,)" in ev.location


def test_wrong_value_in_covered_cell_is_unexpected() -> None:
    d = compare(snap(rows("100.00")), snap(rows("20.00")), MANIFEST)
    assert len(d.unexpected) == 1
    assert not d.expected


def test_new_rows_need_permission() -> None:
    extra = {"id": 3, "country": "FR", "net": "1", "vat": "0"}
    after = [*rows("19.00"), extra]
    d = compare(snap(rows("100.00")), snap(after), MANIFEST)
    assert len(d.new) == 1
    assert not d.clean
    ok = Manifest("finance", (RULE,), allow_new_rows=True)
    assert compare(snap(rows("100.00")), snap(after), ok).clean


def test_removed_row_is_unexpected() -> None:
    d = compare(snap(rows("100.00")), snap(rows("19.00")[:1]), MANIFEST)
    assert any(f.column == "<row>" for f in d.unexpected)


def test_ratio_and_within_rules() -> None:
    ratio = ChangeRule("net", "ratio", D("1.10"), tolerance=D("0.01"))
    d = compare(snap(rows()), snap(_scaled(1.10)), Manifest("o", (ratio,)))
    assert len(d.expected) == 2
    near = ChangeRule("net", "within", D("5"))
    d2 = compare(snap(rows()), snap(_scaled(1.10)), Manifest("o", (near,)))
    assert len(d2.expected) == 1
    assert len(d2.unexpected) == 1


def _scaled(f: float) -> list[dict[str, object]]:
    out = rows()
    for r in out:
        r["net"] = str(round(float(str(r["net"])) * f, 2))
    return out


def test_normalizer_hides_float_noise() -> None:
    a = Snapshot.build(
        [{"id": 1, "x": 0.1 + 0.2}], ["id"], normalizers={"x": round_to(2)}
    )
    b = Snapshot.build(
        [{"id": 1, "x": 0.3}], ["id"], normalizers={"x": round_to(2)}
    )
    assert a == b
    assert a.fingerprint() == b.fingerprint()


def test_duplicate_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Snapshot.build([{"id": 1}, {"id": 1}], ["id"])


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact(2, 10) == pytest.approx(0.03857421875)
    with pytest.raises(ValueError, match="negative"):
        mcnemar_exact(-1, 2)


def test_paired_evidence_and_gate() -> None:
    a = [True] * 10 + [False] * 2 + [True] * 8
    b = [False] * 10 + [False] * 2 + [True] * 8
    assert discordant(a, b) == (10, 0)
    ev = paired_evidence(a, b, run_id="r", recorded_at=1.0, scope="prompt")
    assert ev.verdict == "fail"
    gate = release_gate([ev], required=["progression.paired"])
    assert not gate.passed
    with pytest.raises(ValueError, match="same cases"):
        discordant([True], [True, False])
