"""Chapter 41: layered reconciliation of a source and a target table.

Each layer is cheap and catches a class of defect the layer before it
cannot. Row counts (L2), control totals (L3) and a row-level diff by key
and fingerprint (L4) work on lists of rows, so a SQLite file, a Fabric
query result and a test double all look the same. Every check returns an
:class:`~ai_engineer.enterprise.evidence.Evidence` record.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ai_engineer.enterprise.evidence import Evidence, Verdict
from ai_engineer.enterprise.warehouse import Row, Warehouse

Key = tuple[Any, ...]
_NULL = "\x00null"
_SEP = "\x1f"


# region row_fingerprint
def _normalize(value: Any) -> str:
    """One canonical text per value, so equal facts hash equally."""
    if value is None:
        return _NULL
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float | Decimal):
        number = Decimal(str(value))
        return format(number.normalize(), "f") if number else "0"
    return str(value).strip()


def row_fingerprint(
    row: Mapping[str, Any], columns: Sequence[str], *, bits: int = 64
) -> str:
    """Hash the normalized columns of a row into a hex fingerprint.

    ``12.5`` and ``Decimal("12.50")`` hash alike; a changed digit, a
    swapped column or a missing value does not.
    """
    if bits % 8 or not 8 <= bits <= 512:
        raise ValueError("bits must be a multiple of 8, from 8 to 512")
    text = _SEP.join(_normalize(row[c]) for c in columns)
    digest = hashlib.blake2b(text.encode(), digest_size=bits // 8)
    return digest.hexdigest()


# endregion


@dataclass(frozen=True, slots=True)
class Diff:
    """Keys missing from the target, extra in it, or changed.

    ``duplicate`` lists keys that appear more than once on either side:
    a key diff cannot compare a row with itself twice.
    """

    missing: tuple[Key, ...] = ()
    extra: tuple[Key, ...] = ()
    changed: tuple[Key, ...] = ()
    duplicate: tuple[Key, ...] = ()

    @property
    def clean(self) -> bool:
        return not (
            self.missing or self.extra or self.changed or self.duplicate
        )

    def summary(self) -> str:
        return (
            f"missing={len(self.missing)} extra={len(self.extra)} "
            f"changed={len(self.changed)} duplicate={len(self.duplicate)}"
        )


def load(warehouse: Warehouse, table: str) -> list[Row]:
    """Read every row of a table through the warehouse port."""
    return warehouse.query(f"SELECT * FROM {table}")


def _verdict(ok: bool) -> Verdict:
    return "pass" if ok else "fail"


def compare_counts(
    source: Sequence[Row],
    target: Sequence[Row],
    *,
    scope: str,
    run_id: str,
    recorded_at: float,
) -> Evidence:
    """L2: the same number of rows on both sides."""
    return Evidence(
        test_id="L2.row_counts",
        run_id=run_id,
        recorded_at=recorded_at,
        scope=scope,
        expected=str(len(source)),
        observed=str(len(target)),
        verdict=_verdict(len(source) == len(target)),
    )


# region control_totals
def control_totals(
    rows: Sequence[Row], group_by: Sequence[str], measure: str
) -> dict[Key, Decimal]:
    """Sum a measure per group, exactly, with Decimal."""
    totals: dict[Key, Decimal] = {}
    for row in rows:
        key = tuple(row[c] for c in group_by)
        amount = Decimal(str(row[measure]))
        totals[key] = totals.get(key, Decimal(0)) + amount
    return totals


# endregion


def compare_totals(
    source: Sequence[Row],
    target: Sequence[Row],
    *,
    group_by: Sequence[str],
    measure: str,
    scope: str,
    run_id: str,
    recorded_at: float,
    tolerance: Decimal = Decimal(0),
) -> Evidence:
    """L3: every group total agrees within ``tolerance``."""
    a = control_totals(source, group_by, measure)
    b = control_totals(target, group_by, measure)
    zero = Decimal(0)
    off = sorted(
        (
            k
            for k in a.keys() | b.keys()
            if abs(a.get(k, zero) - b.get(k, zero)) > tolerance
        ),
        key=str,
    )
    shown = "; ".join(
        f"{k}: {a.get(k, zero)} vs {b.get(k, zero)}" for k in off[:3]
    )
    return Evidence(
        test_id="L3.control_totals",
        run_id=run_id,
        recorded_at=recorded_at,
        scope=scope,
        expected=f"{len(a)} groups agree",
        observed=f"{len(off)} groups differ" + (f" ({shown})" if off else ""),
        verdict=_verdict(not off),
    )


# region key_diff
def key_diff(
    source: Sequence[Row],
    target: Sequence[Row],
    *,
    key: Sequence[str],
    columns: Sequence[str],
    bits: int = 64,
) -> Diff:
    """L4: compare two row sets by key, then by row fingerprint."""

    def index(rows: Sequence[Row]) -> tuple[dict[Key, str], set[Key]]:
        seen: dict[Key, str] = {}
        dup: set[Key] = set()
        for row in rows:
            k = tuple(row[c] for c in key)
            if k in seen:
                dup.add(k)
            seen[k] = row_fingerprint(row, columns, bits=bits)
        return seen, dup

    a, dup_a = index(source)
    b, dup_b = index(target)
    dup = dup_a | dup_b
    return Diff(
        missing=tuple(sorted(a.keys() - b.keys(), key=str)),
        extra=tuple(sorted(b.keys() - a.keys(), key=str)),
        changed=tuple(
            sorted(
                (k for k in a.keys() & b.keys() - dup if a[k] != b[k]),
                key=str,
            )
        ),
        duplicate=tuple(sorted(dup, key=str)),
    )


# endregion


def diff_evidence(
    diff: Diff, *, scope: str, run_id: str, recorded_at: float
) -> Evidence:
    """Turn a diff into the record a release gate reads."""
    return Evidence(
        test_id="L4.row_diff",
        run_id=run_id,
        recorded_at=recorded_at,
        scope=scope,
        expected="missing=0 extra=0 changed=0 duplicate=0",
        observed=diff.summary(),
        verdict=_verdict(diff.clean),
    )


def collision_probability(rows: int, bits: int) -> float:
    """Birthday bound: chance that two of ``rows`` hashes collide."""
    pairs = rows * (rows - 1) / 2
    return -math.expm1(-pairs / 2.0**bits)
