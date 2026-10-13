"""Chapter 41: schema drift between a source and a target table.

L1 of the migration checks. Two column lists, read through the
warehouse port, are compared by name. A column that vanished and a new
one of the same type in the same place is reported as a probable rename,
because a rename is the drift that silently breaks downstream queries.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ai_engineer.enterprise.evidence import Evidence
from ai_engineer.enterprise.warehouse import Column

Kind = Literal[
    "dropped", "added", "renamed", "widened", "narrowed", "retyped", "nullable"
]
_BLOCKING: frozenset[Kind] = frozenset(
    {"dropped", "renamed", "narrowed", "retyped"}
)
_INT_RANK = {"TINYINT": 1, "SMALLINT": 2, "INT": 3, "INTEGER": 3, "BIGINT": 4}


@dataclass(frozen=True, slots=True)
class Drift:
    kind: Kind
    column: str
    before: str = ""
    after: str = ""

    @property
    def blocking(self) -> bool:
        return self.kind in _BLOCKING


def _width(type_name: str) -> tuple[str, int] | None:
    """Family and size of a type, or None when we cannot rank it."""
    t = type_name.upper().strip()
    if t in _INT_RANK:
        return "int", _INT_RANK[t]
    m = re.fullmatch(r"(VAR)?CHAR\((\d+)\)", t)
    if m:
        return "char", int(m.group(2))
    m = re.fullmatch(r"DECIMAL\((\d+),\s*(\d+)\)", t)
    if m:
        return f"dec{m.group(2)}", int(m.group(1))
    return None


def _retype(before: str, after: str) -> Kind:
    a, b = _width(before), _width(after)
    if a is None or b is None or a[0] != b[0]:
        return "retyped"
    return "widened" if b[1] >= a[1] else "narrowed"


# region schema_drift
def schema_drift(
    source: Sequence[Column], target: Sequence[Column]
) -> list[Drift]:
    """List every difference between two column lists."""
    src = {c.name: (i, c) for i, c in enumerate(source)}
    tgt = {c.name: (i, c) for i, c in enumerate(target)}
    found: list[Drift] = []
    for name, (_, col) in src.items():
        if name in tgt:
            other = tgt[name][1]
            if col.type != other.type:
                found.append(
                    Drift(
                        _retype(col.type, other.type),
                        name,
                        col.type,
                        other.type,
                    )
                )
            elif col.nullable != other.nullable:
                found.append(
                    Drift(
                        "nullable",
                        name,
                        str(col.nullable),
                        str(other.nullable),
                    )
                )
    gone = [n for n in src if n not in tgt]
    new = [n for n in tgt if n not in src]
    for name in gone:
        i, col = src[name]
        twin = next(
            (n for n in new if tgt[n][0] == i and tgt[n][1].type == col.type),
            None,
        )
        if twin is None:
            found.append(Drift("dropped", name, col.type))
        else:
            new.remove(twin)
            found.append(Drift("renamed", name, name, twin))
    found.extend(Drift("added", n, "", tgt[n][1].type) for n in new)
    return found


# endregion


def drift_evidence(
    drifts: Sequence[Drift], *, scope: str, run_id: str, recorded_at: float
) -> Evidence:
    """L1 record: fail on a dropped, renamed, narrowed or retyped column."""
    bad = [d for d in drifts if d.blocking]
    seen = ", ".join(f"{d.kind} {d.column}" for d in drifts) or "none"
    return Evidence(
        test_id="L1.schema_drift",
        run_id=run_id,
        recorded_at=recorded_at,
        scope=scope,
        expected="no blocking drift",
        observed=seen,
        verdict="fail" if bad else "pass",
    )
