"""Chapter 41: the as-of join for time-dependent master data.

A dimension row is valid for ``valid_from <= date < valid_to``. Joining
a fact to the row that was valid on its posting date gives the answer
the business saw then; joining to today's row rewrites history. Dates
are ISO strings, which sort the same as dates. Volume V reuses this
interface, so it stays small: one query builder, one join, one totals
helper and one validity check.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from ai_engineer.enterprise.warehouse import Row, Warehouse

OPEN_END = "9999-12-31"
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class Validity:
    """A broken validity chain for one key."""

    key: str
    problem: str  # "overlap" | "gap" | "inverted"
    at: str


def _ident(name: str) -> str:
    if not _IDENT.fullmatch(name):
        raise ValueError(f"not an identifier: {name!r}")
    return name


# region as_of_join
def as_of_sql(
    fact: str,
    dim: str,
    *,
    key: str,
    date: str,
    attr: str,
    measure: str,
    mode: str = "as_of",
) -> str:
    """SQL that attaches ``attr`` to each fact row.

    ``mode="as_of"`` picks the dimension row valid on the fact's date.
    ``mode="current"`` picks the open-ended row, valid to ``OPEN_END``.
    """
    f, d, k, dt, a, m = map(_ident, (fact, dim, key, date, attr, measure))
    if mode == "as_of":
        cond = f"f.{dt} >= d.valid_from AND f.{dt} < d.valid_to"
    elif mode == "current":
        cond = f"d.valid_to = '{OPEN_END}'"
    else:
        raise ValueError("mode must be 'as_of' or 'current'")
    return (
        f"SELECT f.{k} AS {k}, f.{dt} AS {dt}, f.{m} AS {m}, d.{a} AS {a} "
        f"FROM {f} f LEFT JOIN {d} d ON d.{k} = f.{k} AND {cond}"
    )


def as_of_totals(
    warehouse: Warehouse,
    fact: str,
    dim: str,
    *,
    key: str,
    date: str,
    attr: str,
    measure: str,
    mode: str = "as_of",
) -> dict[str, Decimal]:
    """Sum the measure per attribute value; unmatched facts get ``?``."""
    sql = as_of_sql(
        fact, dim, key=key, date=date, attr=attr, measure=measure, mode=mode
    )
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for row in warehouse.query(sql):
        totals[row[attr] or "?"] += Decimal(str(row[measure]))
    return dict(totals)


# endregion


def validity_problems(rows: list[Row], *, key: str) -> list[Validity]:
    """Find overlaps, gaps and inverted ranges in a dimension's chain."""
    by_key: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_key[str(row[key])].append(row)
    found: list[Validity] = []
    for k, chain in sorted(by_key.items()):
        chain.sort(key=lambda r: r["valid_from"])
        for row in chain:
            if row["valid_from"] >= row["valid_to"]:
                found.append(Validity(k, "inverted", row["valid_from"]))
        for prev, nxt in pairwise(chain):
            if prev["valid_to"] > nxt["valid_from"]:
                found.append(Validity(k, "overlap", nxt["valid_from"]))
            elif prev["valid_to"] < nxt["valid_from"]:
                found.append(Validity(k, "gap", prev["valid_to"]))
    return found
