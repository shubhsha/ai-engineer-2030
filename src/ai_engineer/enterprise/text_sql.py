"""Chapter 43: trust the SQL a model writes.

Three ideas, each small. Score generated SQL by what it returns, on more
than one database. Let the model choose approved metrics and let code
write the SQL. Ship every answer with its SQL, its metric versions and a
check against a trusted total. Every query runs under ``sqlsafe``.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ai_engineer.enterprise.evidence import Evidence
from ai_engineer.enterprise.sqlsafe import UnsafeSql, guarded_query
from ai_engineer.llm.model import Model, Request, user
from ai_engineer.llm.structured import parse_json

Rows = tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    type: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class Table:
    name: str
    columns: tuple[Column, ...]


def tables_from(conn: sqlite3.Connection) -> tuple[Table, ...]:
    """Read the tables and columns of a database, without its rows."""
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return tuple(
        Table(
            name,
            tuple(
                Column(c[1], c[2], "nullable" if not c[3] and not c[5] else "")
                for c in conn.execute(f"PRAGMA table_info({name})")
            ),
        )
        for name in names
    )


# region schema_prompt
def schema_prompt(tables: Sequence[Table]) -> str:
    """One line per table: ``orders(order_id INTEGER, freight REAL?)``.

    A trailing ``?`` marks a column that may be NULL, because the model
    cannot see that from the names alone.
    """
    lines = []
    for table in tables:
        cols = ", ".join(
            f"{c.name} {c.type}{'?' if c.note == 'nullable' else ''}"
            for c in table.columns
        )
        lines.append(f"{table.name}({cols})")
    return "\n".join(lines)


# endregion schema_prompt

SYSTEM = (
    "You write one read-only SQLite SELECT statement that answers the "
    "question. Reply with the SQL only.\nSchema (? means may be NULL):\n"
)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """The SQL in a reply: the fenced block if there is one."""
    found = _FENCE.search(text)
    return (found.group(1) if found else text).strip().rstrip(";")


def generate_sql(model: Model, question: str, schema: str) -> str:
    """Ask a model for SQL. The result is untrusted until it is run."""
    request = Request(SYSTEM + schema, (user(question),), max_tokens=400)
    return extract_sql(model.complete(request).text)


# region result_match
def _tables_of(conn: sqlite3.Connection) -> list[str]:
    return [t.name for t in tables_from(conn)]


def _norm(rows: Rows) -> list[tuple[Any, ...]]:
    def one(value: Any) -> Any:
        return round(value, 4) if isinstance(value, float) else value

    return [tuple(one(v) for v in row) for row in rows]


def result_match(conn: sqlite3.Connection, predicted: str, gold: str) -> bool:
    """True when both queries return the same rows on this database.

    Rows are compared as multisets, unless the gold query has an
    ORDER BY, which then makes the order part of the answer. A query
    that is refused or fails does not match.
    """
    allowed = _tables_of(conn)
    try:
        want = _norm(guarded_query(conn, gold, allowed).rows)
        got = _norm(guarded_query(conn, predicted, allowed).rows)
    except (UnsafeSql, sqlite3.Error):
        return False
    if "order by" in gold.lower():
        return got == want
    return Counter(got) == Counter(want)


def suite_accuracy(
    conns: Sequence[sqlite3.Connection], predicted: str, gold: str
) -> bool:
    """True only if the query matches the gold on every database."""
    return all(result_match(c, predicted, gold) for c in conns)


# endregion result_match


@dataclass(frozen=True, slots=True)
class Case:
    question: str
    gold: str
    predicted: str


@dataclass(frozen=True, slots=True)
class Scores:
    cases: int
    exact: int
    execution: int
    suite: int


def _squash(sql: str) -> str:
    return " ".join(sql.lower().replace(";", " ").split())


def score(
    cases: Iterable[Case], conns: Sequence[sqlite3.Connection]
) -> Scores:
    """Count hits under three scorers. ``conns[0]`` is the single database."""
    n = exact = execution = suite = 0
    for case in cases:
        n += 1
        exact += _squash(case.predicted) == _squash(case.gold)
        execution += result_match(conns[0], case.predicted, case.gold)
        suite += suite_accuracy(conns, case.predicted, case.gold)
    return Scores(n, exact, execution, suite)


@dataclass(frozen=True, slots=True)
class Metric:
    """One approved definition. Change it and the version changes."""

    name: str
    version: int
    expression: str
    grain: str
    where: str
    description: str


@dataclass(frozen=True, slots=True)
class Dimension:
    name: str
    expression: str


_FROM = {
    "line": (
        "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
        "LEFT JOIN customers c ON c.customer_id = o.customer_id"
    ),
    "order": (
        "FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id"
    ),
}
_LIVE = "o.status <> 'cancelled'"
FISCAL_YEAR = (
    "(CAST(substr(o.order_date, 1, 4) AS INTEGER) "
    "+ (substr(o.order_date, 6, 2) >= '04'))"
)
SALES_DIMENSIONS = (
    Dimension("region", "c.region"),
    Dimension("fiscal_year", FISCAL_YEAR),
)
SALES_METRICS = (
    Metric(
        "net_revenue",
        2,
        "SUM(l.qty * l.unit_price * (1 - COALESCE(l.discount, 0)))",
        "line",
        _LIVE,
        "Billed value after discounts, cancelled orders excluded.",
    ),
    Metric(
        "gross_revenue",
        1,
        "SUM(l.qty * l.unit_price)",
        "line",
        _LIVE,
        "List value before discounts, cancelled orders excluded.",
    ),
    Metric(
        "avg_discount",
        1,
        "AVG(COALESCE(l.discount, 0))",
        "line",
        _LIVE,
        "Mean discount per line; a missing discount counts as zero.",
    ),
    Metric(
        "order_count",
        1,
        "COUNT(*)",
        "order",
        _LIVE,
        "Orders that were not cancelled.",
    ),
    Metric(
        "freight_total",
        1,
        "SUM(o.freight)",
        "order",
        _LIVE,
        "Freight charged, once per order.",
    ),
    Metric(
        "active_customers",
        1,
        "COUNT(DISTINCT o.customer_id)",
        "order",
        _LIVE,
        "Customers with at least one live order.",
    ),
)


# region metric_registry
class MetricError(ValueError):
    """The request names something the registry does not define."""


def _literal(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise MetricError(f"filter value {value!r} must be int or str")
    if isinstance(value, int):
        return str(value)
    return "'" + value.replace("'", "''") + "'"


class MetricRegistry:
    """Named metrics and dimensions. Code, not the model, writes SQL."""

    def __init__(
        self,
        metrics: Iterable[Metric] = SALES_METRICS,
        dimensions: Iterable[Dimension] = SALES_DIMENSIONS,
    ) -> None:
        self.metrics = {m.name: m for m in metrics}
        self.dimensions = {d.name: d for d in dimensions}

    def compose(
        self,
        metric: str,
        filters: Mapping[str, object] | None = None,
        group_by: Sequence[str] = (),
    ) -> str:
        if metric not in self.metrics:
            raise MetricError(f"unknown metric {metric!r}")
        m = self.metrics[metric]
        wanted = [*(filters or {}), *group_by]
        for name in wanted:
            if name not in self.dimensions:
                raise MetricError(f"unknown dimension {name!r}")
        where = [m.where] + [
            f"{self.dimensions[k].expression} = {_literal(v)}"
            for k, v in (filters or {}).items()
        ]
        keys = [self.dimensions[g].expression for g in group_by]
        select = [f"{e} AS {g}" for e, g in zip(keys, group_by, strict=True)]
        sql = f"SELECT {', '.join([*select, m.expression + ' AS ' + m.name])}"
        sql += f" {_FROM[m.grain]} WHERE {' AND '.join(where)}"
        if keys:
            sql += f" GROUP BY {', '.join(keys)} ORDER BY {', '.join(keys)}"
        return sql

    def versions(self, names: Iterable[str]) -> tuple[tuple[str, int], ...]:
        return tuple((n, self.metrics[n].version) for n in names)


# endregion metric_registry


def select_metric(
    model: Model, question: str, registry: MetricRegistry
) -> str | None:
    """Let the model pick a metric, filters and grouping; code writes SQL.

    Returns None when the model declines or names something unknown.
    """
    menu = "\n".join(
        f"{m.name}: {m.description}" for m in registry.metrics.values()
    )
    dims = ", ".join(registry.dimensions)
    system = (
        "Choose one metric. Reply with JSON: "
        '{"metric": str, "filters": {dim: value}, "group_by": [dim]}, '
        'or {"metric": null} if none fits.\nMetrics:\n'
        f"{menu}\nDimensions: {dims}"
    )
    reply = model.complete(Request(system, (user(question),), max_tokens=200))
    data, problems = parse_json(reply.text)
    if problems or not isinstance(data, dict) or not data.get("metric"):
        return None
    try:
        return registry.compose(
            str(data["metric"]),
            data.get("filters") or {},
            data.get("group_by") or (),
        )
    except MetricError:
        return None


# region answer_with_evidence
@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    sql: str
    metrics: tuple[tuple[str, int], ...]
    rows: Rows


def reconcile_total(rows: Rows, trusted: Decimal) -> bool:
    """Do the last-column values add up to a total we already trust?"""
    got = sum((Decimal(str(r[-1])) for r in rows), Decimal(0))
    return abs(got - trusted) <= Decimal("0.005")


def answer_with_evidence(
    conn: sqlite3.Connection,
    question: str,
    sql: str,
    metrics: tuple[tuple[str, int], ...],
    *,
    trusted: Decimal | None,
    run_id: str,
    recorded_at: float,
) -> tuple[Answer | None, Evidence]:
    """Run guarded SQL and record whether the number can be trusted.

    The answer is withheld unless the reconciliation passes. With no
    trusted total the verdict is ``skip``: nothing was checked.
    """
    scope = f"text_sql:{question}"
    try:
        rows = guarded_query(conn, sql, _tables_of(conn)).rows
    except (UnsafeSql, sqlite3.Error) as exc:
        return None, Evidence(
            "text_sql.run",
            run_id,
            recorded_at,
            scope,
            "the query runs",
            str(exc),
            "error",
            sql,
        )
    answer = Answer(question, sql, metrics, rows)
    if trusted is None:
        return answer, Evidence(
            "text_sql.reconcile",
            run_id,
            recorded_at,
            scope,
            "no trusted total",
            "unchecked",
            "skip",
            sql,
        )
    ok = reconcile_total(rows, trusted)
    total = sum((Decimal(str(r[-1])) for r in rows), Decimal(0))
    return (answer if ok else None), Evidence(
        "text_sql.reconcile",
        run_id,
        recorded_at,
        scope,
        str(trusted),
        str(total),
        "pass" if ok else "fail",
        sql,
    )


# endregion answer_with_evidence
