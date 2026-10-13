"""Chapter 42: run model-written SQL under least privilege.

A denylist of bad words tries to recognize an attack in the text of the
statement. This module does the opposite. SQLite itself reports every
action a statement would perform, before it runs, and the authorizer
below says yes only to a short list: one SELECT, reads of allowlisted
tables and a few pure functions. Everything else is refused by default.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

# region unsafe_sql
SAFE_FUNCTIONS = frozenset(
    {"count", "sum", "min", "max", "avg", "abs", "length", "coalesce"}
    | {"round", "lower", "upper", "substr", "typeof", "nullif", "ifnull"}
)


class UnsafeSql(Exception):
    """The statement is refused. The message says why, for the model."""


def _authorizer(allowed_tables: Collection[str]) -> Any:
    tables = {name.lower() for name in allowed_tables}

    def decide(action: int, arg1: Any, arg2: Any, *_: Any) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and str(arg1).lower() in tables:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and arg2 in SAFE_FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return decide


# endregion unsafe_sql


def check_sql(
    conn: sqlite3.Connection, sql: str, allowed_tables: Collection[str]
) -> None:
    """Compile ``sql`` under the authorizer without running it."""
    _run(conn, "EXPLAIN " + sql, allowed_tables, max_rows=1, max_steps=None)


@dataclass(frozen=True, slots=True)
class QueryResult:
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool


# region guarded_query
def guarded_query(
    conn: sqlite3.Connection,
    sql: str,
    allowed_tables: Collection[str],
    *,
    max_rows: int = 1000,
    max_steps: int = 5_000_000,
) -> QueryResult:
    """Run one read-only statement, with a row limit and a work limit.

    ``max_steps`` counts SQLite virtual-machine instructions, which is a
    deterministic stand-in for a time limit: the same query and data
    always stop at the same step, so tests need no clock.
    """
    return _run(conn, sql, allowed_tables, max_rows, max_steps)


# endregion guarded_query


def _run(
    conn: sqlite3.Connection,
    sql: str,
    allowed_tables: Collection[str],
    max_rows: int,
    max_steps: int | None,
) -> QueryResult:
    steps = 0

    def tick() -> int:
        nonlocal steps
        steps += 1000
        return 1 if max_steps is not None and steps > max_steps else 0

    conn.set_authorizer(_authorizer(allowed_tables))
    if max_steps is not None:
        conn.set_progress_handler(tick, 1000)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.DatabaseError as exc:
        raise UnsafeSql(_reason(exc)) from exc
    finally:
        conn.set_authorizer(None)
        conn.set_progress_handler(None, 0)
    return QueryResult(
        tuple(tuple(r) for r in rows[:max_rows]), len(rows) > max_rows
    )


def _reason(exc: sqlite3.DatabaseError) -> str:
    text = str(exc)
    if "not authorized" in text:
        return "refused: only SELECT on allowlisted tables is permitted"
    if "interrupted" in text:
        return "refused: the query exceeded its work limit"
    if "one statement" in text:
        return "refused: send exactly one statement"
    return f"rejected by the database: {text}"
