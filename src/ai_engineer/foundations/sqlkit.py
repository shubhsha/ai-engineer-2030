"""Chapter 8: schema, NULL traps, windows, transactions and migrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

Migration = tuple[int, str]


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a connection whose transactions we control explicitly."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def split_statements(script: str) -> list[str]:
    """Split a SQL script into statements; quotes and triggers are safe."""
    statements, buffer = [], ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        raise ValueError(f"incomplete statement: {buffer.strip()[:60]!r}")
    return statements


def schema_version(conn: sqlite3.Connection) -> int:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY)"
    )
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return 0 if row[0] is None else int(row[0])


def run_migrations(
    conn: sqlite3.Connection, migrations: Sequence[Migration]
) -> list[int]:
    """Apply each newer migration in order; each is one transaction."""
    versions = [version for version, _ in migrations]
    if versions != sorted(set(versions)):
        raise ValueError("migration versions must be unique and ascending")
    current, applied = schema_version(conn), []
    for version, script in migrations:
        if version <= current:
            continue
        conn.execute("BEGIN")
        try:
            for statement in split_statements(script):
                conn.execute(statement)
            conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        applied.append(version)
    return applied


# region revenue-trend
REVENUE_TREND = """
SELECT day,
       revenue,
       SUM(revenue) OVER (ORDER BY day)          AS running_total,
       AVG(revenue) OVER (ORDER BY day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS avg_7d,
       revenue - LAG(revenue) OVER (ORDER BY day) AS change
FROM daily_revenue
ORDER BY day
"""
# endregion


def revenue_trend(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(REVENUE_TREND).fetchall()


# region orders-schema
ORDERS_SCHEMA = """
CREATE TABLE customer (id INTEGER PRIMARY KEY,
    name TEXT NOT NULL, region TEXT NOT NULL);
CREATE TABLE product (id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE, name TEXT NOT NULL);
CREATE TABLE orders (id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customer (id),
    created_at TEXT NOT NULL);
CREATE TABLE order_line (
    order_id INTEGER NOT NULL REFERENCES orders (id),
    product_id INTEGER NOT NULL REFERENCES product (id),
    qty INTEGER NOT NULL CHECK (qty > 0),
    unit_price INTEGER NOT NULL,
    PRIMARY KEY (order_id, product_id));
"""
# endregion


# region null-traps
NOT_IN = """
SELECT name FROM customer
WHERE id NOT IN (SELECT customer_id FROM test_account)
"""
NOT_EXISTS = """
SELECT name FROM customer c
WHERE NOT EXISTS (SELECT 1 FROM test_account t WHERE t.customer_id = c.id)
"""
ORDER_COUNTS = """
SELECT c.name, COUNT(o.id) AS orders
FROM customer c LEFT JOIN orders o ON o.customer_id = c.id
GROUP BY c.id ORDER BY c.id
"""
# endregion


# region adjust-stock
def adjust_stock(conn: sqlite3.Connection, sku: str, delta: int) -> int:
    """Add delta to a SKU's stock under the write lock; return the qty."""
    conn.execute("BEGIN IMMEDIATE")  # take the write lock before reading
    try:
        row = conn.execute("SELECT qty FROM stock WHERE sku = ?", (sku,))
        found = row.fetchone()
        if found is None:
            raise KeyError(sku)
        qty = found["qty"] + delta
        # the table's CHECK (qty >= 0) rejects an oversell
        conn.execute("UPDATE stock SET qty = ? WHERE sku = ?", (qty, sku))
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return int(qty)


# endregion


# region rename-steps
RENAME_REGION = [
    # expand: add the new column, then let new code leave the old one empty
    (3, "ALTER TABLE customer ADD COLUMN country TEXT;"),
    (4, "ALTER TABLE customer ALTER COLUMN region DROP NOT NULL;"),
    # migrate: copy what old code wrote (rerun until no NULLs are left)
    (5, "UPDATE customer SET country = region WHERE country IS NULL;"),
    # contract: only after no deployed code reads or writes region
    (6, "ALTER TABLE customer DROP COLUMN region;"),
]
# endregion


# region backfill
def backfill_country(conn: sqlite3.Connection, batch: int = 10_000) -> int:
    """Copy region into country one id range at a time; return rows copied."""
    last = conn.execute("SELECT coalesce(max(id), 0) FROM customer")
    copied = 0
    for low in range(0, last.fetchone()[0], batch):
        conn.execute("BEGIN IMMEDIATE")  # one short write lock per batch
        done = conn.execute(
            "UPDATE customer SET country = region "
            "WHERE id > ? AND id <= ? AND country IS NULL",
            (low, low + batch),
        )
        conn.execute("COMMIT")
        copied += done.rowcount
    return copied


# endregion


# region query-plan
def query_plan(
    conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()
) -> list[str]:
    """Return the plan SQLite would use for a query; it is not run."""
    rows = conn.execute("EXPLAIN QUERY PLAN " + sql, tuple(params))
    return [str(row["detail"]) for row in rows]


# endregion
