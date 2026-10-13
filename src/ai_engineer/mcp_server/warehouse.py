"""Chapter 36: a read-only MCP server for a SQLite warehouse.

Three tools (list_tables, describe_table, query) give any MCP client,
such as Claude Code, read access to one database. Safety comes in
layers, so that no single mistake opens a write path: a read-only
connection, SQLite's query_only pragma, an authorizer that permits only
reads, a time limit and a row limit.

Run it with ``python -m ai_engineer.mcp_server.warehouse --db PATH``.
A stdio server owns stdout for the protocol, so it never prints;
logging goes to stderr.
"""

from __future__ import annotations

import argparse
import random
import re
import sqlite3
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from ai_engineer.enterprise.warehouse import SqliteWarehouse

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:  # the MCP SDK's pydantic rejects typing.TypedDict before 3.12
    from typing_extensions import TypedDict

MAX_ROWS = 200
TIME_LIMIT_S = 2.0
INSTRUCTIONS = (
    "Read-only access to a sales warehouse. Call list_tables, then "
    "describe_table, before writing SQL. Aggregate and filter in SQL: "
    f"query returns at most {MAX_ROWS} rows."
)


class QueryRejected(ValueError):
    """A query this server will not run, with a reason a model can use."""


class ColumnInfo(TypedDict):
    name: str
    type: str
    nullable: bool


class QueryResult(TypedDict):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


_READS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)
_BLOCKED_FUNCTIONS = frozenset({"load_extension", "fts3_tokenizer"})


def _authorize(
    action: int,
    arg1: str | None,
    arg2: str | None,
    database: str | None,
    source: str | None,
) -> int:
    """Allow reading and nothing else: no writes, ATTACH or PRAGMA."""
    if action == sqlite3.SQLITE_FUNCTION and arg2 in _BLOCKED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in _READS else sqlite3.SQLITE_DENY


def _deadline(seconds: float) -> Callable[[], int]:
    end = time.monotonic() + seconds
    return lambda: int(time.monotonic() > end)  # nonzero stops the query


def open_read_only(path: str | Path) -> SqliteWarehouse:
    """The Chapter 13 adapter, opened read-only, with writes off twice."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"no database at {path}")
    warehouse = SqliteWarehouse(path, read_only=True)
    warehouse.connection.execute("PRAGMA query_only = ON")
    return warehouse


def list_tables(path: str | Path) -> list[str]:
    with closing(open_read_only(path)) as warehouse:
        return warehouse.tables()


def describe_table(path: str | Path, table: str) -> list[ColumnInfo]:
    with closing(open_read_only(path)) as warehouse:
        try:
            columns = warehouse.columns(table)
        except KeyError:
            raise QueryRejected(
                f"No table named {table!r}. Call list_tables to see them."
            ) from None
    return [
        {"name": c.name, "type": c.type, "nullable": c.nullable}
        for c in columns
    ]


_COMMENTS = re.compile(r"^(?:\s+|--[^\n]*\n?|/\*.*?\*/)*", re.DOTALL)


def check_query(sql: str) -> str:
    """The friendly first layer: one SELECT, with a helpful refusal."""
    body = _COMMENTS.sub("", sql).strip().rstrip(";").strip()
    first = body.split(None, 1)[0].upper() if body else ""
    if first not in ("SELECT", "WITH"):
        raise QueryRejected(
            "Only a single read-only SELECT (or WITH ... SELECT) runs here."
        )
    return body


def run_query(
    path: str | Path,
    sql: str,
    *,
    max_rows: int = MAX_ROWS,
    time_limit: float = TIME_LIMIT_S,
) -> QueryResult:
    """Run one read-only query and return at most ``max_rows`` rows."""
    body = check_query(sql)
    with closing(open_read_only(path)) as warehouse:
        conn = warehouse.connection
        conn.set_authorizer(_authorize)
        conn.set_progress_handler(_deadline(time_limit), 1_000)
        try:
            cursor = conn.execute(body)
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.Error as exc:
            raise QueryRejected(_explain(exc, time_limit)) from None
        columns = [d[0] for d in cursor.description or ()]
    return {
        "columns": columns,
        "rows": [list(row) for row in rows[:max_rows]],
        "truncated": len(rows) > max_rows,
    }


def _explain(exc: sqlite3.Error, time_limit: float) -> str:
    text = str(exc)
    if "not authorized" in text:
        return "This server only reads. Writes, ATTACH and PRAGMA are off."
    if "interrupted" in text:
        return (
            f"The query ran longer than {time_limit:g} s and was stopped. "
            "Filter or aggregate more, or add a LIMIT."
        )
    if "one statement" in text:
        return "Send one statement at a time."
    return f"SQL error: {text}. Check names with describe_table."


def build_server(path: str | Path) -> Any:
    """The MCP server. The SDK is imported here, so the rest of this
    module works and is tested without it."""
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    server = MCPServer("warehouse", instructions=INSTRUCTIONS)
    reads = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    @server.tool(name="list_tables", annotations=reads)
    def tables_tool() -> list[str]:
        """List the tables in the warehouse."""
        return list_tables(path)

    @server.tool(name="describe_table", annotations=reads)
    def describe_tool(table: str) -> list[ColumnInfo]:
        """List one table's columns with their types and nullability."""
        try:
            return describe_table(path, table)
        except QueryRejected as exc:
            raise ToolError(str(exc)) from None

    @server.tool(name="query", annotations=reads)
    def query_tool(sql: str) -> QueryResult:
        """Run one read-only SELECT and return columns and rows.

        At most 200 rows come back; ``truncated`` is true when more
        matched. Aggregate in SQL instead of reading raw rows.
        """
        try:
            return run_query(path, sql)
        except QueryRejected as exc:
            raise ToolError(str(exc)) from None

    return server


def create_demo_db(path: str | Path, *, seed: int = 2030) -> None:
    """Write a small synthetic sales database: invented names only."""
    rng = random.Random(seed)
    names = [
        f"{a} {b}"
        for a in ("Cedar", "Harbor", "Juniper", "Maple", "Quartz", "Willow")
        for b in ("Labs", "Foods", "Logistics", "Studio", "Health")
    ]
    regions = ("North", "South", "East", "West")
    products = [
        (1, "Sensor kit", "Hardware", 120.0),
        (2, "Gateway", "Hardware", 340.0),
        (3, "Analytics seat", "Software", 45.0),
        (4, "Support plan", "Services", 900.0),
        (5, "Training day", "Services", 1500.0),
    ]
    statuses = ("paid", "paid", "paid", "refunded", "open")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE customers (id INTEGER PRIMARY KEY,
                name TEXT NOT NULL, region TEXT NOT NULL);
            CREATE TABLE products (id INTEGER PRIMARY KEY,
                name TEXT NOT NULL, category TEXT NOT NULL,
                unit_price REAL NOT NULL);
            CREATE TABLE orders (id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                product_id INTEGER NOT NULL REFERENCES products(id),
                quantity INTEGER NOT NULL, ordered_on TEXT NOT NULL,
                status TEXT NOT NULL);
            """
        )
        conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [(i, n, rng.choice(regions)) for i, n in enumerate(names, 1)],
        )
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    i,
                    rng.randint(1, len(names)),
                    rng.randint(1, len(products)),
                    rng.randint(1, 20),
                    f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}",
                    rng.choice(statuses),
                )
                for i in range(1, 501)
            ],
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Read-only warehouse MCP.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--create-demo",
        action="store_true",
        help="create a synthetic demo database if none exists",
    )
    args = parser.parse_args(argv)
    if args.create_demo and not args.db.exists():
        args.db.parent.mkdir(parents=True, exist_ok=True)
        create_demo_db(args.db)
    build_server(args.db).run(transport="stdio")


if __name__ == "__main__":
    main()
