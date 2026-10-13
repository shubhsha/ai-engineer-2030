"""Chapter 13: a warehouse port and its SQLite adapter.

The rest of the system depends on :class:`Warehouse`, a protocol with
three methods. SQLite, Fabric or a test double can sit behind it.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    type: str
    nullable: bool


class Warehouse(Protocol):
    """What the application needs from a warehouse, and nothing more."""

    def tables(self) -> list[str]: ...

    def columns(self, table: str) -> list[Column]: ...

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Row]: ...


class SqliteWarehouse:
    """Adapter: the Warehouse port implemented on a SQLite file."""

    def __init__(
        self, path: str | Path = ":memory:", *, read_only: bool = False
    ) -> None:
        mode = "?mode=ro" if read_only else ""
        self._conn = sqlite3.connect(
            f"file:{path}{mode}", uri=True, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def tables(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [row["name"] for row in rows]

    def columns(self, table: str) -> list[Column]:
        rows = self._conn.execute(
            'SELECT name, type, "notnull" FROM pragma_table_info(?)', (table,)
        ).fetchall()
        if not rows:
            raise KeyError(f"no such table: {table}")
        return [Column(r["name"], r["type"], not r["notnull"]) for r in rows]

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        return [dict(row) for row in self._conn.execute(sql, params)]

    def close(self) -> None:
        self._conn.close()


_SELECT_ALL = re.compile(
    r"\s*SELECT\s+\*\s+FROM\s+(\w+)\s*;?\s*", re.IGNORECASE
)


class InMemoryWarehouse:
    """Adapter: the Warehouse port on plain dicts, for tests and demos.

    It runs one query shape, ``SELECT * FROM <table>``, and refuses every
    other, so a test cannot pass on a query the fake silently ignored.
    """

    def __init__(
        self,
        schema: Mapping[str, Sequence[Column]],
        rows: Mapping[str, Sequence[Row]],
    ) -> None:
        self._schema = {name: list(cols) for name, cols in schema.items()}
        self._rows = {name: [dict(r) for r in rs] for name, rs in rows.items()}

    def tables(self) -> list[str]:
        return sorted(self._schema)

    def columns(self, table: str) -> list[Column]:
        if table not in self._schema:
            raise KeyError(f"no such table: {table}")
        return list(self._schema[table])

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        match = _SELECT_ALL.fullmatch(sql)
        if match is None or params:
            raise NotImplementedError(f"InMemoryWarehouse cannot run: {sql}")
        table = match.group(1)
        if table not in self._schema:
            raise KeyError(f"no such table: {table}")
        return [dict(r) for r in self._rows.get(table, [])]
