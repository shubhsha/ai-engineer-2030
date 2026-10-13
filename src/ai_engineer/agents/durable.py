"""Chapter 31: durable steps that survive a crash.

A workflow runs its side effects through ``Run.step``. Each completed
step is recorded in a journal; after a crash, the same run replays the
recorded results instead of repeating the work, and continues from the
first step that never finished.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

T = TypeVar("T")


class NonDeterminism(RuntimeError):
    """The workflow asked for different steps than it did last time."""


class Journal:
    """Completed steps, keyed by run id and position, in SQLite."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS step (run TEXT, seq INTEGER, "
            "name TEXT NOT NULL, result TEXT NOT NULL, "
            "PRIMARY KEY (run, seq))"
        )

    def get(self, run: str, seq: int) -> tuple[str, Any] | None:
        row = self._db.execute(
            "SELECT name, result FROM step WHERE run = ? AND seq = ?",
            (run, seq),
        ).fetchone()
        return None if row is None else (row[0], json.loads(row[1]))

    def put(self, run: str, seq: int, name: str, result: Any) -> None:
        self._db.execute(
            "INSERT INTO step VALUES (?, ?, ?, ?)",
            (run, seq, name, json.dumps(result)),
        )


class Run:
    def __init__(self, journal: Journal, run_id: str) -> None:
        self.journal, self.run_id = journal, run_id
        self.seq = self.replayed = 0

    def step(self, name: str, fn: Callable[[], T]) -> T:
        """Return the recorded result of this step, or run and record it.

        Steps are matched by position, so the workflow must request them
        in the same order on every run. A crash after ``fn`` succeeds but
        before it is recorded repeats ``fn`` on replay: steps with side
        effects still need idempotency keys (Chapter 9).
        """
        seq, self.seq = self.seq, self.seq + 1
        recorded = self.journal.get(self.run_id, seq)
        if recorded is not None:
            if recorded[0] != name:
                found = f"step {seq} was {recorded[0]!r}, now {name!r}"
                raise NonDeterminism(found)
            self.replayed += 1
            return cast(T, recorded[1])
        result = fn()
        self.journal.put(self.run_id, seq, name, result)
        return result
