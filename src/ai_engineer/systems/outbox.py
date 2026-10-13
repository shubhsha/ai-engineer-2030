"""The transactional outbox, an idempotent consumer and a saga (chapter 55).

Everything runs on SQLite so a test can crash the process between any two
steps and look at what survived.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, item TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS processed (message_id INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS emails (message_id INTEGER, body TEXT);
"""


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    payload: str


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


# region place
def place_order(conn: sqlite3.Connection, order_id: str, item: str) -> None:
    """Write the business row and its message in ONE transaction."""
    with conn:
        conn.execute("INSERT INTO orders VALUES (?, ?)", (order_id, item))
        conn.execute(
            "INSERT INTO outbox (payload) VALUES (?)",
            (f"order {order_id}: {item}",),
        )


# endregion place


# region relay
def relay(conn: sqlite3.Connection, send: Callable[[Message], None]) -> int:
    """Deliver every unacknowledged message at least once.

    The row is marked delivered only after ``send`` returns. A crash in
    between leaves it unmarked, so the next run sends it again.
    """
    rows = conn.execute(
        "SELECT id, payload FROM outbox WHERE delivered = 0 ORDER BY id"
    ).fetchall()
    for row_id, payload in rows:
        send(Message(row_id, payload))
        with conn:
            conn.execute(
                "UPDATE outbox SET delivered = 1 WHERE id = ?", (row_id,)
            )
    return len(rows)


# endregion relay


# region consumer
def consume_idempotent(conn: sqlite3.Connection, msg: Message) -> bool:
    """Apply the effect once per message id; return False for a repeat."""
    with conn:
        seen = conn.execute(
            "INSERT OR IGNORE INTO processed VALUES (?)", (msg.id,)
        )
        if seen.rowcount == 0:
            return False
        conn.execute("INSERT INTO emails VALUES (?, ?)", (msg.id, msg.payload))
    return True


def consume_naive(conn: sqlite3.Connection, msg: Message) -> bool:
    """The same effect with no memory of what it has already done."""
    with conn:
        conn.execute("INSERT INTO emails VALUES (?, ?)", (msg.id, msg.payload))
    return True


# endregion consumer


# region saga
@dataclass(frozen=True, slots=True)
class SagaStep:
    name: str
    do: Callable[[], None]
    undo: Callable[[], None]


def run_saga(steps: Sequence[SagaStep]) -> tuple[bool, list[str]]:
    """Run steps in order; on a failure, undo the finished ones in reverse.

    Returns (succeeded, log). Each undo must itself be safe to repeat.
    """
    log: list[str] = []
    done: list[SagaStep] = []
    for step in steps:
        try:
            step.do()
        except Exception as error:
            log.append(f"{step.name} failed: {error}")
            for finished in reversed(done):
                finished.undo()
                log.append(f"undo {finished.name}")
            return False, log
        done.append(step)
        log.append(f"did {step.name}")
    return True, log


# endregion saga
