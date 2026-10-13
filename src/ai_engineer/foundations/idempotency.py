"""Chapter 9: idempotent writes and RFC 9457 problem details."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ai_engineer.foundations.abstractions import Clock, SystemClock, WallClock

JSON = dict[str, Any]
log = logging.getLogger(__name__)


def fingerprint(payload: Mapping[str, Any]) -> str:
    """A stable hash of a JSON body: key order and spacing do not matter."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class IdempotencyConflict(Exception):
    """The same key arrived with a different body: a client bug."""


class IdempotencyInProgress(Exception):
    """The first attempt with this key is still running: answer 409."""


class IdempotencyBackend(Protocol):
    """What an endpoint needs from a store: run a handler once per key."""

    def execute(
        self, key: str, payload: Mapping[str, Any], handler: Callable[[], JSON]
    ) -> tuple[JSON, bool]: ...


@dataclass(frozen=True, slots=True)
class _Entry:
    fingerprint: str
    response: JSON
    expires_at: float


@dataclass
class IdempotencyStore:
    """Remember responses by idempotency key for ``ttl`` seconds.

    In-memory and single-process, and it checks the key before running
    the handler but claims it only after: two attempts that arrive
    together can both run it. ``SqliteIdempotencyStore`` closes that gap.
    """

    ttl: float = 24 * 3600
    clock: Clock = field(default_factory=SystemClock)
    _entries: dict[str, _Entry] = field(default_factory=dict)

    def execute(
        self, key: str, payload: Mapping[str, Any], handler: Callable[[], JSON]
    ) -> tuple[JSON, bool]:
        """Run ``handler`` at most once per key.

        Returns ``(response, replayed)``.
        """
        now, digest = self.clock.now(), fingerprint(payload)
        entry = self._entries.get(key)
        if entry is not None and entry.expires_at > now:
            if entry.fingerprint != digest:
                raise IdempotencyConflict(key)
            return entry.response, True
        response = handler()
        self._entries[key] = _Entry(digest, response, now + self.ttl)
        return response, False

    def purge(self) -> int:
        """Forget entries past their ttl and return how many."""
        now = self.clock.now()
        stale = [k for k, e in self._entries.items() if e.expires_at <= now]
        for key in stale:
            del self._entries[key]
        return len(stale)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    owner TEXT NOT NULL,      -- the attempt that holds the claim
    response TEXT,            -- NULL until that attempt finishes
    expires_at REAL NOT NULL  -- end of the lease, then of the ttl
)
"""


@dataclass
class SqliteIdempotencyStore:
    """An idempotency store that every process and server can share.

    Each attempt claims the key in one atomic statement *before* the
    handler runs, so of any number of simultaneous attempts exactly one
    runs it. A running claim lasts ``lease`` seconds, so a server that
    dies mid-request blocks the key for a minute, not a day; the lease
    must outlast the slowest handler. Expiry times are stored, so they
    come from a wall clock.
    """

    path: str | Path
    ttl: float = 24 * 3600
    lease: float = 60.0
    clock: Clock = field(default_factory=WallClock)

    def __post_init__(self) -> None:
        with closing(self._connect()) as db:
            db.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # Autocommit: each statement is its own transaction. The timeout
        # makes a writer wait for the lock instead of failing at once.
        return sqlite3.connect(self.path, timeout=5.0, isolation_level=None)

    def execute(
        self, key: str, payload: Mapping[str, Any], handler: Callable[[], JSON]
    ) -> tuple[JSON, bool]:
        """Run ``handler`` at most once per key, across processes."""
        now, digest = self.clock.now(), fingerprint(payload)
        owner = uuid.uuid4().hex  # identifies this attempt's claim
        with closing(self._connect()) as db:
            # One statement, so the database applies it atomically: insert
            # the key, take over an expired entry, or change nothing.
            claim = db.execute(
                """
                INSERT INTO idempotency (key, fingerprint, owner, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (key) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    owner = excluded.owner,
                    response = NULL,
                    expires_at = excluded.expires_at
                WHERE idempotency.expires_at <= ?
                """,
                (key, digest, owner, now + self.lease, now),
            )
            if claim.rowcount == 1:  # this attempt holds the key
                return self._run(db, key, owner, handler), False
            row = db.execute(
                "SELECT fingerprint, response FROM idempotency WHERE key = ?",
                (key,),
            ).fetchone()
        if row is not None and row[0] != digest:
            raise IdempotencyConflict(key)
        if row is None or row[1] is None:  # the first attempt is running
            raise IdempotencyInProgress(key)
        return json.loads(row[1]), True

    def purge(self) -> int:
        """Delete rows past their ttl or lease and return how many.

        Expired rows are ignored but never freed, so run this on a timer.
        """
        with closing(self._connect()) as db:
            return db.execute(
                "DELETE FROM idempotency WHERE expires_at <= ?",
                (self.clock.now(),),
            ).rowcount

    def _run(
        self,
        db: sqlite3.Connection,
        key: str,
        owner: str,
        handler: Callable[[], JSON],
    ) -> JSON:
        try:
            response = handler()
        except BaseException:
            # Release the key, so that the client's retry can run.
            db.execute(
                "DELETE FROM idempotency WHERE key = ? AND owner = ?",
                (key, owner),
            )
            raise
        stored = db.execute(
            "UPDATE idempotency SET response = ?, expires_at = ? "
            "WHERE key = ? AND owner = ?",
            (json.dumps(response), self.clock.now() + self.ttl, key, owner),
        )
        if stored.rowcount == 0:
            log.warning(
                "lease on idempotency key %r expired before the handler "
                "finished; it may have run twice",
                key,
            )
        return response


def problem(
    status: int,
    title: str,
    detail: str,
    type_uri: str = "about:blank",
    **extra: Any,
) -> JSON:
    """An RFC 9457 problem-details body (media type
    ``application/problem+json``)."""
    return {
        "type": type_uri,
        "title": title,
        "status": status,
        "detail": detail,
        **extra,
    }
