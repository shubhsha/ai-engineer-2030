"""Tests for chapter 55: clocks, quorums, the outbox and sagas."""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Callable

import pytest

from ai_engineer.systems.clocks import LamportClock, VectorClock, compare
from ai_engineer.systems.outbox import (
    Message,
    SagaStep,
    connect,
    consume_idempotent,
    consume_naive,
    place_order,
    relay,
    run_saga,
)
from ai_engineer.systems.quorum import Cluster, stale_read_rate


def test_lamport_receive_jumps_past_sender() -> None:
    a = LamportClock().tick().send()
    b = LamportClock().receive(a.time)
    assert (a.time, b.time) == (2, 3)


def test_vector_clocks_find_concurrent_events() -> None:
    p0 = VectorClock.zero(0, 2).tick()
    p1 = VectorClock.zero(1, 2).tick()
    assert compare(p0.counts, p1.counts) == "concurrent"
    got = p1.receive(p0.send().counts)
    assert compare(p0.counts, got.counts) == "before"
    assert compare(got.counts, p0.counts) == "after"
    assert compare(p0.counts, p0.counts) == "equal"


def test_lamport_cannot_see_concurrency_but_vector_can() -> None:
    a, b = LamportClock().tick(), LamportClock().tick().tick()
    assert a.time < b.time  # looks ordered, but no message links them
    va = VectorClock.zero(0, 2).tick()
    vb = VectorClock.zero(1, 2).tick().tick()
    assert compare(va.counts, vb.counts) == "concurrent"


def test_compare_rejects_mismatched_sizes() -> None:
    with pytest.raises(ValueError, match="different process"):
        compare((1,), (1, 2))


def test_overlapping_quorum_never_reads_stale(rng: random.Random) -> None:
    stats = stale_read_rate(3, 2, 2, rng, 2000)
    assert stats.stale_rate == 0.0


def test_weak_quorum_reads_stale_and_is_faster() -> None:
    weak = stale_read_rate(3, 1, 1, random.Random(2030), 2000)
    strong = stale_read_rate(3, 2, 2, random.Random(2030), 2000)
    assert weak.stale_rate > 0.1
    assert weak.mean_read_ms < strong.mean_read_ms


def test_cluster_returns_written_value(rng: random.Random) -> None:
    c = Cluster(3, rng)
    c.write("a", "x", 3)
    assert c.read("a", 1).value == "x"
    with pytest.raises(ValueError, match="between 1 and n"):
        c.read("a", 4)


class Crash(Exception):
    pass


def emails_after_crash(
    consume: Callable[[sqlite3.Connection, Message], bool],
) -> int:
    """Crash once between the effect and the mark, restart, count emails."""
    conn = connect()
    place_order(conn, "o1", "book")
    calls = 0

    def crashing(msg: Message) -> None:
        nonlocal calls
        calls += 1
        consume(conn, msg)
        if calls == 1:
            raise Crash

    with pytest.raises(Crash):
        relay(conn, crashing)
    assert relay(conn, crashing) == 1  # redelivered
    row = conn.execute("SELECT COUNT(*) FROM emails").fetchone()
    return int(row[0])


def test_crash_before_ack_duplicates_naive_but_not_idempotent() -> None:
    assert emails_after_crash(consume_naive) == 2
    assert emails_after_crash(consume_idempotent) == 1


def test_idempotent_consumer_applies_once() -> None:
    conn = connect()
    place_order(conn, "o1", "book")
    msg = Message(1, "order o1: book")
    assert consume_idempotent(conn, msg) is True
    assert consume_idempotent(conn, msg) is False
    assert conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0] == 1


def test_order_and_message_commit_together() -> None:
    conn = connect()
    place_order(conn, "o1", "book")
    with pytest.raises(sqlite3.IntegrityError):
        place_order(conn, "o1", "again")  # duplicate id rolls back both
    assert conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 1


def test_saga_compensates_in_reverse() -> None:
    def boom() -> None:
        raise RuntimeError("no seats")

    def noop() -> None:
        return None

    steps = [
        SagaStep("flight", noop, noop),
        SagaStep("hotel", noop, noop),
        SagaStep("car", boom, noop),
    ]
    ok, log = run_saga(steps)
    assert not ok
    assert log == [
        "did flight",
        "did hotel",
        "car failed: no seats",
        "undo hotel",
        "undo flight",
    ]
