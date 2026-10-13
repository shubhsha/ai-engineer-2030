import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ai_engineer.foundations.idempotency import (
    IdempotencyConflict,
    IdempotencyInProgress,
    IdempotencyStore,
    SqliteIdempotencyStore,
    fingerprint,
    problem,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path, clock):
    """Every test that asks for ``store`` runs once with each store."""
    if request.param == "memory":
        return IdempotencyStore(ttl=10, clock=clock)
    return SqliteIdempotencyStore(tmp_path / "keys.db", ttl=10, clock=clock)


def test_fingerprint_ignores_key_order():
    assert fingerprint({"a": 1, "b": [1, 2]}) == fingerprint(
        {"b": [1, 2], "a": 1}
    )


def test_handler_runs_once_per_key(store):
    calls = []

    def handler():
        calls.append(1)
        return {"id": len(calls)}

    first = store.execute("k1", {"x": 1}, handler)
    again = store.execute("k1", {"x": 1}, handler)
    assert first == ({"id": 1}, False)
    assert again == ({"id": 1}, True)
    assert len(calls) == 1


def test_reusing_a_key_with_another_body_is_a_conflict(store):
    store.execute("k1", {"x": 1}, lambda: {})
    with pytest.raises(IdempotencyConflict):
        store.execute("k1", {"x": 2}, lambda: {})


def test_entries_expire_exactly_at_the_ttl(store, clock):
    store.execute("k1", {}, lambda: {"n": 1})
    clock.advance(9.5)
    assert store.execute("k1", {}, lambda: {"n": 2}) == ({"n": 1}, True)
    clock.advance(0.5)  # exactly ten seconds: the entry has expired
    assert store.execute("k1", {}, lambda: {"n": 3}) == ({"n": 3}, False)


def test_a_failed_attempt_leaves_the_key_free(store):
    def charge():
        raise ConnectionError("payment provider unavailable")

    with pytest.raises(ConnectionError):
        store.execute("k1", {}, charge)
    assert store.execute("k1", {}, lambda: {"n": 1}) == ({"n": 1}, False)


def test_twenty_attempts_at_once_run_the_handler_once(tmp_path):
    store = SqliteIdempotencyStore(tmp_path / "keys.db")
    start = threading.Barrier(20)  # releases all twenty threads together
    refused = threading.Semaphore(0)
    runs = []

    def charge():
        runs.append(1)
        for _ in range(19):  # hold the key until the others are refused
            refused.acquire(timeout=5)
        return {"charge_id": 1}

    def attempt(_):
        start.wait(timeout=5)
        try:
            store.execute("order-7", {"amount": 5}, charge)
        except IdempotencyInProgress:
            refused.release()
            return 409
        return 201

    with ThreadPoolExecutor(max_workers=20) as pool:
        statuses = list(pool.map(attempt, range(20)))
    assert len(runs) == 1
    assert sorted(statuses) == [201] + [409] * 19


def test_a_claim_lasts_only_for_its_lease(tmp_path, clock, caplog):
    store = SqliteIdempotencyStore(tmp_path / "keys.db", lease=60, clock=clock)

    def stuck():
        # While this attempt runs, a retry is refused...
        with pytest.raises(IdempotencyInProgress):
            store.execute("k1", {}, lambda: {"n": 2})
        # ...until the lease ends, as if this attempt's server had died.
        clock.advance(60)
        assert store.execute("k1", {}, lambda: {"n": 2}) == ({"n": 2}, False)
        return {"n": 1}

    store.execute("k1", {}, stuck)
    assert "may have run twice" in caplog.text
    assert store.execute("k1", {}, lambda: {"n": 3}) == ({"n": 2}, True)


def test_problem_details_shape():
    body = problem(422, "Unprocessable", "bad", instance="/v1/quotes/1")
    assert body["type"] == "about:blank"
    assert body["status"] == 422
    assert body["instance"] == "/v1/quotes/1"


def run_with_an_overlapping_retry(store):
    """A handler that calls ``execute`` again plays a retry that arrives
    while the first attempt is still running: no threads, no timing."""
    runs = []

    def second():
        runs.append("second")
        return {"n": 2}

    def first():
        runs.append("first")
        try:
            store.execute("k1", {}, second)
        except IdempotencyInProgress:
            runs.append("refused")
        return {"n": 1}

    store.execute("k1", {}, first)
    return runs


def test_the_memory_store_runs_the_handler_for_an_overlapping_retry():
    runs = run_with_an_overlapping_retry(IdempotencyStore())
    assert runs == ["first", "second"]  # it checked before either acted


def test_sqlite_refuses_an_overlapping_retry(tmp_path):
    store = SqliteIdempotencyStore(tmp_path / "keys.db")
    assert run_with_an_overlapping_retry(store) == ["first", "refused"]


def test_purge_frees_only_the_expired_entries(store, clock):
    store.execute("old", {}, lambda: {"n": 1})
    clock.advance(6)
    store.execute("new", {}, lambda: {"n": 2})
    clock.advance(5)  # "old" is eleven seconds old, "new" only five
    assert store.purge() == 1
    assert store.execute("new", {}, lambda: {"n": 3}) == ({"n": 2}, True)
    assert store.execute("old", {}, lambda: {"n": 4}) == ({"n": 4}, False)
