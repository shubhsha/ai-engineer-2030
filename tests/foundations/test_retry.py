import asyncio
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai_engineer.foundations.retry import (
    CircuitBreaker,
    RetryBudget,
    RetryError,
    backoff_delays,
    decorrelated_jitter_delays,
    equal_jitter_delays,
    is_retryable_status,
    retry_after_seconds,
    retry_async,
)


@given(
    st.integers(min_value=1, max_value=12),
    st.floats(min_value=0.01, max_value=2),
    st.floats(min_value=0.5, max_value=60),
)
def test_delays_are_bounded_by_the_capped_exponential(attempts, base, cap):
    waits = list(
        backoff_delays(
            base=base, cap=cap, attempts=attempts, rng=random.Random(1)
        )
    )
    assert len(waits) == attempts - 1
    for n, wait in enumerate(waits):
        assert 0 <= wait <= min(cap, base * 2**n)


def test_transient_errors_are_retried_until_success(rng):
    calls, slept = [], []

    async def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("reset")
        return "ok"

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)  # record the wait, and return at once

    result = asyncio.run(retry_async(flaky, sleep=fake_sleep, rng=rng))
    assert result == "ok"
    assert len(calls) == 3
    assert len(slept) == 2  # two waits between three attempts
    assert 0 <= slept[0] <= 0.5  # base * 2**0
    assert 0 <= slept[1] <= 1.0  # base * 2**1


def test_permanent_errors_are_not_retried():
    calls = []

    async def bad_request() -> str:
        calls.append(1)
        raise ValueError("400: malformed")

    with pytest.raises(ValueError, match="malformed"):
        asyncio.run(retry_async(bad_request))
    assert len(calls) == 1


def test_gives_up_after_the_last_attempt():
    async def down() -> str:
        raise TimeoutError

    async def no_sleep(seconds: float) -> None:
        return None

    with pytest.raises(RetryError) as err:
        asyncio.run(retry_async(down, attempts=3, sleep=no_sleep))
    assert err.value.attempts == 3
    assert isinstance(err.value.last, TimeoutError)


@pytest.mark.parametrize("status", [408, 429, 502, 503, 504])
def test_transient_statuses_are_retryable(status):
    assert is_retryable_status(status)


@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 422])
def test_permanent_statuses_are_not_retryable(status):
    assert not is_retryable_status(status)


def test_retry_after_reads_seconds_and_http_dates():
    now = 1_790_000_000.0  # seconds since the epoch; a FakeClock reading
    assert retry_after_seconds("120", now) == 120.0
    assert retry_after_seconds(" 3 ", now) == 3.0
    in_30s = "Mon, 21 Sep 2026 14:13:50 GMT"  # now + 30 s
    assert retry_after_seconds(in_30s, now) == 30.0
    past = "Mon, 21 Sep 2026 14:13:00 GMT"  # now - 20 s
    assert retry_after_seconds(past, now) == 0.0
    assert retry_after_seconds("soon", now) is None
    assert retry_after_seconds("-5", now) is None


@given(
    st.integers(min_value=2, max_value=12),
    st.floats(min_value=0.01, max_value=2),
    st.floats(min_value=0.5, max_value=60),
)
def test_jitter_variants_stay_inside_their_bounds(attempts, base, cap):
    equal = list(
        equal_jitter_delays(
            base=base, cap=cap, attempts=attempts, rng=random.Random(2)
        )
    )
    for n, wait in enumerate(equal):
        top = min(cap, base * 2**n)
        assert top / 2 <= wait <= top
    decorrelated = list(
        decorrelated_jitter_delays(
            base=base, cap=cap, attempts=attempts, rng=random.Random(3)
        )
    )
    assert len(decorrelated) == attempts - 1
    for wait in decorrelated:
        assert min(base, cap) <= wait <= cap


def test_budget_caps_retries_near_ten_percent_of_calls():
    budget = RetryBudget(ratio=0.1, burst=10)
    retries = 0
    for _ in range(1_000):  # every call fails and asks for one retry
        budget.record_attempt()
        if budget.allow_retry():
            retries += 1
    assert 105 <= retries <= 110  # 10 saved up, plus 0.1 per call
    assert not budget.allow_retry()  # the bucket is empty


def test_budget_refills_only_up_to_the_burst():
    budget = RetryBudget(ratio=0.5, burst=2)
    for _ in range(100):
        budget.record_attempt()
    assert budget.tokens == 2


def test_breaker_opens_then_probes_after_the_cooldown(clock):
    breaker = CircuitBreaker(clock=clock, threshold=3, cooldown=30)
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()  # refused without calling the service
    clock.advance(29)
    assert not breaker.allow()
    clock.advance(1)
    assert breaker.state == "half-open"
    assert breaker.allow()  # the probe
    assert not breaker.allow()  # everyone else waits for its result
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow()


def test_breaker_reopens_when_the_probe_fails(clock):
    breaker = CircuitBreaker(clock=clock, threshold=1, cooldown=10)
    breaker.record_failure()
    clock.advance(10)
    assert breaker.allow()  # the probe
    breaker.record_failure()
    assert breaker.state == "open"
    clock.advance(10)
    assert breaker.state == "half-open"


def test_a_success_resets_the_count_of_failures_in_a_row(clock):
    breaker = CircuitBreaker(clock=clock, threshold=3)
    for _ in range(10):
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
    assert breaker.state == "closed"
