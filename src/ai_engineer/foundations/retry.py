"""Chapter 10: retries, backoff, retry budgets and a circuit breaker."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import TypeVar

from ai_engineer.foundations.abstractions import Clock

T = TypeVar("T")
Sleep = Callable[[float], Awaitable[None]]


def backoff_delays(
    *,
    base: float = 0.5,
    cap: float = 30.0,
    attempts: int = 5,
    rng: random.Random | None = None,
) -> Iterator[float]:
    """Yield the ``attempts - 1`` waits between ``attempts`` tries.

    Wait n is uniform on [0, min(cap, base * 2**n)] ("full jitter").
    Randomizing the whole interval spreads out retries from many
    clients, so a recovering service is not hit by synchronized waves.
    """
    rng = rng or random.Random()
    for n in range(attempts - 1):
        yield rng.uniform(0, min(cap, base * 2**n))


class RetryError(Exception):
    def __init__(self, attempts: int, last: BaseException) -> None:
        super().__init__(f"gave up after {attempts} attempts: {last!r}")
        self.attempts = attempts
        self.last = last


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    retry_on: tuple[type[BaseException], ...] = (
        ConnectionError,
        TimeoutError,
    ),
    attempts: int = 5,
    base: float = 0.5,
    cap: float = 30.0,
    sleep: Sleep = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call ``fn`` until it succeeds, retrying only transient errors."""
    waits = backoff_delays(base=base, cap=cap, attempts=attempts, rng=rng)
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on as exc:
            if attempt == attempts:
                raise RetryError(attempts, exc) from exc
            await sleep(next(waits))
    raise AssertionError("unreachable")


RETRYABLE_STATUS = frozenset({408, 429, 502, 503, 504})


def is_retryable_status(status: int) -> bool:
    """True when a response says "try again", false when it says "no".

    These five describe the moment, not the request. A 400, 401, 403,
    404 or 422 fails the same way every time, so retrying only adds load.
    """
    return status in RETRYABLE_STATUS


def retry_after_seconds(value: str, now: float) -> float | None:
    """Read a ``Retry-After`` header: seconds, or an HTTP date.

    ``now`` is seconds since the Unix epoch (a ``WallClock`` reading).
    Returns the seconds to wait, never negative, or None if unreadable.
    """
    text = value.strip()
    if text.isascii() and text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, when.timestamp() - now)


# region jitter-variants
def equal_jitter_delays(
    *,
    base: float = 0.5,
    cap: float = 30.0,
    attempts: int = 5,
    rng: random.Random | None = None,
) -> Iterator[float]:
    """Half of each wait is fixed and half is random ("equal jitter")."""
    rng = rng or random.Random()
    for n in range(attempts - 1):
        top = min(cap, base * 2**n)
        yield top / 2 + rng.uniform(0, top / 2)


def decorrelated_jitter_delays(
    *,
    base: float = 0.5,
    cap: float = 30.0,
    attempts: int = 5,
    rng: random.Random | None = None,
) -> Iterator[float]:
    """Each wait is random between ``base`` and 3 times the last wait."""
    rng = rng or random.Random()
    wait = base
    for _ in range(attempts - 1):
        wait = min(cap, rng.uniform(base, wait * 3))
        yield wait


# endregion


@dataclass
class RetryBudget:
    """Allow retries up to ``ratio`` times the number of first attempts.

    Each first attempt earns ``ratio`` tokens (at most ``burst`` are
    kept) and each retry spends one. At a ratio of 0.1, ten calls earn
    one retry, so retries add at most about 10% to the load.
    """

    ratio: float = 0.1
    burst: float = 10.0
    tokens: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.burst

    def record_attempt(self) -> None:
        self.tokens = min(self.burst, self.tokens + self.ratio)

    def allow_retry(self) -> bool:
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True


@dataclass
class CircuitBreaker:
    """Stop calling a dependency that keeps failing, then probe it.

    Closed: calls flow and failures in a row are counted. Open: calls
    are refused for ``cooldown`` seconds. Half-open: the next call is a
    probe; success closes the circuit and failure opens it again. The
    probe restarts the timer, so one that never reports back costs one
    more cooldown, not a stuck circuit.
    """

    clock: Clock
    threshold: int = 5
    cooldown: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock.now() - self.opened_at < self.cooldown:
            return "open"
        return "half-open"

    def allow(self) -> bool:
        state = self.state
        if state == "half-open":
            self.opened_at = self.clock.now()
        return state != "open"

    def record_success(self) -> None:
        self.failures, self.opened_at = 0, None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self.clock.now()
