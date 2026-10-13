"""Chapter 3: value objects, protocols, generators and context managers."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from types import TracebackType
from typing import Protocol, Self, TypeVar

T = TypeVar("T")
CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class Money:
    """An amount in one currency, rounded to cents with banker's rounding.

    Frozen, so it can be shared and hashed; compared by value, so two
    ``Money("1.00")`` objects are equal however they were computed.
    """

    amount: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        if len(self.currency) != 3 or not self.currency.isupper():
            raise ValueError(f"not an ISO 4217 code: {self.currency!r}")
        exact = Decimal(self.amount).quantize(CENT, ROUND_HALF_EVEN)
        object.__setattr__(self, "amount", exact)

    def __add__(self, other: Money) -> Money:
        if other.currency != self.currency:
            raise ValueError(f"cannot add {other.currency} to {self.currency}")
        return Money(self.amount + other.amount, self.currency)

    def times(self, factor: Decimal | int) -> Money:
        return Money(self.amount * factor, self.currency)

    @classmethod
    def zero(cls, currency: str = "USD") -> Money:
        return cls(Decimal(0), currency)


class Clock(Protocol):
    """Anything with ``now()``. Inject it; never call time directly."""

    def now(self) -> float: ...


class SystemClock:
    """Monotonic seconds: for durations inside one process."""

    def now(self) -> float:
        return time.monotonic()


class WallClock:
    """Seconds since the Unix epoch: for times that are stored or shared.

    Its readings mean the same on every machine, unlike ``SystemClock``,
    but they can jump when the system clock is corrected.
    """

    def now(self) -> float:
        return time.time()


@dataclass
class FakeClock:
    """A clock for tests: time moves only when you say so."""

    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time only moves forward; advance by >= 0")
        self.t += seconds


def wait_until(
    condition: Callable[[], bool],
    *,
    timeout: float,
    interval: float = 0.05,
    clock: Clock | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Poll ``condition`` until it holds; return the seconds waited.

    A fixed sleep waits for the worst case on every run and still fails
    when the worst case is exceeded. Polling waits as long as the
    condition takes, rounded up to ``interval``, and gives up with a
    clear error at the deadline instead of hanging.
    """
    if timeout < 0 or interval <= 0:
        raise ValueError("need timeout >= 0 and interval > 0")
    now = (clock or SystemClock()).now
    start = now()
    deadline = start + timeout
    while not condition():
        left = deadline - now()
        if left <= 0:
            raise TimeoutError(f"condition still false after {timeout} s")
        sleep(min(interval, left))
    return now() - start


def batched(items: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    """Yield tuples of ``size`` items; the last one may be shorter.

    A generator holds one batch in memory however long the input is.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


class Timer:
    """Time a block: ``with Timer() as t: work()`` then read ``t.elapsed``."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock or SystemClock()
        self.started = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Self:
        self.started = self._clock.now()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed = self._clock.now() - self.started
