from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai_engineer.foundations.abstractions import (
    FakeClock,
    Money,
    Timer,
    batched,
    wait_until,
)

amounts = st.decimals(
    min_value=-(10**6),
    max_value=10**6,
    places=2,
    allow_nan=False,
    allow_infinity=False,
)


def test_money_rounds_half_even_to_cents():
    assert Money(Decimal("0.125")).amount == Decimal("0.12")
    assert Money(Decimal("0.135")).amount == Decimal("0.14")


def test_money_refuses_mixed_currencies():
    with pytest.raises(ValueError, match="cannot add"):
        Money(Decimal(1), "USD") + Money(Decimal(1), "EUR")


def test_money_is_a_value_object():
    assert Money(Decimal("1.0")) == Money(Decimal("1.00"))
    assert len({Money(Decimal(1)), Money(Decimal("1.00"))}) == 1


# region money-properties
@given(amounts, amounts, amounts)
def test_money_addition_is_associative_and_commutative(a, b, c):
    x, y, z = Money(a), Money(b), Money(c)
    assert x + y == y + x
    assert (x + y) + z == x + (y + z)
    assert x + Money.zero() == x


# endregion


# region batched-properties
@given(st.lists(st.integers()), st.integers(min_value=1, max_value=7))
def test_batched_preserves_items_and_sizes(items, size):
    batches = list(batched(items, size))
    assert [x for batch in batches for x in batch] == items
    assert all(len(batch) == size for batch in batches[:-1])
    assert all(1 <= len(batch) <= size for batch in batches)


# endregion


def test_batched_is_lazy():
    def endless():
        n = 0
        while True:
            yield n
            n += 1

    first = next(batched(endless(), 3))
    assert first == (0, 1, 2)


def test_timer_uses_injected_clock():
    clock = FakeClock()
    with Timer(clock) as t:
        clock.advance(1.5)
    assert t.elapsed == pytest.approx(1.5)


def test_fake_clock_cannot_go_backwards(clock):
    with pytest.raises(ValueError, match="forward"):
        clock.advance(-1)


def test_wait_until_returns_as_soon_as_the_condition_holds(clock):
    def sleep(seconds: float) -> None:
        clock.advance(seconds)  # a fake sleep moves the fake clock

    def job_done() -> bool:
        return clock.now() >= 0.6  # the job finishes at 0.6 s

    waited = wait_until(
        job_done, timeout=5, interval=0.25, clock=clock, sleep=sleep
    )
    assert waited == 0.75  # 0.6 rounded up to the next poll


def test_wait_until_stops_at_the_deadline(clock):
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)  # a spy: record the request,
        clock.advance(seconds)  # then let fake time pass

    with pytest.raises(TimeoutError, match="after 1 s"):
        wait_until(
            lambda: False, timeout=1, interval=0.75, clock=clock, sleep=sleep
        )
    assert slept == [0.75, 0.25]  # the last wait is cut to the deadline
    assert clock.now() == 1
