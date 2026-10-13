import asyncio

import pytest

from ai_engineer.foundations.concurrency import gather_bounded


def test_results_keep_input_order_and_limit_is_respected():
    in_flight = peak = 0

    def make(i: int):
        async def job() -> int:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.001 * (5 - i % 5))
            in_flight -= 1
            return i * i

        return job

    results = asyncio.run(
        gather_bounded([make(i) for i in range(20)], limit=4)
    )
    assert results == [i * i for i in range(20)]
    assert peak <= 4


def test_one_failure_cancels_the_rest():
    cancelled = []

    async def slow() -> int:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return 1

    async def boom() -> int:
        await asyncio.sleep(0.01)
        raise ConnectionError("upstream reset")

    with pytest.raises(ExceptionGroup) as err:
        asyncio.run(gather_bounded([slow, slow, boom], limit=3))
    assert err.group_contains(ConnectionError)
    assert cancelled == [True, True]


def test_timeout_cancels_everything():
    async def slow() -> int:
        await asyncio.sleep(10)
        return 1

    with pytest.raises(TimeoutError):
        asyncio.run(gather_bounded([slow] * 3, limit=2, timeout=0.05))


def test_limit_must_be_positive():
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(gather_bounded([], limit=0))
