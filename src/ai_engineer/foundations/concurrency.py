"""Chapter 5: bounded, cancellable concurrency with asyncio."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
Job = Callable[[], Awaitable[T]]


async def gather_bounded(
    jobs: Sequence[Job[T]], *, limit: int, timeout: float | None = None
) -> list[T]:
    """Run ``jobs`` with at most ``limit`` in flight; keep input order.

    Structured concurrency: if one job fails, the TaskGroup cancels the
    others and raises an ExceptionGroup. If the batch exceeds ``timeout``
    seconds, every job is cancelled and TimeoutError is raised. Nothing
    is ever left running in the background.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    gate = asyncio.Semaphore(limit)

    async def run(job: Job[T]) -> T:
        async with gate:
            return await job()

    async with asyncio.timeout(timeout), asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run(job)) for job in jobs]
    return [task.result() for task in tasks]
