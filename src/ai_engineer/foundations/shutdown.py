"""Chapter 6: when asked to stop, finish the current job, then stop."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

T = TypeVar("T")

STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


async def run_until_stopped(
    jobs: Iterable[T],
    handle: Callable[[T], Awaitable[None]],
    *,
    stop: asyncio.Event | None = None,
) -> int:
    """Handle jobs in order until they run out or a stop signal arrives.

    SIGTERM, which ``docker stop`` and Kubernetes send, and SIGINT, which
    Ctrl-C sends, set ``stop`` instead of ending the process mid-job. The
    loop checks it between jobs, so the job in progress finishes and no
    new one starts. Returns the number of jobs handled. Unix only, and
    only in the main thread, which is where Python receives signals.
    """
    stop = stop if stop is not None else asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in STOP_SIGNALS:
        loop.add_signal_handler(sig, stop.set)
    handled = 0
    try:
        for job in jobs:
            if stop.is_set():
                break
            await handle(job)
            handled += 1
    finally:
        for sig in STOP_SIGNALS:
            loop.remove_signal_handler(sig)
    return handled
