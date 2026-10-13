import asyncio
import os
import signal

from ai_engineer.foundations.shutdown import run_until_stopped


def test_sigterm_lets_the_current_job_finish() -> None:
    stop = asyncio.Event()
    handled: list[int] = []

    async def handle(job: int) -> None:
        if job == 2:
            os.kill(os.getpid(), signal.SIGTERM)  # what docker stop sends
            await asyncio.wait_for(stop.wait(), timeout=1)
        handled.append(job)  # the job in progress still finishes

    done = asyncio.run(run_until_stopped(range(10), handle, stop=stop))
    assert handled == [0, 1, 2]  # job 3 never started
    assert done == 3
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
