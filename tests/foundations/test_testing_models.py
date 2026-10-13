"""Chapter 11: code around a model, tested with nothing live.

The model is a script, the waiting is a recording, and the randomness is
seeded. A timeout that strikes once in a thousand calls in production
becomes a test that runs in under a millisecond.
"""

from __future__ import annotations

import asyncio
import random

from ai_engineer.foundations.retry import backoff_delays, retry_async
from ai_engineer.llm.model import Reply, Request, ScriptedModel, ask


# region model-timeouts
def test_a_model_that_times_out_twice_is_retried_with_seeded_waits(
    rng: random.Random,
) -> None:
    def timeout(request: Request) -> Reply:
        raise TimeoutError("no reply within 30 s")

    model = ScriptedModel([timeout, timeout, "42"])
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    async def call() -> Reply:
        return ask(model, "What is 6 times 7?")

    reply = asyncio.run(retry_async(call, sleep=fake_sleep, rng=rng))

    assert reply.text == "42"
    assert len(model.requests) == 3
    # The two waits are the first two draws of the seed behind `rng`.
    assert waits == list(backoff_delays(rng=random.Random(2030)))[:2]


# endregion


# region pass-rate
def test_a_five_sample_gate_on_a_90_percent_model_fails_41_percent(
    rng: random.Random,
) -> None:
    runs = 10_000
    green = sum(all(rng.random() < 0.9 for _ in range(5)) for _ in range(runs))
    # 0.9 ** 5 = 0.5905, so the gate is red in 4 runs of 10, though the
    # model did nothing wrong.
    assert abs(green / runs - 0.9**5) < 0.015


# endregion
