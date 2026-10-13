"""Capacity arithmetic for systems with a model inside (chapter 54).

Estimates, Little's law, a single queue, a latency budget per hop, and a
seeded fan-out simulation with hedged requests.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

SECONDS_PER_DAY = 86_400
MTOK = Decimal(1_000_000)


# region estimate
@dataclass(frozen=True, slots=True)
class Estimate:
    """A back-of-the-envelope load estimate."""

    daily_users: int
    requests_per_user: float
    peak_factor: float
    tokens_in: int
    tokens_out: int

    @property
    def daily_requests(self) -> float:
        return self.daily_users * self.requests_per_user

    @property
    def average_rps(self) -> float:
        return self.daily_requests / SECONDS_PER_DAY

    @property
    def peak_rps(self) -> float:
        return self.average_rps * self.peak_factor

    def peak_tokens_per_second(self) -> tuple[float, float]:
        """(input, output) tokens per second at the peak."""
        return (
            self.peak_rps * self.tokens_in,
            self.peak_rps * self.tokens_out,
        )

    def monthly_cost(
        self, input_price: Decimal, output_price: Decimal, days: int = 30
    ) -> Decimal:
        """Dollars per month; prices are dollars per million tokens."""
        calls = Decimal(str(self.daily_requests)) * days
        cost_in = calls * self.tokens_in * input_price / MTOK
        cost_out = calls * self.tokens_out * output_price / MTOK
        return cost_in + cost_out


# endregion estimate


# region littles-law
def littles_law(throughput: float, latency: float) -> float:
    """Average number in the system: L = lambda * W."""
    if throughput < 0 or latency < 0:
        raise ValueError("throughput and latency must not be negative")
    return throughput * latency


def mm1_wait(utilization: float, service_time: float) -> float:
    """Mean time spent waiting in an M/M/1 queue, before service starts."""
    if not 0 <= utilization < 1:
        raise ValueError("utilization must be in [0, 1)")
    return utilization / (1 - utilization) * service_time


# endregion littles-law


# region latency-budget
@dataclass(frozen=True, slots=True)
class LatencyBudget:
    """Named hops, each with an allowance in seconds."""

    hops: tuple[tuple[str, float], ...]

    @property
    def total(self) -> float:
        return sum(allowance for _, allowance in self.hops)

    def check(self, measured: Mapping[str, float]) -> tuple[str, ...]:
        """Names of the hops whose measured time exceeds the allowance."""
        return tuple(
            name
            for name, allowance in self.hops
            if measured.get(name, 0.0) > allowance
        )


# endregion latency-budget


# region fanout
def fanout_slow_share(p_slow: float, n: int) -> float:
    """Chance that at least one of n calls is slow: 1 - (1 - p)^n."""
    return 1 - (1 - p_slow) ** n


@dataclass(frozen=True, slots=True)
class FanoutResult:
    p50: float
    p99: float
    extra_load: float


def _percentile(sorted_values: list[float], q: float) -> float:
    index = min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[index]


def simulate_fanout(
    rng: random.Random,
    n: int,
    hedge_after: float | None = None,
    trials: int = 4000,
    sigma: float = 0.6,
    median: float = 0.1,
) -> FanoutResult:
    """Latency of a request that waits for n lognormal calls.

    With ``hedge_after`` set, a call still unanswered at that time is
    sent a second time and the faster answer wins. ``extra_load`` is the
    share of additional calls that the hedges cost.
    """
    mu = math.log(median)
    totals: list[float] = []
    extra = 0
    for _ in range(trials):
        slowest = 0.0
        for _ in range(n):
            t = rng.lognormvariate(mu, sigma)
            if hedge_after is not None and t > hedge_after:
                extra += 1
                t = min(t, hedge_after + rng.lognormvariate(mu, sigma))
            slowest = max(slowest, t)
        totals.append(slowest)
    totals.sort()
    return FanoutResult(
        p50=_percentile(totals, 0.50),
        p99=_percentile(totals, 0.99),
        extra_load=extra / (trials * n),
    )


# endregion fanout
