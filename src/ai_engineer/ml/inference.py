"""Chapter 24: the arithmetic of serving a language model.

Back-of-the-envelope numbers settle most inference designs before any
benchmark runs: how much memory the weights and the KV cache need, how
long the first token takes, how fast tokens stream and what batching
does to both. These functions make that arithmetic explicit and tested.
All sizes are in bytes and all times in seconds unless a name says so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ai_engineer.ml.numerics import Array

Z95 = 1.6449  # standard normal quantile for the 95th percentile


def weight_bytes(params: float, bits: float) -> float:
    return params * bits / 8


def kv_cache_bytes(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    tokens: int,
    batch: int = 1,
    bytes_per_value: float = 2,
) -> float:
    """Memory for cached keys and values: two tensors per layer.

    Grouped-query attention shares each key/value head among several
    query heads, which is why ``kv_heads`` is often far below the
    number of attention heads.
    """
    per_token = 2 * layers * kv_heads * head_dim * bytes_per_value
    return per_token * tokens * batch


@dataclass(frozen=True)
class Accelerator:
    name: str
    memory: float  # bytes
    bandwidth: float  # bytes per second
    flops: float  # dense operations per second at serving precision


def decode_step_seconds(
    acc: Accelerator,
    *,
    params: float,
    weights: float,
    kv_per_sequence: float,
    batch: int,
) -> float:
    """Roofline estimate of one decoding step for ``batch`` sequences.

    Memory: each step streams every weight once, shared by the whole
    batch, plus each sequence's own KV cache. Compute: about two
    operations per parameter per token. The slower of the two wins.
    """
    memory = (weights + batch * kv_per_sequence) / acc.bandwidth
    compute = 2 * params * batch / acc.flops
    return max(memory, compute)


@dataclass(frozen=True)
class BatchPoint:
    batch: int
    per_user_tps: float  # tokens per second each user sees
    total_tps: float  # tokens per second the server produces


def batch_tradeoff(
    acc: Accelerator,
    *,
    params: float,
    weights: float,
    kv_per_sequence: float,
    batches: Sequence[int],
) -> list[BatchPoint]:
    """Throughput rises with batch size; per-user speed eventually falls."""
    points = []
    for b in batches:
        step = decode_step_seconds(
            acc,
            params=params,
            weights=weights,
            kv_per_sequence=kv_per_sequence,
            batch=b,
        )
        points.append(BatchPoint(b, 1 / step, b / step))
    return points


@dataclass(frozen=True)
class Hop:
    """One step on a request path, with its median and p95 latency."""

    name: str
    p50_ms: float
    p95_ms: float


def simulate_path(
    hops: Sequence[Hop], *, n: int = 100_000, seed: int = 0
) -> Array:
    """End-to-end latency samples for a request that visits every hop.

    Each hop is modeled as lognormal, matched to its p50 and p95. The
    point of the simulation is the tail: the p95 of the sum is less
    than the sum of the p95s, but more than any single hop's p95.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(n)
    for hop in hops:
        sigma = math.log(hop.p95_ms / hop.p50_ms) / Z95
        total += rng.lognormal(math.log(hop.p50_ms), sigma, n)
    return total


def percentile(samples: Array, q: float) -> float:
    return float(np.percentile(samples, q))


def littles_law_concurrency(arrivals_per_s: float, latency_s: float) -> float:
    """Requests in flight = arrival rate x time in the system."""
    return arrivals_per_s * latency_s


def cost_per_million_tokens(
    hourly_price: float, tokens_per_s: float, utilization: float = 1.0
) -> float:
    """Self-hosting cost: what an hour of hardware buys, per million tokens.

    ``utilization`` is the share of the hour spent serving real traffic;
    idle capacity is paid for all the same.
    """
    tokens_per_hour = tokens_per_s * 3600 * utilization
    return hourly_price / tokens_per_hour * 1e6


# region request_latency
def request_latency(ttft: float, tpot: float, output_tokens: int) -> float:
    """End-to-end time: first token, then one token per ``tpot``."""
    return ttft + max(output_tokens - 1, 0) * tpot


# endregion


# region speculative_tokens_per_step
def speculative_tokens_per_step(alpha: float, gamma: int) -> float:
    """Expected tokens per target-model pass with a draft of ``gamma``.

    Each drafted token is accepted with probability ``alpha``; the pass
    always yields one more token, so the sum runs to ``gamma + 1``.
    """
    if alpha >= 1:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


# endregion


# region mm1_queue
def mm1_wait_seconds(arrival_rate: float, service_time: float) -> float:
    """Mean time in an M/M/1 system (queue plus service).

    With utilization ``rho`` = arrival rate x service time, the mean
    time in system is ``service_time / (1 - rho)``. At ``rho`` >= 1 the
    queue grows without bound.
    """
    rho = arrival_rate * service_time
    if rho >= 1:
        return math.inf
    return service_time / (1 - rho)


# endregion
