"""A seeded replica simulation for chapter 55: quorums and stale reads.

Time is in milliseconds and is modeled, never slept. A write goes to every
replica and each copy lands after its own random delay. The write returns
when ``w`` replicas have it. A read asks every replica and takes the
newest answer among the ``r`` fastest.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WriteResult:
    version: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class ReadResult:
    value: str | None
    version: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class QuorumStats:
    n: int
    r: int
    w: int
    stale_rate: float
    mean_read_ms: float
    mean_write_ms: float


# region cluster
class Cluster:
    """N replicas of a key-value store with delayed propagation."""

    def __init__(self, n: int, rng: random.Random) -> None:
        if n < 1:
            raise ValueError("need at least one replica")
        self.n = n
        self._rng = rng
        self.now = 0.0
        self._version = 0
        # per replica: key -> [(version, value, time the copy lands)]
        self._log: list[dict[str, list[tuple[int, str, float]]]] = [
            {} for _ in range(n)
        ]

    def _check(self, count: int) -> None:
        if not 1 <= count <= self.n:
            raise ValueError("quorum size must be between 1 and n")

    def write(self, key: str, value: str, w: int) -> WriteResult:
        """Send to all replicas; return when the w-th copy has landed."""
        self._check(w)
        self._version += 1
        delays = [2.0 + self._rng.expovariate(1 / 8) for _ in range(self.n)]
        for replica, delay in zip(self._log, delays, strict=True):
            copy = (self._version, value, self.now + delay)
            replica.setdefault(key, []).append(copy)
        latency = sorted(delays)[w - 1]
        self.now += latency
        return WriteResult(self._version, latency)

    def read(self, key: str, r: int) -> ReadResult:
        """Ask all replicas; the r fastest answers decide."""
        self._check(r)
        delays = [1.0 + self._rng.expovariate(1 / 3) for _ in range(self.n)]
        fastest = sorted(range(self.n), key=delays.__getitem__)[:r]
        best: tuple[int, str | None] = (0, None)
        for i in fastest:
            seen_at = self.now + delays[i]
            for version, value, lands in self._log[i].get(key, []):
                if lands <= seen_at and version > best[0]:
                    best = (version, value)
        return ReadResult(best[1], best[0], max(delays[i] for i in fastest))


# endregion cluster


# region stale
def stale_read_rate(
    n: int, r: int, w: int, rng: random.Random, trials: int
) -> QuorumStats:
    """Write, then read at once, many times; count the stale answers."""
    cluster = Cluster(n, rng)
    stale = 0
    read_ms = write_ms = 0.0
    for i in range(trials):
        written = cluster.write(f"k{i}", f"v{i}", w)
        got = cluster.read(f"k{i}", r)
        stale += got.version != written.version
        read_ms += got.latency_ms
        write_ms += written.latency_ms
    return QuorumStats(
        n, r, w, stale / trials, read_ms / trials, write_ms / trials
    )


# endregion stale
