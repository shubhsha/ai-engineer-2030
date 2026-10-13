"""Logical clocks for chapter 55: Lamport and vector clocks.

Both clocks are immutable values. Every event returns the next clock, so a
test (or a reader with a pencil) can keep the whole history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Order = Literal["before", "after", "concurrent", "equal"]


# region lamport
@dataclass(frozen=True, slots=True)
class LamportClock:
    """A scalar clock: one integer that never goes backwards."""

    time: int = 0

    def tick(self) -> LamportClock:
        """A local event."""
        return LamportClock(self.time + 1)

    def send(self) -> LamportClock:
        """Sending is a local event; the new time travels as the stamp."""
        return self.tick()

    def receive(self, stamp: int) -> LamportClock:
        """Jump past the sender's stamp, then count the receive itself."""
        return LamportClock(max(self.time, stamp) + 1)


# endregion lamport


# region vector
@dataclass(frozen=True, slots=True)
class VectorClock:
    """One counter per process; ``me`` is this process's slot."""

    me: int
    counts: tuple[int, ...]

    @classmethod
    def zero(cls, me: int, processes: int) -> VectorClock:
        return cls(me, (0,) * processes)

    def tick(self) -> VectorClock:
        counts = list(self.counts)
        counts[self.me] += 1
        return VectorClock(self.me, tuple(counts))

    def send(self) -> VectorClock:
        return self.tick()

    def receive(self, stamp: tuple[int, ...]) -> VectorClock:
        merged = tuple(
            max(mine, theirs)
            for mine, theirs in zip(self.counts, stamp, strict=True)
        )
        return VectorClock(self.me, merged).tick()


def compare(a: tuple[int, ...], b: tuple[int, ...]) -> Order:
    """Order two vector stamps: before, after, equal or concurrent."""
    if len(a) != len(b):
        raise ValueError("stamps come from different process sets")
    le = all(x <= y for x, y in zip(a, b, strict=True))
    ge = all(x >= y for x, y in zip(a, b, strict=True))
    if le and ge:
        return "equal"
    if le:
        return "before"
    if ge:
        return "after"
    return "concurrent"


# endregion vector
