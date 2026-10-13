"""A personal operating manual kept like an architecture decision record.

Entries are dated, carry a status, and are never edited: a changed mind is a
new entry that supersedes the old one. Dates come from the caller, so the
module never reads the clock.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

Status = Literal["proposed", "accepted", "superseded"]
Side = Literal["for", "against", "tie"]


# region manual_entry
@dataclass(frozen=True, slots=True)
class Entry:
    """One line of the manual: a value, a rule or a practice."""

    date: date
    title: str
    body: str
    status: Status = "proposed"
    supersedes: str | None = None


# endregion


# region manual_class
@dataclass(frozen=True, slots=True)
class Manual:
    """An append-only list of entries."""

    entries: tuple[Entry, ...] = ()

    def add(self, entry: Entry) -> Manual:
        titles = {e.title for e in self.entries}
        if entry.title in titles:
            raise ValueError(f"duplicate title: {entry.title!r}")
        if self.entries and entry.date < self.entries[-1].date:
            raise ValueError("entries must be added in date order")
        if entry.supersedes is not None and entry.supersedes not in titles:
            raise ValueError(f"unknown entry: {entry.supersedes!r}")
        return Manual((*self.entries, entry))

    def current(self) -> tuple[Entry, ...]:
        """Entries that still stand: not superseded, by status or by name."""
        replaced = {e.supersedes for e in self.entries if e.supersedes}
        return tuple(
            e
            for e in self.entries
            if e.status != "superseded" and e.title not in replaced
        )

    def stale(self, today: date, days: int = 90) -> tuple[Entry, ...]:
        """Current entries not revised for `days` days: time to re-read."""
        limit = today - timedelta(days=days)
        return tuple(e for e in self.current() if e.date < limit)


# endregion


# region manual_weigh
@dataclass(frozen=True, slots=True)
class Verdict:
    side: Side
    margin: int


def weigh(for_points: Sequence[int], against_points: Sequence[int]) -> Verdict:
    """Vote for the heavier side; the margin says how close it was."""
    if any(p < 0 for p in (*for_points, *against_points)):
        raise ValueError("points must not be negative")
    margin = sum(for_points) - sum(against_points)
    if margin == 0:
        return Verdict("tie", 0)
    return Verdict("for" if margin > 0 else "against", abs(margin))


# endregion
