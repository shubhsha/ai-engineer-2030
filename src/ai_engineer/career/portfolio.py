"""A portfolio checker, a readiness report and a STAR story builder.

An entry is only evidence when it names a problem, a role, something
checkable and a lesson. The checks are deliberately blunt: they find the
empty fields, not the weak ideas.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_NUMBER = re.compile(r"\d")
_LINK = re.compile(r"https?://")
SPOKEN_WORDS_PER_MINUTE = 150


def _checkable(item: str) -> bool:
    """True when an evidence item has a number or a link in it."""
    return bool(_NUMBER.search(item) or _LINK.search(item))


# region portfolio_entry
@dataclass(frozen=True, slots=True)
class Entry:
    """One project, written so a stranger can judge it."""

    title: str
    problem: str
    role: str
    evidence: tuple[str, ...]
    lesson: str
    skills: tuple[str, ...] = ()


def check_entry(entry: Entry) -> tuple[str, ...]:
    """Return what is wrong with an entry; empty means it passes."""
    found: list[str] = []
    for name in ("problem", "role", "lesson"):
        if not getattr(entry, name).strip():
            found.append(f"{name} is empty")
    if not entry.evidence:
        found.append("no evidence")
    elif not any(_checkable(item) for item in entry.evidence):
        found.append("evidence has no number and no link")
    return tuple(found)


# endregion


# region portfolio_readiness
@dataclass(frozen=True, slots=True)
class Role:
    """A job, reduced to the skills it asks for."""

    name: str
    requirements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Readiness:
    """Which requirements have evidence behind them, and which do not."""

    met: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def share(self) -> float:
        total = len(self.met) + len(self.missing)
        return len(self.met) / total if total else 1.0


def readiness(entries: Iterable[Entry], role: Role) -> Readiness:
    """Match each requirement to a passing entry that claims the skill."""
    shown = {
        skill.casefold()
        for entry in entries
        if not check_entry(entry)
        for skill in entry.skills
    }
    met = tuple(r for r in role.requirements if r.casefold() in shown)
    missing = tuple(r for r in role.requirements if r.casefold() not in shown)
    return Readiness(met, missing)


# endregion


# region portfolio_star
@dataclass(frozen=True, slots=True)
class Story:
    """A behavioral answer in four parts."""

    situation: str
    task: str
    action: str
    result: str

    def text(self) -> str:
        return " ".join((self.situation, self.task, self.action, self.result))

    @property
    def seconds(self) -> int:
        """Spoken length at 150 words a minute, rounded to a second."""
        words = len(self.text().split())
        return round(words * 60 / SPOKEN_WORDS_PER_MINUTE)


def star(situation: str, task: str, action: str, result: str) -> Story:
    """Build a story; refuse one whose result cannot be measured."""
    parts = {"situation": situation, "task": task, "action": action}
    for name, value in parts.items():
        if not value.strip():
            raise ValueError(f"{name} is empty")
    if not _NUMBER.search(result):
        raise ValueError("result needs a number")
    return Story(situation, task, action, result)


# endregion
