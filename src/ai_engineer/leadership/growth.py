"""Growing engineers on purpose: a skills matrix and a 90-day plan.

Skills are rated on the Dreyfus scale, novice to expert. A person has a
current and a target level for each skill. The gap is the distance, and
the plan closes at most two gaps per quarter, because a third would be
a wish. Dates come from the caller; nothing reads the real clock.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import IntEnum


# region matrix
class Level(IntEnum):
    """The five stages of the Dreyfus model of skill acquisition."""

    NOVICE = 1
    ADVANCED_BEGINNER = 2
    COMPETENT = 3
    PROFICIENT = 4
    EXPERT = 5


@dataclass(frozen=True, slots=True)
class SkillRow:
    """One skill for one person: where they are and where they aim."""

    skill: str
    current: Level
    target: Level

    @property
    def gap(self) -> int:
        """Levels still to climb; zero when the target is met."""
        return max(0, self.target - self.current)


@dataclass(frozen=True, slots=True)
class Gap:
    """A skill gap with the weight the team gives to the skill."""

    skill: str
    size: int
    need: int

    @property
    def priority(self) -> int:
        return self.size * self.need


def gaps(
    matrix: Sequence[SkillRow], needs: Mapping[str, int] | None = None
) -> list[Gap]:
    """Open gaps, biggest priority first; ties break on skill name."""
    need = needs or {}
    found = [
        Gap(row.skill, row.gap, need.get(row.skill, 1))
        for row in matrix
        if row.gap > 0
    ]
    return sorted(found, key=lambda g: (-g.priority, g.skill))


def team_gaps(
    matrices: Mapping[str, Sequence[SkillRow]],
    needs: Mapping[str, int],
) -> list[Gap]:
    """Sum each skill's gap over the team, ranked by weighted size."""
    totals: dict[str, int] = {}
    for matrix in matrices.values():
        for row in matrix:
            totals[row.skill] = totals.get(row.skill, 0) + row.gap
    rows = [Gap(s, n, needs.get(s, 1)) for s, n in totals.items() if n]
    return sorted(rows, key=lambda g: (-g.priority, g.skill))


# endregion


# region plan
_STRETCH = {
    Level.ADVANCED_BEGINNER: "Do a small {skill} task with a checklist, "
    "then explain each step to your mentor.",
    Level.COMPETENT: "Own one {skill} task end to end, "
    "including the plan and the follow-up.",
    Level.PROFICIENT: "Take a {skill} problem with no clear recipe "
    "and decide the approach yourself.",
    Level.EXPERT: "Lead a {skill} decision others will follow, "
    "and write down why.",
}


@dataclass(frozen=True, slots=True)
class PlanItem:
    """One gap, one stretch assignment, one weekly practice, one date."""

    gap: str
    stretch: str
    practice: str
    check_in: date


def plan_90_days(
    person: str,
    matrix: Sequence[SkillRow],
    needs: Mapping[str, int],
    start: date,
) -> tuple[PlanItem, ...]:
    """At most two items: the top two gaps, checked at day 45 and 90.

    The stretch sits one level above the person's current level, just
    out of reach, never at the target if the target is further away.
    """
    by_skill = {row.skill: row for row in matrix}
    items: list[PlanItem] = []
    for index, gap in enumerate(gaps(matrix, needs)[:2]):
        row = by_skill[gap.skill]
        step = Level(row.current + 1)
        items.append(
            PlanItem(
                gap=gap.skill,
                stretch=_STRETCH[step].format(skill=gap.skill),
                practice=f"{person}: one hour a week of {gap.skill}, "
                "done by hand before any agent, reviewed on Fridays.",
                check_in=start + timedelta(days=45 * (index + 1)),
            )
        )
    return tuple(items)


# endregion


# region simulate
def follow_plan(
    matrix: Sequence[SkillRow],
    plan: Sequence[PlanItem],
    rng: random.Random,
    p_climb: float = 0.7,
) -> list[SkillRow]:
    """One quarter later: each planned skill climbs a level with p_climb."""
    planned = {item.gap for item in plan}
    out: list[SkillRow] = []
    for row in matrix:
        if row.skill in planned and row.gap and rng.random() < p_climb:
            row = replace(row, current=Level(row.current + 1))
        out.append(row)
    return out


# endregion
