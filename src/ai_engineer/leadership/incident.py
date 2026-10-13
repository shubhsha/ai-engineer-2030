"""Incident leadership: states, roles, update cadence and action items.

Times are whole minutes from any fixed origin and are passed in, so
nothing here reads a clock. An incident moves one way through four
states; the commander never holds the operations role; every severity
has an update cadence; and a review closes only when no action item is
open.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum


class Severity(Enum):
    SEV1 = 1
    SEV2 = 2
    SEV3 = 3


# region comms_cadence
CADENCE_MINUTES = {Severity.SEV1: 15, Severity.SEV2: 30, Severity.SEV3: 60}


def missed_updates(
    updates: list[int], severity: Severity, declared_at: int, closed_at: int
) -> int:
    """Count update deadlines that passed with no update sent.

    The first deadline is one cadence after declaration; each update
    restarts the clock. A deadline that falls exactly at ``closed_at``
    is not missed.
    """
    step = CADENCE_MINUTES[severity]
    last, missed = declared_at, 0
    for sent in sorted(u for u in updates if u <= closed_at):
        while last + step < sent:
            missed += 1
            last += step
        last = sent
    while last + step < closed_at:
        missed += 1
        last += step
    return missed


# endregion comms_cadence


class State(Enum):
    DECLARED = 1
    MITIGATED = 2
    RESOLVED = 3
    REVIEWED = 4


class IllegalTransition(ValueError):
    """Raised when an incident is moved in a way the process forbids."""


# region action_items
@dataclass(frozen=True, slots=True)
class ActionItem:
    title: str
    owner: str
    due: int
    done: bool = False


def open_items(items: tuple[ActionItem, ...]) -> list[ActionItem]:
    """Items not yet done, earliest due first."""
    return sorted((i for i in items if not i.done), key=lambda i: i.due)


def overdue(items: tuple[ActionItem, ...], now: int) -> list[ActionItem]:
    return [i for i in open_items(items) if i.due < now]


# endregion action_items


# region incident_states
@dataclass(frozen=True, slots=True)
class Incident:
    name: str
    severity: Severity
    declared_at: int
    state: State = State.DECLARED
    history: tuple[tuple[State, int], ...] = ()
    items: tuple[ActionItem, ...] = ()

    def transition(self, to: State, at: int) -> Incident:
        """Move one step forward, or refuse."""
        if to.value != self.state.value + 1:
            raise IllegalTransition(f"{self.state.name} -> {to.name}")
        last = self.history[-1][1] if self.history else self.declared_at
        if at < last:
            raise IllegalTransition("time went backwards")
        if to is State.REVIEWED and open_items(self.items):
            raise IllegalTransition("review has open action items")
        return replace(
            self,
            state=to,
            history=(*self.history, (to, at)),
        )

    def minutes_to(self, state: State) -> int | None:
        for reached, at in self.history:
            if reached is state:
                return at - self.declared_at
        return None


# endregion incident_states


class Role(Enum):
    COMMANDER = "commander"
    OPERATIONS = "operations"
    COMMS = "comms"
    SCRIBE = "scribe"


# region incident_roles
def assign_roles(people: dict[Role, str]) -> dict[Role, str]:
    """Check an assignment of people to roles.

    The commander must exist and must not also be the operations lead.
    """
    if Role.COMMANDER not in people:
        raise ValueError("an incident needs a commander")
    commander = people[Role.COMMANDER]
    if people.get(Role.OPERATIONS) == commander:
        raise ValueError("the commander does not hold operations")
    return dict(people)


# endregion incident_roles


@dataclass(frozen=True, slots=True)
class Event:
    at: int
    kind: str
    text: str


def timeline(events: list[Event], declared_at: int) -> list[str]:
    """Lines like ``+12 mitigate  rolled back``, in time order."""
    rows = sorted(events, key=lambda e: e.at)
    return [f"{e.at - declared_at:+d} {e.kind:<9} {e.text}" for e in rows]


@dataclass(frozen=True, slots=True)
class SimResult:
    with_ic: float
    without_ic: float
    missed_with: int
    missed_without: int
    by_severity: dict[str, tuple[int, int]]


# region incident_sim
def simulate(seed: int = 2030, n: int = 40) -> SimResult:
    """Simulate n incidents with and without a separate commander.

    Assumptions (they are the whole result): the same hands-on work
    takes the same minutes in both arms; a lead who also operates is
    heads-down, so mitigation takes 35% longer and each due update is
    sent with probability 0.55, against 0.95 with a commander.
    """
    rng = random.Random(seed)
    arms: dict[bool, tuple[list[int], int]] = {
        True: ([], 0),
        False: ([], 0),
    }
    by_sev = {s.name: [0, 0] for s in Severity}
    for _ in range(n):
        sev = rng.choice(list(Severity))
        work = rng.randint(15, 90)
        for sep in (True, False):
            take = work if sep else round(work * 1.35)
            p = 0.95 if sep else 0.55
            step = CADENCE_MINUTES[sev]
            sent = [t for t in range(step, take, step) if rng.random() < p]
            miss = missed_updates(sent, sev, 0, take)
            arms[sep][0].append(take)
            arms[sep] = (arms[sep][0], arms[sep][1] + miss)
            by_sev[sev.name][0 if sep else 1] += miss
    mean = {k: sum(v[0]) / n for k, v in arms.items()}
    return SimResult(
        mean[True],
        mean[False],
        arms[True][1],
        arms[False][1],
        {k: (v[0], v[1]) for k, v in by_sev.items()},
    )


# endregion incident_sim
