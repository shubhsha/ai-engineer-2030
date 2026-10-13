"""Technical strategy as a checkable document.

A strategy has a kernel of three parts: a diagnosis of the challenge,
guiding policies that answer it, and coherent actions that carry the
policies out. Here each part has an id so that the links between them
can be checked by a program: every action traces to a policy, every
policy to the diagnosis, and every action has an owner and a date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


# region strategy_parts
@dataclass(frozen=True, slots=True)
class Diagnosis:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class Policy:
    id: str
    text: str
    answers: tuple[str, ...]  # diagnosis ids


@dataclass(frozen=True, slots=True)
class Action:
    id: str
    text: str
    owner: str
    due: date | None
    traces_to: tuple[str, ...]  # policy ids


# endregion strategy_parts


# region strategy_check
@dataclass(frozen=True, slots=True)
class Findings:
    untraced_actions: tuple[str, ...]
    orphan_policies: tuple[str, ...]
    ungrounded_policies: tuple[str, ...]
    unanswered_diagnoses: tuple[str, ...]
    missing_owners: tuple[str, ...]
    missing_dates: tuple[str, ...]

    @property
    def count(self) -> int:
        return (
            len(self.untraced_actions)
            + len(self.orphan_policies)
            + len(self.ungrounded_policies)
            + len(self.unanswered_diagnoses)
            + len(self.missing_owners)
            + len(self.missing_dates)
        )

    @property
    def ok(self) -> bool:
        return self.count == 0


@dataclass(frozen=True, slots=True)
class Strategy:
    diagnoses: tuple[Diagnosis, ...]
    policies: tuple[Policy, ...]
    actions: tuple[Action, ...]

    def check(self) -> Findings:
        d_ids = {d.id for d in self.diagnoses}
        p_ids = {p.id for p in self.policies}
        used = {t for a in self.actions for t in a.traces_to if t in p_ids}
        answered = {t for p in self.policies for t in p.answers if t in d_ids}
        return Findings(
            untraced_actions=tuple(
                a.id
                for a in self.actions
                if not any(t in p_ids for t in a.traces_to)
            ),
            orphan_policies=tuple(
                p.id for p in self.policies if p.id not in used
            ),
            ungrounded_policies=tuple(
                p.id
                for p in self.policies
                if not any(t in d_ids for t in p.answers)
            ),
            unanswered_diagnoses=tuple(
                d.id for d in self.diagnoses if d.id not in answered
            ),
            missing_owners=tuple(
                a.id for a in self.actions if not a.owner.strip()
            ),
            missing_dates=tuple(a.id for a in self.actions if a.due is None),
        )


# endregion strategy_check


# region stage_advice
@dataclass(frozen=True, slots=True)
class Advice:
    decision: str  # "build", "buy" or "rent"
    reason: str


_STAGES = {
    "genesis": Advice(
        "build",
        "No product exists and the idea is unproven: a small team "
        "explores, and the work is the learning.",
    ),
    "custom": Advice(
        "build",
        "Products do not fit yet, and this is where you differ from "
        "others: own it, but keep the interface narrow.",
    ),
    "product": Advice(
        "buy",
        "Several vendors compete on features: buying is faster than "
        "building, and switching stays possible.",
    ),
    "commodity": Advice(
        "rent",
        "Supply is standard and priced by use: pay per call, and do "
        "not run it yourself.",
    ),
}


def stage_advice(stage: str) -> Advice:
    """Map a Wardley evolution stage to build, buy or rent, with why."""
    try:
        return _STAGES[stage.strip().lower()]
    except KeyError:
        known = ", ".join(_STAGES)
        raise ValueError(
            f"unknown stage {stage!r}; use one of {known}"
        ) from None


# endregion stage_advice
