from __future__ import annotations

from datetime import date

import pytest

from ai_engineer.leadership.strategy import (
    Action,
    Diagnosis,
    Policy,
    Strategy,
    stage_advice,
)

D = date


# region composite_strategies
def draft_strategy() -> Strategy:
    """The first draft: goals with a few links missing."""
    return Strategy(
        diagnoses=(Diagnosis("D1", "We are behind on AI."),),
        policies=(
            Policy("P1", "Be AI-first.", ("D1",)),
            Policy("P2", "Modernize the data platform.", ()),
            Policy("P3", "Improve quality.", ("D1",)),
        ),
        actions=(
            Action("A1", "Launch copilots", "", None, ()),
            Action("A2", "Migrate the warehouse", "Data", None, ("P2",)),
            Action("A3", "Hire an AI team", "People", D(2027, 3, 1), ()),
        ),
    )


def revised_strategy() -> Strategy:
    """The revision: one diagnosis, one policy, three traced actions."""
    return Strategy(
        diagnoses=(
            Diagnosis(
                "D1",
                "Nine teams run nine prompt stacks with no shared "
                "evaluation, so none can tell a better model from a "
                "worse one.",
            ),
        ),
        policies=(
            Policy(
                "P1",
                "Rent models, own evaluation: one gateway and one "
                "evaluation service for every team.",
                ("D1",),
            ),
        ),
        actions=(
            Action(
                "A1",
                "Route all calls via the gateway",
                "Platform",
                D(2027, 1, 31),
                ("P1",),
            ),
            Action(
                "A2",
                "Publish a shared eval set per product",
                "Quality",
                D(2027, 2, 28),
                ("P1",),
            ),
            Action(
                "A3",
                "Retire the eight private prompt stacks",
                "Leads",
                D(2027, 6, 30),
                ("P1",),
            ),
        ),
    )


# endregion composite_strategies


def test_draft_findings_are_counted() -> None:
    f = draft_strategy().check()
    assert f.untraced_actions == ("A1", "A3")
    assert f.orphan_policies == ("P1", "P3")
    assert f.ungrounded_policies == ("P2",)
    assert f.unanswered_diagnoses == ()
    assert f.missing_owners == ("A1",)
    assert f.missing_dates == ("A1", "A2")
    assert f.count == 8
    assert not f.ok


def test_revised_strategy_is_clean() -> None:
    f = revised_strategy().check()
    assert f.ok
    assert f.count == 0


def test_unknown_policy_id_does_not_count_as_a_trace() -> None:
    s = Strategy(
        (Diagnosis("D1", "x"),),
        (Policy("P1", "y", ("D1",)),),
        (Action("A1", "z", "Ana", D(2027, 1, 1), ("P9",)),),
    )
    f = s.check()
    assert f.untraced_actions == ("A1",)
    assert f.orphan_policies == ("P1",)


def test_diagnosis_with_no_policy_is_reported() -> None:
    s = Strategy(
        (Diagnosis("D1", "x"), Diagnosis("D2", "y")),
        (Policy("P1", "p", ("D1",)),),
        (Action("A1", "a", "Ana", D(2027, 1, 1), ("P1",)),),
    )
    assert s.check().unanswered_diagnoses == ("D2",)


def test_blank_owner_counts_as_missing() -> None:
    s = Strategy(
        (Diagnosis("D1", "x"),),
        (Policy("P1", "p", ("D1",)),),
        (Action("A1", "a", "   ", D(2027, 1, 1), ("P1",)),),
    )
    assert s.check().missing_owners == ("A1",)


@pytest.mark.parametrize(
    ("stage", "decision"),
    [
        ("genesis", "build"),
        ("custom", "build"),
        ("product", "buy"),
        ("commodity", "rent"),
        (" Commodity ", "rent"),
    ],
)
def test_stage_advice(stage: str, decision: str) -> None:
    advice = stage_advice(stage)
    assert advice.decision == decision
    assert advice.reason


def test_unknown_stage_names_the_choices() -> None:
    with pytest.raises(ValueError, match="genesis"):
        stage_advice("legacy")
