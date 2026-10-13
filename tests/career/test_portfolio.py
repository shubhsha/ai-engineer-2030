"""Tests for the portfolio checker, readiness report and STAR builder."""

from __future__ import annotations

import pytest

from ai_engineer.career.portfolio import (
    Entry,
    Role,
    check_entry,
    readiness,
    star,
)


def _entry(
    problem: str = "Finance sorted 400 invoices a week by hand.",
    evidence: tuple[str, ...] = ("92 of 100 held-out invoices routed",),
    lesson: str = "Label the hard cases first.",
) -> Entry:
    return Entry(
        title="Invoice triage",
        problem=problem,
        role="Built the classifier and its eval set.",
        evidence=evidence,
        lesson=lesson,
        skills=("evals", "python"),
    )


def test_complete_entry_passes() -> None:
    assert check_entry(_entry()) == ()


def test_empty_fields_are_named() -> None:
    found = check_entry(_entry(problem=" ", lesson=""))
    assert found == ("problem is empty", "lesson is empty")


def test_evidence_needs_a_number_or_a_link() -> None:
    assert check_entry(_entry(evidence=("it works well",))) == (
        "evidence has no number and no link",
    )
    assert check_entry(_entry(evidence=("https://example.org/pr/1",))) == ()
    assert check_entry(_entry(evidence=())) == ("no evidence",)


def test_readiness_lists_met_and_missing() -> None:
    role = Role("AI engineer", ("Evals", "RAG", "Python"))
    report = readiness([_entry()], role)
    assert report.met == ("Evals", "Python")
    assert report.missing == ("RAG",)
    assert report.share == pytest.approx(2 / 3)


def test_failing_entry_earns_no_credit() -> None:
    role = Role("AI engineer", ("Evals",))
    weak = _entry(evidence=("it works well",))
    assert readiness([weak], role).missing == ("Evals",)


def test_empty_role_is_ready() -> None:
    assert readiness([], Role("none", ())).share == 1.0


def test_star_refuses_a_result_without_a_number() -> None:
    with pytest.raises(ValueError, match="number"):
        star("s", "t", "a", "It went much better.")
    with pytest.raises(ValueError, match="action"):
        star("s", "t", " ", "Cut errors by 30%.")


def test_star_length_in_seconds() -> None:
    story = star("one two", "three four", "five six", "seven 8")
    assert story.seconds == 3
    assert "seven 8" in story.text()
