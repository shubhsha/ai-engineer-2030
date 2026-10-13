"""Chapter 38: a model as grader, with known biases designed out.

Judges favor the first answer they see, longer answers and their own
style. Score against a written rubric, reason before scoring, and for
pairwise comparisons ask twice with the order swapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_engineer.llm.model import Model, Request, Usage, user
from ai_engineer.llm.prompt import tagged
from ai_engineer.llm.structured import generate_validated

_REASONING = {"type": "string"}
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": _REASONING,
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["reasoning", "score"],
    "additionalProperties": False,
}
PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": _REASONING,
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
    },
    "required": ["reasoning", "winner"],
    "additionalProperties": False,
}
JUDGE = (
    "You grade answers against a rubric. Quote the evidence for each "
    "point of the rubric, then decide. Length and confidence are not "
    "quality. Reply with JSON only."
)


@dataclass(frozen=True, slots=True)
class Grade:
    score: int
    reasoning: str
    usage: Usage


def grade(model: Model, rubric: str, task: str, answer: str) -> Grade:
    """Score one answer from 1 to 5 against the rubric."""
    prompt = "\n\n".join(
        [
            tagged("rubric", rubric),
            tagged("task", task),
            tagged("answer", answer),
        ]
    )
    out = generate_validated(
        model, Request(JUDGE, (user(prompt),)), GRADE_SCHEMA
    )
    return Grade(out.value["score"], out.value["reasoning"], out.usage)


Winner = Literal["first", "second", "tie"]


@dataclass(frozen=True, slots=True)
class Comparison:
    winner: Winner
    consistent: bool  # both orders agreed
    usage: Usage


def _pick(
    model: Model, rubric: str, task: str, a: str, b: str
) -> tuple[str, Usage]:
    prompt = "\n\n".join(
        [
            tagged("rubric", rubric),
            tagged("task", task),
            tagged("answer_a", a),
            tagged("answer_b", b),
        ]
    )
    out = generate_validated(
        model, Request(JUDGE, (user(prompt),)), PAIR_SCHEMA
    )
    return out.value["winner"], out.usage


def compare(
    model: Model, rubric: str, task: str, first: str, second: str
) -> Comparison:
    """Judge two answers in both orders. A win counts only when both
    orders pick the same answer; disagreement is position bias, which
    is scored as a tie."""
    one, used_one = _pick(model, rubric, task, first, second)
    two, used_two = _pick(model, rubric, task, second, first)
    named: dict[str, Winner] = {"A": "first", "B": "second", "tie": "tie"}
    swapped: dict[str, Winner] = {"A": "second", "B": "first", "tie": "tie"}
    agree = named[one] == swapped[two]
    winner = named[one] if agree else "tie"
    return Comparison(winner, agree, used_one + used_two)
