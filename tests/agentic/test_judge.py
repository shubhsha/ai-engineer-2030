import json

from ai_engineer.evals.judge import compare, grade
from ai_engineer.llm.model import ScriptedModel

RUBRIC = "States the total; cites the invoice; under 50 words."
TASK = "What did the customer pay in March?"
VAGUE, CITED = "About a thousand.", "1,200 euros [invoice-3]."


def verdict(**fields):
    return json.dumps({"reasoning": "Checked each rubric point.", **fields})


def test_grade_puts_the_rubric_before_the_answer():
    model = ScriptedModel([verdict(score=4)])
    assert grade(model, RUBRIC, TASK, CITED).score == 4
    prompt = model.requests[0].messages[0]["content"]
    assert prompt.index("<rubric>") < prompt.index("<answer>")


def test_an_out_of_range_score_is_repaired():
    model = ScriptedModel([verdict(score=7), verdict(score=5)])
    assert grade(model, RUBRIC, TASK, CITED).score == 5
    assert "7 is not one of" in model.requests[1].messages[-1]["content"]


def fair_judge(request):
    prompt = request.messages[0]["content"]
    first = prompt.split("<answer_a>")[1].split("</answer_a>")[0]
    return verdict(winner="A" if "[invoice-3]" in first else "B")


def test_a_fair_judge_picks_the_same_answer_in_both_orders():
    out = compare(
        ScriptedModel(default=fair_judge), RUBRIC, TASK, VAGUE, CITED
    )
    assert (out.winner, out.consistent) == ("second", True)


def test_position_bias_is_scored_as_a_tie():
    model = ScriptedModel(default=verdict(winner="A"))
    out = compare(model, RUBRIC, TASK, VAGUE, CITED)
    assert (out.winner, out.consistent) == ("tie", False)
