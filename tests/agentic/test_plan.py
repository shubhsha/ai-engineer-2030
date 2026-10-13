import json

import pytest

from ai_engineer.agents.plan import (
    CycleError,
    Step,
    check_plan,
    critical_path,
    make_plan,
    run_plan,
    waves,
)
from ai_engineer.llm.model import ScriptedModel

ACTIONS = {
    "extract": "Copy the source tables to staging.",
    "load": "Load staging into the warehouse.",
    "profile": "Profile the staged data.",
    "reconcile": "Compare totals between source and warehouse.",
}
PIPELINE = [
    Step("extract", "extract"),
    Step("load", "load", ("extract",)),
    Step("profile", "profile", ("extract",)),
    Step("reconcile", "reconcile", ("load", "profile")),
]


def test_independent_steps_share_a_wave():
    assert waves(PIPELINE) == [
        ["extract"],
        ["load", "profile"],
        ["reconcile"],
    ]


def test_a_cycle_cannot_be_scheduled():
    with pytest.raises(CycleError, match="cycle"):
        waves([Step("a", "load", ("b",)), Step("b", "load", ("a",))])


def test_check_plan_reports_every_problem():
    plan = [
        Step("a", "extract"),
        Step("a", "load"),
        Step("b", "teleport", ("g",)),
    ]
    assert check_plan(plan, ACTIONS) == [
        "duplicate step id 'a'",
        "step 'b': unknown action 'teleport'",
        "step 'b' needs 'g', which is not a step",
    ]
    assert check_plan(PIPELINE, ACTIONS) == []


def test_make_plan_repairs_an_impossible_step():
    bad = {"steps": [{"id": "s1", "action": "teleport"}]}
    good = {
        "steps": [
            {"id": "s1", "action": "extract"},
            {"id": "s2", "action": "load", "needs": ["s1"]},
        ]
    }
    model = ScriptedModel([json.dumps(bad), json.dumps(good)])
    steps = make_plan(model, "Refresh the sales mart.", ACTIONS)
    assert waves(steps) == [["s1"], ["s2"]]
    repair = model.requests[1].messages[-1]["content"]
    assert "step 's1': unknown action 'teleport'" in repair


def test_critical_path_is_the_number_of_waves():
    assert critical_path(PIPELINE) == 3
    assert critical_path([Step("a", "extract"), Step("b", "load")]) == 1


def test_run_plan_replans_once_then_finishes():
    def run(step):
        return step.id != "profile"

    def replan(done, failed):
        assert failed == "profile"
        assert done == {"extract", "load"}
        return [
            Step("profile2", "profile", tuple(done)),
            Step("reconcile", "reconcile", ("load", "profile2")),
        ]

    result = run_plan(PIPELINE, run, replan)
    assert result.failed is None
    assert result.replans == 1
    assert result.done[-1] == "reconcile"


def test_run_plan_stops_when_the_budget_is_spent():
    def replan(done, failed):
        return [Step(failed, "load")]

    result = run_plan(
        [Step("a", "load")], lambda step: False, replan, max_replans=2
    )
    assert result.failed == "a"
    assert result.replans == 2
