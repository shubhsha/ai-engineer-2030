import pytest

from ai_engineer.agents.workflows import (
    GateFailed,
    chain,
    orchestrate,
    parallel,
    refine,
    route,
    vote,
)
from ai_engineer.llm.model import ScriptedModel


def first_message(model, i):
    return model.requests[i].messages[0]["content"]


def test_chain_feeds_each_output_to_the_next_step():
    model = ScriptedModel(["three bullet points", "THREE BULLET POINTS"])
    out = chain(model, ["Summarize.", "Uppercase it."], "a long report")
    assert out == "THREE BULLET POINTS"
    assert "<input>\nthree bullet points\n</input>" in first_message(model, 1)


def test_a_gate_stops_the_chain_between_steps():
    model = ScriptedModel(["draft with TODO", "unused"])
    with pytest.raises(GateFailed) as info:
        chain(
            model,
            ["Draft.", "Polish."],
            "notes",
            gate=lambda step, out: "TODO" not in out,
        )
    assert info.value.step == 0
    assert len(model.requests) == 1


HANDLERS = {
    "billing": lambda message: "billing team",
    "general": lambda message: "help desk",
}


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Billing", ("billing", "billing team")),
        ("legal", ("general", "help desk")),
    ],
)
def test_route_trusts_only_known_labels(label, expected):
    model = ScriptedModel([label])
    out = route(model, "I was charged twice", HANDLERS, fallback="general")
    assert out == expected


def test_parallel_keeps_the_input_order():
    model = ScriptedModel(default=lambda r: r.messages[0]["content"].upper())
    out = parallel(model, ["a", "b", "c", "d", "e"], limit=3)
    assert out == ["A", "B", "C", "D", "E"]


def test_vote_returns_the_majority_and_its_share():
    model = ScriptedModel(["42", "41", "42", "42", "40"])
    assert vote(model, "What is 6 x 7?") == ("42", 0.6)


def test_orchestrate_splits_the_work_and_merges_the_parts():
    planner = ScriptedModel(
        ["Revenue by region\nRefunds by product", "Merged answer"]
    )
    worker = ScriptedModel(default="a finding")
    assert orchestrate(planner, worker, "Explain the Q3 dip.") == (
        "Merged answer"
    )
    assert len(worker.requests) == 2
    assert first_message(planner, 1).count("<part>") == 2


def test_refine_stops_when_the_evaluator_passes():
    generator = ScriptedModel(["draft 1", "draft 2"])
    evaluator = ScriptedModel(["Add the date.", "PASS"])
    out = refine(generator, evaluator, "Write the release note.")
    assert out == ("draft 2", 2)
