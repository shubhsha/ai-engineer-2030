import json

import pytest

from ai_engineer.enterprise.sre_agent import (
    Alert,
    Deploy,
    Span,
    Telemetry,
    approve,
    build_tools,
    hypothesize,
    postmortem_timeline,
)
from ai_engineer.llm.model import ScriptedModel, ToolCall, tool_use

SPANS = (
    Span("t1", "a", "", "chat", 1000.0, 1009.0, "error"),
    Span(
        "t1", "b", "a", "retrieve", 1000.2, 1001.0,
        attributes=(("gen_ai.request.model", "claude-haiku-4-5"),),
    ),
    Span("t1", "c", "a", "generate", 1001.0, 1009.0, "error"),
)  # fmt: skip
DEPLOYS = (
    Deploy(900.0, "router", "v41", "new retriever"),
    Deploy(990.0, "router", "v42", "shorter prompt"),
    Deploy(-4000.0, "router", "v40", "old"),
)
DATA = Telemetry(
    metrics={"error_rate": ((900.0, 0.001), (1000.0, 0.08))},
    traces={"t1": SPANS},
    deploys=DEPLOYS,
    runbooks={"router": "Roll back the last deploy first."},
)
ALERT = Alert("HighErrorRate", 1005.0, "error_rate", "8 percent errors")


def answer(actions):
    return json.dumps(
        {
            "hypotheses": [
                {"cause": "deploy v42", "evidence": "0.08 at 1000"}
            ],
            "actions": actions,
        }
    )


def test_tools_are_read_only_and_answer_from_memory():
    tools = build_tools(DATA)
    assert all(tools[s["name"]].read_only for s in tools.specs())
    out = tools.call(
        ToolCall(
            "1",
            "query_metric",
            {"metric": "error_rate", "start": 950.0, "end": 1100.0},
        )
    )
    assert json.loads(out["content"]) == [[1000.0, 0.08]]
    bad = tools.call(
        ToolCall(
            "2", "query_metric", {"metric": "nope", "start": 0.0, "end": 1.0}
        )
    )
    assert bad["is_error"]
    assert not any(
        "write" in s["name"] or "delete" in s["name"] for s in tools.specs()
    )  # fmt: skip


def test_agent_investigates_then_proposes_and_code_filters():
    model = ScriptedModel(
        [
            tool_use("1", "recent_deploys", before=1005.0, hours=0.5),
            answer(
                [
                    {"action": "rollback", "target": "router v42"},
                    {"action": "raise_threshold", "target": "HighErrorRate"},
                ]
            ),
        ]
    )
    report = hypothesize(model, ALERT, DATA)
    assert report.steps == 2
    assert report.hypotheses[0].cause == "deploy v42"
    status = {p.action: p.status for p in report.proposals}
    assert status == {
        "rollback": "needs_approval",
        "raise_threshold": "blocked",
    }
    assert "v42" in model.requests[1].messages[-1]["content"][0]["content"]


def test_approval_needs_allowed_action_and_a_named_human():
    report = hypothesize(
        ScriptedModel([answer([{"action": "rollback", "target": "v42"},
                               {"action": "raise_threshold"}])]),
        ALERT, DATA,
    )  # fmt: skip
    good, bad = report.proposals
    assert approve(good, "on-call").approved_by == "on-call"
    with pytest.raises(PermissionError, match="not an allowed"):
        approve(bad, "on-call")
    with pytest.raises(PermissionError, match="named human"):
        approve(good, " ")


def test_unparseable_answer_proposes_nothing():
    report = hypothesize(
        ScriptedModel(["I think it is the network."]), ALERT, DATA
    )
    assert report.stop == "unparseable"
    assert report.proposals == ()


def test_postmortem_timeline_is_ordered_and_skips_old_deploys():
    rows = postmortem_timeline(SPANS, DEPLOYS)
    kinds = [r.kind for r in rows]
    assert kinds == ["deploy", "deploy", "error", "slow", "error"]
    assert [r.at for r in rows] == sorted(r.at for r in rows)
    texts = " ".join(r.text for r in rows)
    assert "v42" in texts
    assert "v40" not in texts
    assert rows[3].text == "chat took 9.0s"
    assert postmortem_timeline((), DEPLOYS) == ()
