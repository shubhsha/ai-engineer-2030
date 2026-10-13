import json
import logging

from ai_engineer.agents.multi import (
    WORKER,
    AgentCard,
    Brief,
    TimedModel,
    amdahl,
    audit,
    discover,
    fan_out,
    research,
)
from ai_engineer.llm.model import Reply, ScriptedModel, Usage


def briefs(*objectives):
    return json.dumps(
        {
            "briefs": [
                {
                    "objective": objective,
                    "output": "Bullets, each with its source.",
                    "boundaries": "Public pricing pages only.",
                }
                for objective in objectives
            ]
        }
    )


def test_each_worker_starts_with_only_its_brief():
    lead = ScriptedModel(
        [briefs("Price of vendor A", "Price of vendor B"), "A is cheaper."]
    )
    worker = ScriptedModel(default="- 10 euros a seat [pricing page]")
    out = research(lead, worker, "Which vendor is cheaper for 50 seats?")
    assert out.answer == "A is cheaper."
    assert len(worker.requests) == 2
    for request in worker.requests:
        [message] = request.messages
        assert request.system == WORKER
        assert message["content"].startswith("<objective>\nPrice of vendor")
        assert "50 seats" not in message["content"]
    assert lead.requests[-1].messages[0]["content"].count("<finding>") == 2


def test_too_many_briefs_are_sent_back_to_the_lead():
    lead = ScriptedModel([briefs("a", "b", "c", "d"), briefs("a", "b"), "ok"])
    worker = ScriptedModel(default="a finding")
    research(lead, worker, "q", max_workers=3)
    repair = lead.requests[1].messages[-1]["content"]
    assert "$.briefs: write 1 to 3 briefs, not 4" in repair
    assert len(worker.requests) == 2


def test_a_failed_worker_leaves_a_note_not_a_crash(caplog):
    def slow_on_b(request):
        if "vendor B" in request.messages[0]["content"]:
            raise TimeoutError("no reply in 60 s")
        return "found A"

    worker = ScriptedModel(default=slow_on_b)
    with caplog.at_level(logging.WARNING):
        replies = fan_out(worker, [Brief("vendor A"), Brief("vendor B")])
    assert [r.text for r in replies] == [
        "found A",
        "[worker failed: TimeoutError]",
    ]
    assert replies[1].stop_reason == "error"
    assert "no reply in 60 s" in caplog.text


def test_usage_counts_the_lead_and_every_worker():
    lead = ScriptedModel(
        [
            Reply(briefs("a", "b"), usage=Usage(100, 50)),
            Reply("answer", usage=Usage(300, 40)),
        ]
    )
    worker = ScriptedModel(default=Reply("finding", usage=Usage(1000, 200)))
    out = research(lead, worker, "q")
    assert out.lead_usage == Usage(400, 90)
    assert out.worker_usage == Usage(2000, 400)
    assert out.usage.total == 2890


def test_amdahl_matches_the_closed_form():
    assert amdahl(0.1, 1) == 1.0
    assert round(amdahl(0.1, 5), 3) == 3.571
    assert round(amdahl(0.3, 5), 3) == 2.273
    assert amdahl(0.1, 10**9) < 10


def test_workers_in_parallel_finish_in_rounds_of_the_slot_count():
    for slots, expected in [(1, 120.0), (3, 40.0), (5, 40.0)]:
        timed = TimedModel(ScriptedModel(default="x"), 20.0, slots)
        fan_out(timed, [Brief(f"part {i}") for i in range(6)], limit=slots)
        assert timed.elapsed == expected


def test_audit_catches_overlap_drift_and_unsourced_claims():
    briefs = [
        Brief("Price of vendor A in 2025"),
        Brief("Price of vendor A in 2025 in euros"),
        Brief("Support terms of vendor B"),
    ]
    findings = [
        "- 10 euros a seat [pricing page]",
        "I could not find it, but vendors usually charge more.",
        "- 99 percent uptime",
    ]
    assert audit(briefs, findings) == [
        "briefs 0 and 1 overlap",
        "finding 1: not a bullet: 'I could not find it, but vendo'",
        "finding 2: no source: '- 99 percent uptime'",
    ]
    assert audit(briefs[2:], findings[:1]) == []


def test_discovery_picks_agents_by_skill_and_keeps_the_order():
    cards = [
        AgentCard("pricing", "https://a.example/a2a", ("quote", "compare")),
        AgentCard("legal", "https://b.example/a2a", ("review",)),
        AgentCard("broker", "https://c.example/a2a", ("compare",), "oauth"),
    ]
    found = discover(cards, "compare")
    assert [card.name for card in found] == ["pricing", "broker"]
    assert discover(cards, "translate") == []
