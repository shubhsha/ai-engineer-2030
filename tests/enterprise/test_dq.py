import json
import math

import pytest

from ai_engineer.enterprise.dq import (
    Rule,
    classic_zscore,
    profile,
    propose_rules,
    robust_anomaly,
    rules_from_json,
    run_rules,
    shape,
    triage,
)
from ai_engineer.enterprise.evidence import Evidence
from ai_engineer.llm.model import ScriptedModel
from ai_engineer.llm.structured import StructuredOutputError

ROWS = [
    {"id": 1, "amount": 10.0, "country": "DE", "sent": 5, "made": 4},
    {"id": 2, "amount": None, "country": "DE", "sent": 5, "made": 4},
    {"id": 2, "amount": 9000.0, "country": "XX", "sent": 3, "made": 4},
    {"id": 4, "amount": 20.0, "country": "FR", "sent": 5, "made": 4},
]


def rule(name, kind, arg="", col="amount", sev="warn", dim="validity"):
    return Rule(name, dim, sev, "orders-team", col, kind, arg)


def test_profile_counts_nulls_range_and_shape():
    p = profile(ROWS)
    assert p["amount"].nulls == 1
    assert p["amount"].null_rate == 0.25
    assert (p["amount"].minimum, p["amount"].maximum) == (10.0, 9000.0)
    assert p["country"].distinct == 3
    assert p["country"].top_shape == "AA"
    assert shape("DE-1042") == "AA-9999"


def test_each_kind_finds_its_rows(clock):
    rules = [
        rule("nn", "not_null", dim="completeness"),
        rule("rg", "range", "0:5000"),
        rule("set", "in_set", "DE|FR", col="country"),
        rule("pat", "pattern", "[A-Z]{2}", col="country"),
        rule("uq", "unique", col="id", dim="uniqueness"),
        rule("nb", "not_before", "made", col="sent", dim="consistency"),
    ]
    got = {
        e.test_id: e.observed
        for e in run_rules(ROWS, rules, "r1", clock).evidence
    }
    assert got["nn"].startswith("1 ")
    assert got["rg"].startswith("1 ")
    assert got["set"].startswith("1 ")
    assert got["pat"].startswith("0 ")
    assert got["uq"].startswith("1 ")
    assert got["nb"].startswith("1 ")


def test_severity_routes_rows(clock):
    warn = run_rules(ROWS, [rule("rg", "range", "0:5000")], "r", clock)
    assert warn.may_load
    assert len(warn.clean) == 4
    quar = rule("rgq", "range", "0:5000", sev="quarantine")
    q = run_rules(ROWS, [quar], "r", clock)
    assert q.may_load
    assert len(q.clean) == 3
    assert q.quarantined[0].rules == ("rgq",)
    blk = rule("rgb", "range", "0:5000", sev="block")
    b = run_rules(ROWS, [blk], "r", clock)
    assert not b.may_load
    assert b.blocking == ("rgb",)


def test_evidence_shape_and_missing_column_is_error(clock):
    clock.advance(50)
    r = run_rules(
        ROWS, [rule("x", "not_null", col="nope", sev="block")], "r", clock
    )
    e = r.evidence[0]
    assert isinstance(e, Evidence)
    assert e.verdict == "error"
    assert e.recorded_at == 50
    assert "owner=orders-team" in e.location
    assert not r.may_load


def test_max_age_uses_the_injected_clock(clock):
    clock.advance(1000)
    rows = [{"t": 900.0}, {"t": 100.0}]
    r = run_rules(rows, [rule("age", "max_age", "500", col="t")], "r", clock)
    assert r.evidence[0].observed.startswith("1 ")


def test_rules_file_and_validation():
    text = (
        '[{"name":"a","dimension":"validity","severity":"warn",'
        '"owner":"o","column":"c","kind":"not_null"}]'
    )
    assert rules_from_json(text)[0].name == "a"
    with pytest.raises(ValueError, match="unique"):
        rules_from_json(text[:-1] + "," + text[1:])
    with pytest.raises(ValueError, match="severity"):
        Rule("a", "validity", "loud", "o", "c", "not_null")
    with pytest.raises(ValueError, match="required"):
        Rule("a", "validity", "warn", "", "c", "not_null")


def test_robust_anomaly_flags_a_drop_and_ignores_an_old_outlier():
    hist = [(0, v) for v in (100, 102, 98, 101, 5, 99)]
    drop = robust_anomaly(hist, 40, 0)
    assert drop.flagged
    assert drop.median == 99.5
    assert drop.score < -3.5
    assert not robust_anomaly(hist, 100, 0).flagged
    # the classic score is blinded by the old outlier
    assert abs(classic_zscore([v for _, v in hist], 40)) < 2


def test_mad_zero_and_short_history():
    flat = [(1, 50.0)] * 5
    assert robust_anomaly(flat, 50.0, 1).score == 0.0
    assert robust_anomaly(flat, 49.0, 1).score == -math.inf
    with pytest.raises(ValueError, match="at least 4"):
        robust_anomaly(flat[:3], 50.0, 1)


def test_triage_sorts_proposals(clock):
    trusted = [r for r in ROWS if r["id"] != 2 and r["amount"] is not None]
    prof = profile(ROWS)
    existing = [rule("have", "not_null")]
    props = [
        rule("p_nn", "not_null"),
        rule("p_echo", "range", "10.0:9000.0"),
        rule("p_ok", "range", "0:5000"),
        rule("p_bad", "in_set", "DE", col="country"),
        rule("p_nocol", "not_null", col="zzz"),
    ]
    got = {
        t.rule.name: t.status
        for t in triage(props, existing, trusted, prof, clock)
    }
    assert got == {
        "p_nn": "covered",
        "p_echo": "echo",
        "p_ok": "new",
        "p_bad": "wrong",
        "p_nocol": "wrong",
    }


def test_propose_rules_are_warn_only_and_owned(clock):
    reply = json.dumps(
        {
            "rules": [
                {
                    "name": "amount_nonneg",
                    "dimension": "validity",
                    "column": "amount",
                    "kind": "range",
                    "arg": "0:5000",
                }
            ]
        }
    )
    model = ScriptedModel([reply])
    got = propose_rules(model, profile(ROWS), "orders-team")
    assert got[0].severity == "warn"
    assert got[0].owner == "orders-team"
    assert "9000.0" in str(model.requests[0].messages[0])
    bad = ScriptedModel([reply.replace("validity", "vibes")] * 3)
    with pytest.raises(StructuredOutputError):
        propose_rules(bad, profile(ROWS), "orders-team")
