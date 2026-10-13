import pytest

from ai_engineer.enterprise.evidence import Evidence
from ai_engineer.enterprise.registry import (
    DAY,
    AiSystem,
    Control,
    Event,
    audit_pack,
    control_status,
    coverage,
    render_pack,
    risk_tier,
)

EVAL = Control("C1", "Quality eval", ("NIST Measure",), ("eval",), 30)
SIGN = Control(
    "C2", "Reviewer sign-off", ("ISO 42001",), ("eval", "signoff"), 90
)


def system(sid="s1", area="software-dev", data=(), affected=()):
    return AiSystem(sid, "team-a", "demo", area, data, affected, ("m1",))


def ev(sid, kind, day, verdict="pass"):
    return Evidence(f"{kind}:x", "r1", day * DAY, sid, "ok", "ok", verdict)


def test_tier_is_the_strictest_matching_rule():
    assert risk_tier(system()).tier == "minimal"
    assert risk_tier(system(data=("personal",))).tier == "limited"
    both = system(area="employment", data=("personal",))
    assert risk_tier(both).tier == "high"


def test_a_rule_with_an_unknown_tier_is_rejected():
    from ai_engineer.enterprise.registry import TierRule

    bad = (TierRule("severe", "x", use_areas=("software-dev",)),)
    with pytest.raises(ValueError, match="unknown tier"):
        risk_tier(system(), bad)


def test_fresh_pass_is_covered_and_old_pass_is_stale():
    s = system()
    assert control_status(s, EVAL, [ev("s1", "eval", 90)], 100 * DAY) == (
        "covered"
    )
    assert control_status(s, EVAL, [ev("s1", "eval", 10)], 100 * DAY) == (
        "stale"
    )


def test_missing_failing_and_other_scopes_do_not_count():
    s = system()
    assert control_status(s, EVAL, [], DAY) == "missing"
    assert control_status(s, EVAL, [ev("other", "eval", 1)], DAY) == "missing"
    assert control_status(s, EVAL, [ev("s1", "eval", 1, "fail")], DAY) == (
        "failing"
    )


def test_every_required_kind_is_needed():
    s = system()
    assert control_status(s, SIGN, [ev("s1", "eval", 1)], 2 * DAY) == (
        "missing"
    )


def test_a_rerun_replaces_an_older_failure():
    s = system()
    runs = [ev("s1", "eval", 1, "fail"), ev("s1", "eval", 2)]
    assert control_status(s, EVAL, runs, 3 * DAY) == "covered"


def test_coverage_counts_only_controls_for_the_tier():
    low = system("low")
    high = system("high", area="credit")
    proofs = [ev("high", "eval", 99)]
    got = coverage([low, high], [EVAL, SIGN], proofs, 100 * DAY)
    assert got.covered == (("high", "C1"),)
    assert got.missing == (("high", "C2"),)
    assert got.total() == 2
    assert got.share_covered() == 0.5


def test_audit_pack_has_six_sections_and_names_each_control():
    s = system("hr", area="employment", data=("personal",))
    proofs = [ev("hr", "eval", 99), ev("other", "eval", 99)]
    pack = audit_pack(
        s,
        [EVAL, SIGN],
        proofs,
        100 * DAY,
        incidents=[Event(5.0, "bad answer")],
        changes=[Event(1.0, "model m1 to m2")],
    )
    assert list(pack) == [
        "system_description",
        "risk_assessment",
        "evaluation_results",
        "human_oversight",
        "incident_log",
        "change_history",
    ]
    assert "C1 (NIST Measure): covered" in pack["risk_assessment"]
    assert "C2 (ISO 42001): missing" in pack["risk_assessment"]
    assert "other" not in pack["evaluation_results"]
    assert render_pack(pack).count("## ") == 6


def test_oversight_level_is_validated():
    with pytest.raises(ValueError, match="oversight"):
        AiSystem("a", "o", "p", "x", (), (), (), human_oversight="maybe")
