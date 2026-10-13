import pytest

from ai_engineer.enterprise.evidence import Evidence, release_gate


def record(test_id, verdict, at=0.0):
    return Evidence(
        test_id, "run-1", at, "bw.orders -> fabric.orders", "12", "12", verdict
    )


def test_a_gate_opens_only_when_every_required_test_passed():
    gate = release_gate(
        [record("count", "pass"), record("sums", "pass")],
        required=["count", "sums"],
    )
    assert gate.passed
    assert dict(gate.counts) == {"pass": 2, "fail": 0, "error": 0, "skip": 0}


def test_missing_evidence_blocks_like_a_failure():
    gate = release_gate([record("count", "pass")], required=["count", "sums"])
    assert (gate.passed, gate.blocking) == (False, ("sums",))


def test_an_error_is_not_a_pass_even_when_not_required():
    gate = release_gate(
        [record("count", "pass"), record("drift", "error")],
        required=["count"],
    )
    assert gate.blocking == ("drift",)


def test_the_latest_run_of_a_test_wins():
    flaky = [record("count", "fail", at=1.0), record("count", "pass", at=2.0)]
    assert release_gate(flaky, required=["count"]).passed
    assert not release_gate(flaky[:1], required=["count"]).passed


@pytest.mark.parametrize("field", ["test_id", "run_id", "scope"])
def test_an_evidence_record_needs_an_identity_and_a_scope(field):
    values = {
        "test_id": "t",
        "run_id": "r",
        "recorded_at": 0.0,
        "scope": "s",
        "expected": "1",
        "observed": "1",
        "verdict": "pass",
    }
    values[field] = ""
    with pytest.raises(ValueError, match="required"):
        Evidence(**values)


def test_an_unknown_verdict_is_rejected():
    with pytest.raises(ValueError, match="verdict"):
        record("count", "maybe")
