from __future__ import annotations

import pytest

from ai_engineer.devops.junit import UnsafeXml, parse_junit
from ai_engineer.enterprise.evidence import release_gate

REPORT = """<testsuites><testsuite name="s">
<testcase classname="pkg.test_a" name="test_ok" file="pkg/test_a.py"/>
<testcase classname="pkg.test_a" name="test_bad">
<failure message="assert 3 == 4">trace</failure></testcase>
<testcase classname="pkg.test_a" name="test_boom">
<error message="ConnectionError">trace</error></testcase>
<testcase classname="pkg.test_a" name="test_later">
<skipped message="needs gpu"/></testcase>
</testsuite></testsuites>"""


# region test_junit_verdicts
def test_junit_verdicts_keep_skip_and_error_apart() -> None:
    items = parse_junit(REPORT, run_id="r1", recorded_at=1.0)
    verdicts = {i.test_id.split(".")[-1]: i.verdict for i in items}
    assert verdicts == {
        "test_ok": "pass",
        "test_bad": "fail",
        "test_boom": "error",
        "test_later": "skip",
    }


# endregion


def test_skipped_required_test_blocks_the_gate() -> None:
    items = parse_junit(REPORT, run_id="r1", recorded_at=1.0)
    gate = release_gate(items, required=["pkg.test_a.test_later"])
    assert not gate.passed
    assert "pkg.test_a.test_later" in gate.blocking


def test_failure_message_becomes_observed() -> None:
    items = parse_junit(REPORT, run_id="r1", recorded_at=1.0)
    bad = next(i for i in items if i.test_id.endswith("test_bad"))
    assert bad.observed == "assert 3 == 4"


# region test_junit_refuses_dtd
def test_junit_refuses_entity_bomb() -> None:
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">'
        '<!ENTITY b "&a;&a;&a;&a;">]><testsuite>&b;</testsuite>'
    )
    with pytest.raises(UnsafeXml):
        parse_junit(bomb, run_id="r", recorded_at=0.0)


# endregion


def test_single_testsuite_root_is_accepted() -> None:
    xml = '<testsuite><testcase name="t"/></testsuite>'
    (item,) = parse_junit(xml, run_id="r", recorded_at=0.0)
    assert item.test_id == "t"
    assert item.verdict == "pass"
