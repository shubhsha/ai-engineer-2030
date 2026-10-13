from __future__ import annotations

from ai_engineer.devops.ado import (
    AzureDevOpsPublisher,
    InMemoryPublisher,
    TestRunPublisher,
)
from ai_engineer.devops.junit import parse_junit

XML = """<testsuite>
<testcase classname="a.b" name="ok"/>
<testcase classname="a.b" name="bad"><failure message="boom"/></testcase>
<testcase classname="a.b" name="gone"><skipped/></testcase>
</testsuite>"""


# region test_azure_payloads
def test_azure_requests_follow_the_rest_shapes() -> None:
    evidence = parse_junit(XML, run_id="r", recorded_at=0.0)
    create, results, close = AzureDevOpsPublisher("shop").requests(
        "nightly", evidence, run_id=31
    )
    assert create.method == "POST"
    assert create.path == "/shop/_apis/test/runs?api-version=7.1"
    assert create.body["automated"] is True
    assert results.path.endswith("/Runs/31/results?api-version=7.1")
    outcomes = [r["outcome"] for r in results.body]
    assert outcomes == ["Passed", "Failed", "NotExecuted"]
    assert results.body[1]["errorMessage"] == "boom"
    assert "errorMessage" not in results.body[0]
    assert close.method == "PATCH"
    assert close.body == {"state": "Completed"}


# endregion


def test_in_memory_publisher_keeps_what_it_got() -> None:
    evidence = parse_junit(XML, run_id="r", recorded_at=0.0)
    fake = InMemoryPublisher()
    port: TestRunPublisher = fake
    port.publish("nightly", evidence)
    assert fake.runs["nightly"] == evidence
