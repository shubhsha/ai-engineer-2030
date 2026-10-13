"""Chapter 46: publishing test results to a test-management service.

The port is one method. The Azure DevOps adapter only builds the REST
requests (method, path and JSON body) and never sends them, so it is
tested without a network. The in-memory fake keeps what it was given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from ai_engineer.enterprise.evidence import Evidence, Verdict

API = "api-version=7.1"
OUTCOME: dict[Verdict, str] = {
    "pass": "Passed",
    "fail": "Failed",
    "error": "Error",
    "skip": "NotExecuted",
}


@dataclass(frozen=True, slots=True)
class HttpRequest:
    method: str
    path: str
    body: Any


# region test_run_publisher
class TestRunPublisher(Protocol):
    __test__: ClassVar[bool] = False

    def publish(
        self, run_name: str, evidence: tuple[Evidence, ...]
    ) -> None: ...


# endregion


# region azure_publisher
class AzureDevOpsPublisher:
    """Builds the three calls that record one automated test run."""

    __test__: ClassVar[bool] = False

    def __init__(self, project: str) -> None:
        self.project = project

    def requests(
        self, run_name: str, evidence: tuple[Evidence, ...], run_id: int
    ) -> tuple[HttpRequest, ...]:
        base = f"/{self.project}/_apis/test"
        create = HttpRequest(
            "POST",
            f"{base}/runs?{API}",
            {"name": run_name, "automated": True, "state": "InProgress"},
        )
        results = HttpRequest(
            "POST",
            f"{base}/Runs/{run_id}/results?{API}",
            [_result(item) for item in evidence],
        )
        close = HttpRequest(
            "PATCH", f"{base}/runs/{run_id}?{API}", {"state": "Completed"}
        )
        return (create, results, close)


def _result(item: Evidence) -> dict[str, Any]:
    result: dict[str, Any] = {
        "testCaseTitle": item.test_id.rsplit(".", 1)[-1],
        "automatedTestName": item.test_id,
        "outcome": OUTCOME[item.verdict],
        "state": "Completed",
    }
    if item.verdict in ("fail", "error"):
        result["errorMessage"] = item.observed
    return result


# endregion


@dataclass(slots=True)
class InMemoryPublisher:
    __test__: ClassVar[bool] = False
    runs: dict[str, tuple[Evidence, ...]] = field(default_factory=dict)

    def publish(self, run_name: str, evidence: tuple[Evidence, ...]) -> None:
        self.runs[run_name] = evidence
