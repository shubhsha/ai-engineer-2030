"""Chapter 46: the four DORA keys from deployment records.

Times are seconds. A deployment records when its oldest commit was made
and when it reached production. An incident records when it began and
when service came back, and names the deployment that caused it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median

DAY = 86_400.0
HOUR = 3_600.0


@dataclass(frozen=True, slots=True)
class Deployment:
    deploy_id: str
    committed_at: float
    deployed_at: float

    def __post_init__(self) -> None:
        if self.deployed_at < self.committed_at:
            raise ValueError("deployed before it was committed")


@dataclass(frozen=True, slots=True)
class Incident:
    deploy_id: str
    started_at: float
    restored_at: float


@dataclass(frozen=True, slots=True)
class FourKeys:
    deploys_per_day: float
    lead_time_hours: float
    change_failure_rate: float
    time_to_restore_hours: float


# region four_keys
def four_keys(
    deployments: Iterable[Deployment],
    incidents: Iterable[Incident],
    window: tuple[float, float],
) -> FourKeys:
    """The four keys for deployments that landed inside ``window``."""
    start, end = window
    if end <= start:
        raise ValueError("window must have positive length")
    inside = [d for d in deployments if start <= d.deployed_at < end]
    if not inside:
        return FourKeys(0.0, 0.0, 0.0, 0.0)
    ids = {d.deploy_id for d in inside}
    hit = [i for i in incidents if i.deploy_id in ids]
    lead = [(d.deployed_at - d.committed_at) / HOUR for d in inside]
    repair = [(i.restored_at - i.started_at) / HOUR for i in hit]
    failed = {i.deploy_id for i in hit}
    return FourKeys(
        deploys_per_day=len(inside) / ((end - start) / DAY),
        lead_time_hours=median(lead),
        change_failure_rate=len(failed) / len(inside),
        time_to_restore_hours=median(repair) if repair else 0.0,
    )


# endregion
