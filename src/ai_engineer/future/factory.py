"""Chapter 71: the software factory as a production line.

Work moves through stations (triage, specification, code, test, review,
release). Each station has a daily capacity, may introduce a defect, and
may catch one. A gate watches a station's catch rate and stops the line
upstream of it when the rate crosses a threshold, the way an andon cord
does. Every station-day writes one ``Evidence`` record.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ai_engineer.enterprise.evidence import Evidence


@dataclass(frozen=True, slots=True)
class Station:
    """One step of the line.

    ``defect_rate`` is the chance that a piece of work leaves this station
    with a new defect. ``catch_rate`` is the chance that the station
    notices an existing defect and sends the work back to where the defect
    began.
    """

    name: str
    capacity: int
    defect_rate: float = 0.0
    catch_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        for rate in (self.defect_rate, self.catch_rate):
            if not 0.0 <= rate <= 1.0:
                raise ValueError("rates must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class Gate:
    """Stop the line when a station catches more than ``threshold``.

    The rate is defects caught divided by work processed on one day, and
    it is judged only when at least ``min_sample`` pieces were processed.
    """

    station: str
    threshold: float
    min_sample: int = 10


@dataclass(frozen=True, slots=True)
class FactoryResult:
    released: int
    escaped: int
    days: int
    stops: int
    finished_day: int | None
    final_queues: tuple[tuple[str, int], ...]
    peak_queues: tuple[tuple[str, int], ...]
    evidence: tuple[Evidence, ...]

    @property
    def throughput(self) -> float:
        """Pieces released per day, up to the day the last one left."""
        return self.released / (self.finished_day or self.days)


# region bottleneck
def bottleneck(stations: Sequence[Station]) -> Station:
    """The station with the least capacity; the first one on a tie."""
    if not stations:
        raise ValueError("a line needs at least one station")
    return min(stations, key=lambda s: s.capacity)


# endregion bottleneck


def simulate_factory(
    stations: Sequence[Station],
    gates: Sequence[Gate],
    issues: int,
    days: int,
    rng: np.random.Generator,
    *,
    fix_factor: float = 0.5,
) -> FactoryResult:
    """Run ``issues`` pieces of work through the line for ``days`` days.

    Work is a deque per station; -1 is clean work and k >= 0 is work
    carrying a defect that station k introduced. Stations run from the
    end of the line backward, so work moves one station per day. A tripped
    gate idles every station before it for the next day and multiplies
    those stations' defect rates by ``fix_factor``: the cause is fixed.
    """
    names = [s.name for s in stations]
    gate_at = {g.station: g for g in gates}
    if len(names) != len(set(names)) or set(gate_at) - set(names):
        raise ValueError("station names must be unique; gates must match")
    rates = [s.defect_rate for s in stations]
    queues: list[deque[int]] = [deque() for _ in stations]
    queues[0].extend([-1] * issues)
    peaks = [issues] + [0] * (len(stations) - 1)
    released = escaped = stops = 0
    finished: int | None = None
    stopped_before = -1  # stations with index below this idle today
    evidence: list[Evidence] = []
    for day in range(days):
        idle_below, stopped_before = stopped_before, -1
        for k in range(len(stations) - 1, -1, -1):
            if k < idle_below:
                continue
            st = stations[k]
            done = [
                queues[k].popleft()
                for _ in range(min(st.capacity, len(queues[k])))
            ]
            caught = 0
            # region station_work
            for item in done:
                if item >= 0 and rng.random() < st.catch_rate:
                    queues[item].append(-1)  # back to where it began
                    caught += 1
                    continue
                if item < 0 and rng.random() < rates[k]:
                    item = k
                if k + 1 < len(stations):
                    queues[k + 1].append(item)
                else:
                    released += 1
                    escaped += item >= 0
            # endregion
            if not done:
                continue
            # region gate_trip
            rate = caught / len(done)
            gate = gate_at.get(st.name)
            tripped = (
                gate is not None
                and len(done) >= gate.min_sample
                and rate > gate.threshold
            )
            if tripped and gate is not None:
                stops += 1
                stopped_before = k
                for j in range(k):
                    rates[j] *= fix_factor
            # endregion
            evidence.append(
                Evidence(
                    test_id=f"{st.name}-catch-rate",
                    run_id=f"day-{day}",
                    recorded_at=float(day),
                    scope=st.name,
                    expected=(
                        f"catch rate <= {gate.threshold}"
                        if gate
                        else "recorded"
                    ),
                    observed=f"{rate:.2f} of {len(done)}",
                    verdict="fail" if tripped else "pass",
                )
            )
        if released == issues and finished is None:
            finished = day + 1
        for k, q in enumerate(queues):
            peaks[k] = max(peaks[k], len(q))
    return FactoryResult(
        released=released,
        escaped=escaped,
        days=days,
        stops=stops,
        finished_day=finished,
        final_queues=tuple(
            (n, len(q)) for n, q in zip(names, queues, strict=True)
        ),
        peak_queues=tuple(zip(names, peaks, strict=True)),
        evidence=tuple(evidence),
    )
