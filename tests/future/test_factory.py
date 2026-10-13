from __future__ import annotations

import numpy as np
import pytest

from ai_engineer.future.factory import (
    Gate,
    Station,
    bottleneck,
    simulate_factory,
)


def line(code: int = 20, review: int = 12) -> list[Station]:
    return [
        Station("triage", 40),
        Station("spec", 40),
        Station("code", code, defect_rate=0.30),
        Station("test", 30, catch_rate=0.5),
        Station("review", review, catch_rate=0.6),
    ]


def run(
    stations: list[Station], gates: list[Gate] | None = None
) -> tuple[int, int, float, int, dict[str, int]]:
    r = simulate_factory(
        stations, gates or [], 500, 80, np.random.default_rng(2030)
    )
    assert r.finished_day is not None
    return (
        r.finished_day,
        r.escaped,
        r.throughput,
        r.stops,
        dict(r.peak_queues),
    )


def test_bottleneck_is_least_capacity_first_on_tie() -> None:
    assert bottleneck(line()).name == "review"
    tied = [Station("a", 5), Station("b", 5), Station("c", 9)]
    assert bottleneck(tied).name == "a"
    with pytest.raises(ValueError, match="at least one"):
        bottleneck([])


def test_more_coding_agents_grow_the_queue_not_throughput() -> None:
    _, _, base, _, base_q = run(line())
    _, _, more, _, more_q = run(line(code=60))
    assert more_q["review"] > base_q["review"]
    assert more <= base * 1.05


def test_raising_the_constraint_raises_throughput() -> None:
    _, _, more, _, _ = run(line(code=60))
    _, _, auto, _, _ = run(line(code=60, review=30))
    assert auto > 1.5 * more


def test_a_gate_cuts_escaped_defects() -> None:
    _, esc_open, _, stops_open, _ = run(line())
    _, esc_gated, _, stops, _ = run(line(), [Gate("test", 0.10)])
    assert stops_open == 0
    assert stops > 0
    assert esc_gated < esc_open / 2


def test_evidence_has_one_record_per_station_day() -> None:
    r = simulate_factory(
        line(), [Gate("test", 0.10)], 100, 30, np.random.default_rng(2030)
    )
    keys = [(e.scope, e.run_id) for e in r.evidence]
    assert len(keys) == len(set(keys))
    fails = [e for e in r.evidence if e.verdict == "fail"]
    assert len(fails) == r.stops
    assert all(e.scope == "test" for e in fails)


def test_same_seed_same_result_and_bad_input_rejected() -> None:
    assert run(line()) == run(line())
    with pytest.raises(ValueError, match="capacity"):
        Station("x", 0)
    with pytest.raises(ValueError, match="gates must match"):
        simulate_factory(
            line(), [Gate("nope", 0.1)], 5, 5, np.random.default_rng(1)
        )
