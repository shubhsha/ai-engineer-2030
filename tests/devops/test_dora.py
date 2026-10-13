from __future__ import annotations

import pytest

from ai_engineer.devops.dora import DAY, HOUR, Deployment, Incident, four_keys


def _dep(n: int, day: float, lead_h: float) -> Deployment:
    at = day * DAY
    return Deployment(f"d{n}", at - lead_h * HOUR, at)


# region test_four_keys
def test_four_keys_by_hand() -> None:
    deps = [_dep(i, i, 2.0 * (i + 1)) for i in range(10)]
    incidents = [
        Incident("d3", 3 * DAY, 3 * DAY + 1 * HOUR),
        Incident("d7", 7 * DAY, 7 * DAY + 3 * HOUR),
    ]
    keys = four_keys(deps, incidents, (0.0, 10 * DAY))
    assert keys.deploys_per_day == 1.0
    assert keys.lead_time_hours == 11.0
    assert keys.change_failure_rate == 0.2
    assert keys.time_to_restore_hours == 2.0


# endregion


def test_empty_window_gives_zeros() -> None:
    keys = four_keys([], [], (0.0, DAY))
    assert keys.deploys_per_day == 0.0


def test_window_bounds_and_bad_window() -> None:
    dep = _dep(0, 5, 1.0)
    assert four_keys([dep], [], (0.0, 5 * DAY)).deploys_per_day == 0.0
    with pytest.raises(ValueError, match="positive"):
        four_keys([dep], [], (5.0, 5.0))


def test_deploy_before_commit_is_refused() -> None:
    with pytest.raises(ValueError, match="before"):
        Deployment("x", 10.0, 5.0)


def test_two_incidents_on_one_deploy_count_one_failure() -> None:
    deps = [_dep(0, 1, 1.0), _dep(1, 2, 1.0)]
    inc = [Incident("d0", 0.0, HOUR), Incident("d0", 1.0, 2 * HOUR)]
    assert four_keys(deps, inc, (0.0, 4 * DAY)).change_failure_rate == 0.5
