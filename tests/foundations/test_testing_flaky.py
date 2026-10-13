"""Chapter 11: the arithmetic of flaky tests, checked by simulation."""

from __future__ import annotations

import random


# region flake-arithmetic
def test_200_tests_with_half_percent_flakes_pass_37_percent_of_runs(
    rng: random.Random,
) -> None:
    runs, tests, flake_rate = 10_000, 200, 0.005
    green = sum(
        all(rng.random() >= flake_rate for _ in range(tests))
        for _ in range(runs)
    )
    # 0.995 ** 200 = 0.367; one standard deviation of 10,000 runs is 0.005.
    assert abs(green / runs - 0.995**tests) < 0.015


# endregion


def test_three_reruns_hide_a_one_in_twenty_failure(
    rng: random.Random,
) -> None:
    runs = 100_000
    red = sum(all(rng.random() < 0.05 for _ in range(3)) for _ in range(runs))
    # A suite is red only if all three tries fail: 0.05 ** 3 = 0.000125.
    assert red / runs < 0.001
