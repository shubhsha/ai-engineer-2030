import pytest

from ai_engineer.evals.harness import (
    Case,
    contains,
    exact,
    json_fields,
    run_suite,
)
from ai_engineer.evals.metrics import (
    paired_compare,
    pass_at_k,
    pass_hat_k,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("passed", "n", "low", "high"),
    [
        (9, 10, 0.596, 0.982),
        (45, 50, 0.786, 0.957),
        (150, 200, 0.686, 0.805),
        (0, 10, 0.0, 0.278),
    ],
)
def test_wilson_interval(passed, n, low, high):
    assert wilson_interval(passed, n) == pytest.approx((low, high), abs=1e-3)


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(11, 10)


def test_pass_at_k_and_pass_hat_k():
    assert pass_at_k(10, 5, 1) == pytest.approx(0.5)
    assert pass_at_k(10, 5, 3) == pytest.approx(11 / 12)
    assert pass_at_k(10, 8, 3) == 1.0  # 3 tries cannot all hit 2 failures
    assert pass_hat_k(10, 5, 3) == pytest.approx(1 / 12)
    assert pass_hat_k(10, 9, 5) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="1 <= k <= n"):
        pass_at_k(10, 5, 11)


@pytest.mark.parametrize(
    ("only_a", "only_b", "p"), [(8, 20, 0.0357), (3, 9, 0.146)]
)
def test_paired_compare_counts_only_disagreements(only_a, only_b, p):
    same = 100 - only_a - only_b
    a = [True] * only_a + [False] * only_b + [True] * same
    b = [False] * only_a + [True] * only_b + [True] * same
    result = paired_compare(a, b)
    assert (result.only_a, result.only_b) == (only_a, only_b)
    assert result.p_value == pytest.approx(p, abs=5e-4)


def test_paired_compare_needs_the_same_cases():
    with pytest.raises(ValueError, match="same cases"):
        paired_compare([True], [True, False])


CASES = [
    Case("capital", "Capital of France?", "Paris", ("geo",)),
    Case("sum", "2 + 2?", "4", ("math",)),
    Case("product", "6 x 7?", "42", ("math",)),
]


def answer(question):
    if question == "6 x 7?":
        raise RuntimeError("model timeout")
    return {"Capital of France?": " paris ", "2 + 2?": "4"}[question]


def test_a_suite_records_errors_and_reports_slices():
    report = run_suite(answer, CASES, exact)
    assert report.pass_rate == pytest.approx(2 / 3)
    assert report.by_tag() == {"geo": (1, 1), "math": (1, 2)}
    [failure] = report.failures()
    assert (failure.case.id, failure.error) == (
        "product",
        "RuntimeError: model timeout",
    )


def test_a_suite_needs_cases_with_unique_ids():
    with pytest.raises(ValueError, match="unique"):
        run_suite(answer, [CASES[0], CASES[0]], exact)


def test_repeated_trials_expose_flakiness_and_widen_the_interval(rng):
    cases = [Case(f"c{i}", str(i), "ok") for i in range(20)]

    def system(question):
        if int(question) % 4 == 0:
            return "wrong"  # a case this system never gets right
        return "ok" if rng.random() < 0.9 else "wrong"

    report = run_suite(system, cases, exact, trials=4)
    assert report.consistency() < report.pass_rate
    passed = sum(o.passed for o in report.outcomes)
    naive_low, naive_high = wilson_interval(passed, len(report.outcomes))
    low, high = report.interval()
    assert high - low > naive_high - naive_low


def test_graders():
    paris = Case("c", "q", "Paris")
    assert exact("  PARIS ", paris)
    assert not exact("Paris, France", paris)
    assert contains("It is Paris, France.", paris)
    assert contains(
        "Sales fell in Q3 in the North.", Case("c", "q", ["q3", "north"])
    )
    labeled = Case("c", "q", {"label": "bug", "reason": "anything"})
    assert json_fields("label")('{"label": "bug", "reason": "x"}', labeled)
    assert not json_fields("label")("label: bug", labeled)
