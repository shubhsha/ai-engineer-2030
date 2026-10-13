import math
import random

import pytest

from ai_engineer.enterprise.slo import (
    Bucket,
    MultiWindowAlert,
    Slo,
    burn_rate,
    error_budget,
    psi,
    psi_shares,
)

SLO = Slo("chat", 0.995)
DAY = 1440.0


def test_budget_of_a_30_day_995_slo_is_216_minutes():
    assert error_budget(SLO, 30 * DAY) == pytest.approx(216.0)


def test_burn_rate_and_empty_window():
    assert burn_rate(5, 100, SLO) == pytest.approx(10.0)
    assert burn_rate(0, 0, SLO) == 0.0


def test_14_4_spends_two_percent_in_one_hour():
    hour = 60.0
    spent = 14.4 * hour / (30 * DAY)
    assert spent == pytest.approx(0.02)


def test_target_must_be_a_share():
    with pytest.raises(ValueError, match=r"between|need"):
        Slo("x", 1.0)


def minutes(rate_by_minute, total=100):
    return [
        Bucket(m * 60.0, total, round(total * rate_by_minute(m)))
        for m in range(0, 24 * 60)
    ]


def test_short_outage_pages_only_while_it_lasts():
    alert = MultiWindowAlert(SLO)
    ev = minutes(lambda m: 0.2 if 600 <= m < 660 else 0.0)
    assert alert.evaluate(ev, 630 * 60.0)
    later = [r.severity for r in alert.evaluate(ev, 700 * 60.0)]
    assert "page" not in later
    assert alert.evaluate(ev, 300 * 60.0) == []


def test_steady_leak_opens_a_ticket_not_a_page():
    alert = MultiWindowAlert(SLO)
    ev = [Bucket(m * 60.0, 100, 1) for m in range(4 * 24 * 60)]
    fired = alert.evaluate(ev, 4 * 24 * 3600.0)
    assert [r.severity for r in fired] == ["ticket"]


def test_psi_is_zero_for_the_same_sample_and_grows_with_shift():
    rng = random.Random(2030)
    base = [rng.gauss(0, 1) for _ in range(2000)]
    same = [rng.gauss(0, 1) for _ in range(2000)]
    moved = [rng.gauss(1, 1) for _ in range(2000)]
    assert psi(base, base, 10) == pytest.approx(0.0)
    assert psi(base, same, 10) < 0.1
    assert psi(base, moved, 10) > 0.25


def test_psi_by_hand_four_bins():
    got = psi_shares([0.25] * 4, [0.10, 0.20, 0.30, 0.40])
    want = sum((a - 0.25) * math.log(a / 0.25) for a in (0.1, 0.2, 0.3, 0.4))
    assert got == pytest.approx(want)


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError, match=r"between|need"):
        psi([], [1.0], 4)
