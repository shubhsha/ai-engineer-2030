from ai_engineer.enterprise.dashboard import (
    DAY,
    check_narrative,
    render_html,
    summarize,
    trend,
    wilson,
)
from ai_engineer.enterprise.evidence import Evidence


def ev(test, verdict, day=0, scope="bw.orders", loc="runs/1.json", run="r1"):
    return Evidence(test, run, day * DAY + 3600, scope, "x", "y", verdict, loc)


def test_skips_stay_out_of_the_denominator():
    rows = summarize([ev("a", "pass"), ev("b", "fail"), ev("c", "skip")])
    row = rows[0]
    assert (row.n, row.passes, row.skips) == (2, 1, 1)
    assert row.rate == 0.5
    assert round(row.skip_share, 3) == 0.333


def test_grouping_by_day_orders_the_days():
    rows = summarize([ev("a", "pass", 2), ev("a", "pass", 0)], by="day")
    assert [r.key for r in rows] == ["1970-01-01", "1970-01-03"]


def test_wilson_with_no_data_is_the_whole_range():
    assert wilson(0, 0) == (0.0, 1.0)
    low, high = wilson(50, 50)
    assert round(low, 3) == 0.929
    assert high == 1.0


def _days(passes_n):
    return [
        r
        for p, n in passes_n
        for r in summarize(
            [ev(f"t{i}", "pass" if i < p else "fail") for i in range(n)]
        )
    ]


def test_a_small_wobble_is_flat_and_a_real_drop_is_down():
    assert trend(_days([(96, 100), (97, 100)])) == "flat"
    assert trend(_days([(97, 100), (80, 100)])) == "down"
    assert trend(_days([(80, 100), (97, 100)])) == "up"
    assert trend(_days([(1, 1)])) == "flat"


def test_the_page_escapes_labels_and_states_status_in_words():
    bad = ev("a", "fail", scope="<script>alert(1)</script>")
    page = render_html(summarize([bad]), "T & <U>", as_of=DAY)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "T &amp; &lt;U&gt;" in page
    assert "BLOCKED" in page
    assert 'href="runs/1.json"' in page


def test_old_evidence_is_labeled_stale():
    page = render_html(summarize([ev("a", "pass")]), "T", as_of=5 * DAY)
    assert "STALE" in page


def test_a_sparkline_appears_when_history_exists():
    days = summarize([ev("a", "pass", 0), ev("a", "fail", 1)], by="day")
    now = summarize([ev("a", "pass", 1)])
    page = render_html(now, "T", DAY, history={"bw.orders": days})
    assert "<polyline" in page
    assert "<script" not in page


def test_narrative_flags_missing_unknown_and_contradicted_claims():
    records = [ev("count", "pass"), ev("sums", "fail")]
    text = (
        "The count check passed [count@r1]. "
        "Sums passed [sums@r1]. "
        "Totals failed [totals@r1]. "
        "Everything looks fine."
    )
    got = check_narrative(text, records)
    assert [p.reason for p in got] == [
        "contradicts the record",
        "unknown id totals@r1",
        "cites no evidence",
    ]


def test_skipping_failures_lifts_the_rate_but_not_the_skip_share():
    before = [ev(f"t{i}", "pass" if i < 90 else "fail") for i in range(100)]
    after = [
        ev(f"t{i}", "pass" if i < 90 else "fail" if i < 92 else "skip")
        for i in range(100)
    ]
    a, b = summarize(before)[0], summarize(after)[0]
    assert (a.rate, b.rate) == (0.9, 90 / 92)
    assert (a.skip_share, b.skip_share) == (0.0, 0.08)
    assert "8 skipped (8% of all)" in render_html([b], "T", as_of=DAY)


def test_a_scope_with_only_skips_is_too_few_not_green():
    rows = summarize([ev("a", "skip"), ev("b", "skip")])
    assert (rows[0].n, rows[0].rate, rows[0].low, rows[0].high) == (
        0,
        0.0,
        0.0,
        1.0,
    )
    assert "TOO FEW" in render_html(rows, "T", as_of=DAY)


def test_a_sentence_that_explains_a_cause_is_refused():
    records = [ev("sums", "fail")]
    text = "Sums failed because a table was reloaded [sums@r1]."
    assert [p.reason for p in check_narrative(text, records)] == [
        "claims a cause"
    ]
