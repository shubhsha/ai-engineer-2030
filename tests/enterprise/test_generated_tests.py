"""Chapter 40: model-written tests are screened before they are trusted."""

from __future__ import annotations

from ai_engineer.enterprise.evidence import release_gate
from ai_engineer.enterprise.generated_tests import (
    Check,
    build,
    fails,
    naive_score,
    screen,
    screening_evidence,
    text_mutants,
)
from tests.foundations.test_testing_mutants import (
    SOURCE,
    examples,
    load,
    mutants,
)

SWAPS = [
    ("if match is None", "if match is not None"),
    ('endswith(".")', 'endswith("!")'),
    ("lines[1].strip()", "lines[1]"),
    ("not lines[0].strip()", "not lines[0]"),
    ('if not ln.startswith("#")', "if True"),
]
CANDIDATES = [
    (
        "edge72",
        "def check_it(check):\n"
        "    assert check('fix: ' + 'x' * 67) == []\n"
        "    assert check('fix: ' + 'x' * 68) != []\n",
    ),
    (
        "period",
        "def check_it(check):\n"
        "    assert 'period' in ' '.join(check('fix: add x.'))\n",
    ),
    (
        "blank",
        "def check_it(check):\n"
        "    assert check('   ') == ['the message is empty']\n",
    ),
    (
        "comments",
        "def check_it(check):\n"
        "    assert check('# note\\nfix: add x') == []\n",
    ),
    (
        "smoke",
        "def check_it(check):\n"
        "    assert isinstance(check('fix: add x\\n\\nbody'), list)\n",
    ),
    (
        "copy_of_example",
        "def check_it(check):\n    assert check('Fixed stuff') != []\n",
    ),
    ("no_build", "def check_it(check)\n    assert check('x')\n"),
    (
        "hallucinated",
        "def check_it(check):\n    assert check('fix: add x.') == []\n",
    ),
    (
        "flaky",
        "calls = [0]\n"
        "def check_it(check):\n"
        "    calls[0] += 1\n"
        "    assert calls[0] % 2 == 1\n",
    ),
]


def subject() -> tuple[Check, list[Check]]:
    sources = [code for _, code in mutants(SOURCE)]
    sources += text_mutants(SOURCE, SWAPS)
    return load(SOURCE), [load(code) for code in sources]


def test_the_filters_sort_the_candidates() -> None:
    original, dead_ends = subject()
    known = frozenset(i for i, m in enumerate(dead_ends) if fails(examples, m))
    screens = screen(CANDIDATES, original, dead_ends, known=known)
    assert {s.name: s.outcome for s in screens} == {
        "edge72": "accepted",
        "period": "accepted",
        "blank": "accepted",
        "comments": "accepted",
        "smoke": "adds_nothing",
        "copy_of_example": "adds_nothing",
        "no_build": "no_build",
        "hallucinated": "fails_original",
        "flaky": "flaky",
    }


def test_the_score_before_filtering_is_flattered() -> None:
    original, mutated = subject()
    known = {i for i, m in enumerate(mutated) if fails(examples, m)}
    assert len(mutated) == 9
    assert len(known) == 4
    assert naive_score(CANDIDATES, mutated) == 1.0
    accepted = [
        s
        for s in screen(CANDIDATES, original, mutated, known=frozenset(known))
        if s.outcome == "accepted"
    ]
    dead = known.union(*(s.killed for s in accepted))
    assert len(dead) == 8  # mutant 6 survives


def test_a_candidate_that_does_not_build_is_an_error_not_a_pass() -> None:
    assert build(CANDIDATES[6][1]) is None
    original, mutated = subject()
    screens = screen(CANDIDATES, original, mutated)
    records = screening_evidence(
        screens, run_id="gen-1", recorded_at=1.0, scope="commitmsg"
    )
    by_id = {r.test_id: r.verdict for r in records}
    assert by_id["generated.no_build"] == "error"
    assert by_id["generated.edge72"] == "pass"
    gate = release_gate(records, required=["generated.edge72"])
    assert gate.blocking == ("generated.no_build",)
