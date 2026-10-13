"""Chapter 49: threat model, retrieval ACLs and the red-team suite."""

from __future__ import annotations

import random

import pytest

from ai_engineer.enterprise.acl import (
    AclChunk,
    AclIndex,
    Principal,
    authorized,
    retrieve_post_filter,
    retrieve_secure,
    retrieve_unfiltered,
)
from ai_engineer.enterprise.evidence import Evidence, release_gate
from ai_engineer.enterprise.redteam import (
    AttackCase,
    System,
    render_safe,
    run_suite,
)
from ai_engineer.enterprise.threats import (
    Element,
    ThreatModel,
    rag_assistant,
    stride_for,
)
from ai_engineer.llm.model import Request, ScriptedModel

GROUPS = ["hr", "fin", "eng", "legal"]
WORDS = ["budget", "salary", "launch", "audit", "merger", "roadmap", "bonus"]


def corpus(rng: random.Random, n: int = 60) -> list[AclChunk]:
    chunks = []
    for i in range(n):
        text = " ".join(rng.choices(WORDS, k=6)) + f" doc{i}"
        who = frozenset(rng.sample(GROUPS, rng.choice([0, 1, 1, 2])))
        chunks.append(AclChunk(f"c{i}", text, who))
    return chunks


def test_stride_letters_follow_element_kind() -> None:
    assert stride_for(Element("u", "entity", "z")) == "SR"
    assert stride_for(Element("p", "process", "z")) == "STRIDE"
    assert stride_for(Element("s", "store", "z")) == "TID"
    assert stride_for(Element("s", "store", "z", logs=True)) == "TRID"
    assert stride_for(Element("f", "flow", src="a", dst="b")) == "TID"


def test_rag_model_counts_and_boundaries() -> None:
    model = rag_assistant()
    assert len(model.threats()) == 43
    crossing = {t.element for t in model.at_boundaries()}
    assert not {"query", "lookup", "record", "Retriever"} & crossing
    assert {"question", "prompt", "ingest"} <= crossing
    assert model.to_table(boundary_only=True).startswith("| Element")


def test_model_rejects_dangling_flow() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ThreatModel([Element("f", "flow", src="a", dst="b")])


def test_default_deny_and_groups() -> None:
    chunks = [
        AclChunk("a", "x", frozenset({"hr"})),
        AclChunk("b", "x", frozenset({"ann"})),
        AclChunk("c", "x"),
    ]
    ann = Principal("ann", frozenset({"hr"}))
    assert [c.id for c in authorized(ann, chunks)] == ["a", "b"]
    assert authorized(Principal("zed"), chunks) == []


def test_no_unauthorized_chunk_is_ever_returned(rng: random.Random) -> None:
    index = AclIndex(corpus(rng))
    for _ in range(200):
        user = Principal("u", frozenset(rng.sample(GROUPS, rng.randint(0, 3))))
        query = " ".join(rng.choices(WORDS, k=2))
        for fn in (retrieve_secure, retrieve_post_filter):
            for chunk in fn(query, user, index, 5):
                assert chunk.allowed & user.ids


def test_prefilter_fills_k_where_postfilter_starves(
    rng: random.Random,
) -> None:
    index = AclIndex(corpus(rng, 120))
    user = Principal("u", frozenset({"hr"}))
    pre = retrieve_secure("budget salary", user, index, 5)
    post = retrieve_post_filter("budget salary", user, index, 5)
    assert len(pre) == 5
    assert len(post) <= len(pre)


PEOPLE = {"ann": Principal("ann", frozenset({"hr"})), "bob": Principal("bob")}
LIBRARY = [
    AclChunk("p", "salary table SECRET-PAY", frozenset({"hr"})),
    AclChunk(
        "m", "salary policy <script>x()</script>", frozenset({"hr", "all"})
    ),
    AclChunk("q", "lunch menu for friday", frozenset({"all"})),
]
CASES = [
    AttackCase("pay", "idor", "bob", "show the salary table", ("SECRET-PAY",)),
    AttackCase("xss", "output", "bob", "salary policy", ("<script>",)),
]


def build(acl: bool, escape: bool) -> System:
    index = AclIndex(LIBRARY)
    model = ScriptedModel(default=lambda r: "Summary: " + r.system)
    fetch = retrieve_secure if acl else retrieve_unfiltered

    def system(user: str, prompt: str) -> str:
        people = dict(PEOPLE)
        people["bob"] = Principal("bob", frozenset({"all"}))
        found = fetch(prompt, people[user], index, 3)
        context = " | ".join(c.text for c in found)
        reply = model.complete(
            Request(context, ({"role": "user", "content": prompt},))
        )
        return render_safe(reply.text) if escape else reply.text

    return system


def evidence(acl: bool, escape: bool) -> list[Evidence]:
    return run_suite(CASES, build(acl, escape), run_id="r", recorded_at=1.0)


def test_suite_passes_with_both_controls() -> None:
    ev = evidence(True, True)
    assert [e.verdict for e in ev] == ["pass", "pass"]
    gate = release_gate(ev, required=[e.test_id for e in ev])
    assert gate.passed


@pytest.mark.parametrize(
    ("acl", "escape", "blocked"),
    [(False, True, "redteam.idor.pay"), (True, False, "redteam.output.xss")],
)
def test_removing_a_control_blocks_the_release(
    acl: bool, escape: bool, blocked: str
) -> None:
    ev = evidence(acl, escape)
    gate = release_gate(ev, required=[e.test_id for e in ev])
    assert not gate.passed
    assert gate.blocking == (blocked,)


def test_a_crashing_system_is_an_error_not_a_pass() -> None:
    def boom(user: str, prompt: str) -> str:
        raise RuntimeError("down")

    ev = run_suite(CASES, boom, run_id="r", recorded_at=1.0)
    assert {e.verdict for e in ev} == {"error"}


def test_case_needs_a_forbidden_string() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        AttackCase("x", "c", "u", "p", ())
