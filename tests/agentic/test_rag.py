from itertools import pairwise
from pathlib import Path

import pytest

from ai_engineer.rag.bm25 import BM25, tokenize
from ai_engineer.rag.chunk import chunk_document
from ai_engineer.rag.fusion import rrf
from ai_engineer.rag.retrieve import (
    HybridIndex,
    recall_at_k,
    reciprocal_rank,
)

HANDBOOK = Path(__file__).parents[2] / "data/synthetic/larkspur_handbook.md"


@pytest.fixture(scope="module")
def chunks():
    text = HANDBOOK.read_text(encoding="utf-8")
    return chunk_document(
        "handbook", "Larkspur handbook", text, size=60, overlap=10
    )


@pytest.fixture(scope="module")
def index(chunks):
    return HybridIndex(chunks)


def test_every_chunk_names_its_document_and_section(chunks):
    assert [c.context.split(" > ")[1] for c in chunks] == [
        "Travel and reimbursement",
        "Travel and reimbursement",
        "Expenses",
        "Expenses",
        "On-call",
        "On-call",
        "Data retention",
        "Security",
        "Leave",
        "Leave",
    ]
    assert chunks[2].indexed.startswith("Larkspur handbook > Expenses\n")


def test_neighboring_chunks_share_the_overlap():
    text = "# Notes\n" + " ".join(f"w{i}" for i in range(25))
    found = chunk_document("d", "T", text, size=10, overlap=3)
    pieces = [chunk.text.split() for chunk in found]
    assert [len(p) for p in pieces] == [10, 10, 10, 4]
    for before, after in pairwise(pieces):
        assert before[-3:] == after[:3]
    assert pieces[-1][-1] == "w24"


@pytest.mark.parametrize("overlap", [10, -1])
def test_the_overlap_must_be_smaller_than_a_chunk(overlap):
    with pytest.raises(ValueError, match="overlap"):
        chunk_document("d", "T", "some words", size=10, overlap=overlap)


def test_tokens_keep_codes_and_identifiers_whole():
    assert tokenize("Error E-4127, see cost_center!") == [
        "error",
        "e-4127",
        "see",
        "cost_center",
    ]


def test_bm25_puts_the_document_matching_more_terms_first():
    bm25 = BM25(["the cat sat", "cat and dog play", "dogs bark"])
    assert [i for i, _ in bm25.search("cat dog")] == [1, 0]


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([["a", "b", "c"], ["b", "c", "a"]])
    assert [item for item, _ in fused] == ["b", "a", "c"]


def test_an_error_code_needs_lexical_search(index):
    assert index.lexical("E-4127", 3) == ["handbook#2"]
    assert index.dense("E-4127", 3)[0] != "handbook#2"
    assert index.search("E-4127", 3)[0] == "handbook#2"


def test_a_typo_needs_dense_search(index):
    assert index.lexical("reimbursment", 3) == []
    assert index.dense("reimbursment", 3)[0] == "handbook#1"
    assert index.search("reimbursment", 3)[0] == "handbook#1"


def test_fusion_recovers_when_one_ranking_is_wrong(index):
    query = "how long do we keep logs"
    assert index.lexical(query, 3)[0] == "handbook#1"
    assert index.dense(query, 3)[0] == "handbook#6"
    assert index.search(query, 3)[0] == "handbook#6"


def test_a_synonym_defeats_both_without_a_real_embedding_model(index):
    # The security section says "two-factor"; nothing says "2FA".
    for method in (index.lexical, index.dense, index.search):
        assert "handbook#7" not in method("2FA", 3)


def test_recall_and_reciprocal_rank():
    retrieved = ["handbook#6", "handbook#1", "handbook#4"]
    assert recall_at_k(retrieved, {"handbook#6", "handbook#7"}, 3) == 0.5
    assert reciprocal_rank(retrieved, {"handbook#1"}) == 0.5
    assert reciprocal_rank(retrieved, {"handbook#9"}) == 0.0
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(retrieved, set(), 3)
