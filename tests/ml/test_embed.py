import numpy as np
import pytest

from ai_engineer.ml.embed import (
    dequantize_int8,
    embed_all,
    hashed_embedding,
    quantize_int8,
    recall_at_k,
    top_k,
)


def test_embeddings_are_deterministic_unit_vectors():
    a, b = (
        hashed_embedding("Invoice total"),
        hashed_embedding("invoice  total"),
    )
    assert np.array_equal(a, b)  # case and spacing are normalized
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_similar_spellings_rank_first():
    docs = ["weather report", "invoice totals by region", "team offsite"]
    best = top_k(hashed_embedding("invoice total"), embed_all(docs), k=2)
    assert best.tolist()[0] == 1


def test_top_k_handles_k_larger_than_the_collection():
    m = embed_all(["a", "b"])
    assert len(top_k(m[0], m, k=5)) == 2


def test_recall_at_k_counts_relevant_rows_in_the_first_k():
    assert recall_at_k([3, 1, 2], {1, 2}, k=2) == 0.5
    assert recall_at_k([3, 1, 2], {1, 2}, k=3) == 1.0


def test_recall_at_k_needs_something_to_recall():
    with pytest.raises(ValueError, match="no relevant"):
        recall_at_k([1], set(), k=1)


def test_int8_codes_are_one_byte_per_number_and_close():
    m = embed_all(
        [f"invoice {i} for customer {i * 7 % 50}" for i in range(40)]
    )
    codes, scale = quantize_int8(m)
    assert codes.dtype == np.int8
    assert codes.nbytes * 8 == m.nbytes  # float64 is eight bytes
    error = np.abs(dequantize_int8(codes, scale) - m)
    assert error.max() <= scale.max() / 2 + 1e-12  # rounding to the grid


def test_an_all_zero_row_survives_quantization():
    codes, scale = quantize_int8(np.zeros((1, 4)))
    assert codes.tolist() == [[0, 0, 0, 0]]
    assert scale.tolist() == [[1.0]]


def test_int8_search_finds_most_of_the_exact_neighbours():
    docs = [
        f"invoice {i} for customer {i * 7 % 50} in region {i % 9}"
        for i in range(200)
    ]
    exact_matrix = embed_all(docs)
    codes, scale = quantize_int8(exact_matrix)
    small = dequantize_int8(codes, scale)
    recalls = [
        recall_at_k(
            top_k(small[i], small, 10).tolist(),
            top_k(exact_matrix[i], exact_matrix, 10).tolist(),
            k=10,
        )
        for i in range(0, 200, 5)
    ]
    assert sum(recalls) / len(recalls) >= 0.9
