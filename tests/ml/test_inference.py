import numpy as np
import pytest

from ai_engineer.ml.inference import (
    Accelerator,
    Hop,
    batch_tradeoff,
    cost_per_million_tokens,
    kv_cache_bytes,
    littles_law_concurrency,
    mm1_wait_seconds,
    percentile,
    request_latency,
    simulate_path,
    speculative_tokens_per_step,
    weight_bytes,
)
from ai_engineer.ml.local import Machine, ModelShape, plan_local

# An 8B-parameter model shaped like common open-weight models: 32 layers,
# 8 key/value heads of width 128 (grouped-query attention).
EIGHT_B = ModelShape("8B-class", 8e9, layers=32, kv_heads=8, head_dim=128)


def test_kv_cache_costs_128_kib_per_token_for_an_8b_model():
    per_token = kv_cache_bytes(layers=32, kv_heads=8, head_dim=128, tokens=1)
    assert per_token == 128 * 1024
    full = kv_cache_bytes(layers=32, kv_heads=8, head_dim=128, tokens=8192)
    assert full == 2**30  # one GiB for one 8k-token conversation


def test_weights_shrink_with_bits():
    assert weight_bytes(8e9, 16) == 16e9
    assert weight_bytes(8e9, 4) == 4e9


def test_batching_buys_throughput_with_per_user_speed():
    gpu = Accelerator("example", 80e9, bandwidth=3.0e12, flops=8e14)
    points = batch_tradeoff(
        gpu,
        params=8e9,
        weights=16e9,
        kv_per_sequence=0.5e9,
        batches=[1, 8, 32, 128],
    )
    totals = [p.total_tps for p in points]
    per_user = [p.per_user_tps for p in points]
    assert totals == sorted(totals)
    assert per_user == sorted(per_user, reverse=True)
    assert totals[-1] > 20 * totals[0]


def test_tail_latencies_do_not_add():
    hops = [Hop("retrieve", 120, 300), Hop("model", 450, 1200)]
    one = simulate_path(hops[:1], seed=1)
    assert percentile(one, 50) == pytest.approx(120, rel=0.02)
    assert percentile(one, 95) == pytest.approx(300, rel=0.03)
    p95 = percentile(simulate_path(hops, seed=1), 95)
    assert 1200 < p95 < 1500


def test_littles_law_and_self_hosting_cost():
    assert littles_law_concurrency(50, 2.0) == 100
    cost = cost_per_million_tokens(2.0, tokens_per_s=1000, utilization=0.5)
    assert cost == pytest.approx(2.0 / 1.8)


def test_an_8b_model_at_4_bits_fits_a_16_gb_laptop():
    laptop = Machine("laptop", memory_gb=16, bandwidth_gb_s=100)
    plan = plan_local(EIGHT_B, laptop, bits=4.5, context=8192)
    assert plan.fits
    assert plan.weights_gb == pytest.approx(4.5)
    assert plan.max_tokens_per_s == pytest.approx(22.2, abs=0.1)


def test_a_70b_model_does_not_fit_in_32_gb():
    big = ModelShape("70B-class", 70e9, layers=80, kv_heads=8, head_dim=128)
    plan = plan_local(big, Machine("desktop", 32, 400), bits=4.5, context=8192)
    assert not plan.fits
    assert np.isclose(plan.weights_gb, 39.375)


# The request path of a retrieval-augmented answer, used in Chapter 24.
RAG_PATH = [
    Hop("gateway", 15, 40),
    Hop("embed query", 25, 60),
    Hop("vector search", 40, 110),
    Hop("rerank", 60, 150),
    Hop("assemble prompt", 15, 35),
    Hop("model: first token", 450, 1200),
]


def test_a_rag_request_spends_most_of_its_budget_in_the_model():
    total = simulate_path(RAG_PATH)
    before_model = simulate_path(RAG_PATH[:-1])
    assert sum(h.p95_ms for h in RAG_PATH) == 1595  # what a naive budget adds
    assert percentile(total, 95) == pytest.approx(1390, abs=10)
    assert percentile(before_model, 95) == pytest.approx(289, abs=5)


def test_request_latency_is_first_token_plus_the_rest() -> None:
    assert request_latency(0.4, 0.02, 101) == pytest.approx(2.4)
    assert request_latency(0.4, 0.02, 1) == pytest.approx(0.4)


def test_speculative_decoding_yields_3_36_tokens_at_alpha_0_8() -> None:
    assert speculative_tokens_per_step(0.8, 4) == pytest.approx(3.3616)
    assert speculative_tokens_per_step(0.0, 4) == pytest.approx(1.0)
    assert speculative_tokens_per_step(1.0, 4) == 5.0


def test_mm1_wait_explodes_near_full_utilization() -> None:
    waits = [mm1_wait_seconds(r, 1.0) for r in (0.5, 0.8, 0.95)]
    assert waits == pytest.approx([2.0, 5.0, 20.0])
    assert mm1_wait_seconds(1.0, 1.0) == float("inf")
