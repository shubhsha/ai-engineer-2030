import numpy as np
import pytest

from ai_engineer.ml.attention import (
    KVCache,
    MultiHeadAttention,
    attention,
    blockwise_attention,
    causal_mask,
    grouped_query_attention,
    kv_cache_bytes,
    padding_mask,
    sliding_window_mask,
)
from ai_engineer.ml.numerics import softmax


def test_each_output_is_a_weighted_average_of_the_values(np_rng):
    q, k = np_rng.normal(size=(2, 5, 4))
    weights = attention(q, k, np.eye(5))  # identity values expose weights
    assert np.allclose(weights.sum(axis=-1), 1.0)
    assert np.all(weights >= 0)


def test_dividing_by_root_d_keeps_the_softmax_soft(np_rng):
    d = 512
    q = np_rng.normal(size=(2000, 1, d))
    k = np_rng.normal(size=(2000, 8, d))
    scores = (q @ np.swapaxes(k, -1, -2))[:, 0]  # 2,000 queries x 8 keys
    assert 20 < scores.std() < 25  # the spread grows like sqrt(d) = 22.6
    peak = softmax(scores).max(axis=-1)
    peak_scaled = softmax(scores / np.sqrt(d)).max(axis=-1)
    assert np.median(peak) > 0.99  # one key takes nearly all the weight
    assert np.median(peak_scaled) < 0.5


def test_the_causal_mask_hides_the_future(np_rng):
    q, k, v = np_rng.normal(size=(3, 6, 4))
    out = attention(q, k, v, causal_mask(6))
    v2 = v.copy()
    v2[4:] += 10.0  # change only the last two positions
    out2 = attention(q, k, v2, causal_mask(6))
    assert np.allclose(out[:4], out2[:4])
    assert not np.allclose(out[4:], out2[4:])


def test_kv_cache_decoding_matches_full_recomputation(np_rng):
    mha = MultiHeadAttention(d_model=16, heads=4, rng=np_rng)
    x = np_rng.normal(size=(7, 16))
    full = mha.forward(x)
    cache = KVCache.empty(heads=4, head_dim=4)
    steps = np.stack([mha.step(x_t, cache) for x_t in x])
    assert len(cache) == 7
    assert np.allclose(full, steps, atol=1e-12)


def test_heads_must_divide_the_model_width(np_rng):
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=10, heads=4, rng=np_rng)


def test_padding_keys_do_not_change_real_outputs(np_rng):
    q, k, v = np_rng.normal(size=(3, 2, 5, 4))  # 2 sequences, 5 slots
    mask = padding_mask([5, 3], 5)  # the second has 2 padding slots
    clean = attention(q, k, v, mask)
    k2, v2 = k.copy(), v.copy()
    k2[1, 3:], v2[1, 3:] = 99.0, -99.0  # garbage in the padding slots
    assert np.allclose(attention(q, k2, v2, mask)[1, :3], clean[1, :3])
    unmasked = attention(q, k2, v2)  # without the mask they leak in
    assert not np.allclose(unmasked[1, :3], clean[1, :3])


def test_blockwise_attention_matches_full_attention(np_rng):
    q, k, v = np_rng.normal(size=(3, 2, 37, 8))  # 2 heads, 37 tokens
    causal = causal_mask(37)
    for block in (1, 5, 16, 37, 64):
        got = blockwise_attention(q, k, v, block, causal)
        assert np.allclose(got, attention(q, k, v, causal))
        got = blockwise_attention(q, k, v, block)
        assert np.allclose(got, attention(q, k, v))


def test_sliding_window_mask_keeps_the_last_w_keys():
    mask = sliding_window_mask(6, 3)
    assert mask.sum(axis=1).tolist() == [1, 2, 3, 3, 3, 3]
    assert mask[5].tolist() == [False, False, False, True, True, True]
    assert not (mask & ~causal_mask(6)).any()  # never looks ahead


def test_grouped_query_heads_share_key_value_heads(np_rng):
    q = np_rng.normal(size=(8, 5, 4))
    k, v = np_rng.normal(size=(2, 2, 5, 4))  # two key/value heads
    out = grouped_query_attention(q, k, v, causal_mask(5))
    for h in range(8):  # heads 0-3 read group 0, heads 4-7 read group 1
        want = attention(q[h], k[h // 4], v[h // 4], causal_mask(5))
        assert np.allclose(out[h], want)
    with pytest.raises(ValueError, match="do not split"):
        grouped_query_attention(q, k[:1].repeat(3, axis=0), v[:1])


def test_kv_cache_bytes_matches_the_published_sizes():
    kib = 1024
    assert kv_cache_bytes(32, 32, 128, 1) == 512 * kib  # 7B, 32 heads
    assert kv_cache_bytes(32, 32, 128, 8192) == 4 * 2**30  # 4 GiB
    assert kv_cache_bytes(32, 8, 128, 1) == 128 * kib  # 8 key/value heads
    assert kv_cache_bytes(40, 40, 128, 1) == 800 * kib  # OPT-13B, vLLM paper
