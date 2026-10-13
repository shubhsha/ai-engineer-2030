import numpy as np

from ai_engineer.ml.attention import attention
from ai_engineer.ml.transformer import (
    TinyDecoder,
    alibi_bias,
    alibi_slopes,
    apply_rope,
    layer_norm,
    rms_norm,
    sinusoidal_positions,
    swiglu,
)


def tiny() -> TinyDecoder:
    return TinyDecoder.create(vocab=50, d_model=32, heads=4, layers=2, seed=1)


def test_decoder_is_causal():
    model = tiny()
    tokens = np.array([3, 14, 15, 9, 26, 5])
    changed = tokens.copy()
    changed[4] = 42
    a, b = model.logits(tokens), model.logits(changed)
    assert a.shape == (6, 50)
    assert np.allclose(a[:4], b[:4])
    assert not np.allclose(a[4:], b[4:])


def test_parameter_count_matches_the_formula():
    v, d, layers = 50, 32, 2
    per_block = 12 * d * d + 2 * d  # 4d^2 attention, 8d^2 MLP, two norms
    assert tiny().parameter_count() == layers * per_block + v * d + d


def test_rope_makes_scores_depend_on_relative_position(np_rng):
    q, k = np_rng.normal(size=(2, 1, 64))

    def score(m: int, n: int) -> float:
        qm = apply_rope(q, np.array([m]))
        kn = apply_rope(k, np.array([n]))
        return float(qm[0] @ kn[0])

    assert np.isclose(score(7, 3), score(107, 103))
    assert not np.isclose(score(7, 3), score(7, 5))


def test_rope_rotates_without_changing_length(np_rng):
    x = np_rng.normal(size=(5, 8))
    rotated = apply_rope(x, np.arange(5))
    assert np.allclose(
        np.linalg.norm(rotated, axis=1), np.linalg.norm(x, axis=1)
    )


def test_rms_norm_and_positions_stay_in_range(np_rng):
    y = rms_norm(np_rng.normal(size=(4, 16)) * 50, np.ones(16))
    assert np.allclose(np.sqrt(np.mean(y**2, axis=-1)), 1.0, atol=1e-6)
    pe = sinusoidal_positions(128, 16)
    assert pe.shape == (128, 16)
    assert np.abs(pe).max() <= 1.0


def test_layer_norm_ignores_scale_and_shift(np_rng):
    x = np_rng.normal(size=(6, 32))
    g, b = np.ones(32), np.zeros(32)
    y = layer_norm(x, g, b)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-9)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-4)
    assert np.allclose(layer_norm(5 * x + 3, g, b), y, atol=1e-4)


def test_rms_norm_is_layer_norm_when_the_mean_is_zero(np_rng):
    x = np_rng.normal(size=(4, 16))
    x -= x.mean(axis=-1, keepdims=True)
    g, b = np.ones(16), np.zeros(16)
    assert np.allclose(rms_norm(x, g), layer_norm(x, g, b), atol=1e-4)


def test_swiglu_at_eight_thirds_width_has_the_parameters_of_a_4d_mlp():
    d, d_ff = 48, 128  # 3 * 48 * 128 == 2 * 48 * 192 == 8 * 48**2
    assert 3 * d * d_ff == 2 * d * (4 * d) == 8 * d * d
    rng = np.random.default_rng(0)
    w = [rng.normal(size=s) for s in [(d, d_ff), (d, d_ff), (d_ff, d)]]
    assert swiglu(rng.normal(size=(5, d)), *w).shape == (5, d)


def test_attention_without_positions_cannot_see_order(np_rng):
    x = np_rng.normal(size=(5, 8))
    order = np.array([3, 0, 4, 1, 2])
    assert np.allclose(
        attention(x, x, x)[order], attention(x[order], x[order], x[order])
    )
    pe = sinusoidal_positions(5, 8)
    shuffled = attention(x[order] + pe, x[order] + pe, x[order] + pe)
    assert not np.allclose(attention(x + pe, x + pe, x + pe)[order], shuffled)


def test_alibi_slopes_and_bias_follow_the_paper():
    assert np.allclose(alibi_slopes(8), 0.5 ** np.arange(1, 9))
    bias = alibi_bias(4, alibi_slopes(8))
    assert bias.shape == (8, 4, 4)
    assert np.allclose(bias[0, 3], [-1.5, -1.0, -0.5, 0.0])
