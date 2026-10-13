import numpy as np
import pytest

from ai_engineer.ml.quant import (
    quantize,
    quantize_asymmetric,
    quantize_groups,
    relative_rmse,
    snr_db,
    to_bfloat16,
)


def test_int8_keeps_gaussian_weights_within_one_percent(np_rng):
    w = np_rng.normal(size=(64, 64))
    q = quantize(w, bits=8)
    assert q.codes.dtype == np.int8
    assert relative_rmse(w, q.dequantize()) < 0.01


def test_per_channel_scales_survive_an_outlier_channel(np_rng):
    w = np_rng.normal(size=(64, 32))
    w[:, 0] *= 50  # one output channel with a much larger range
    per_tensor = quantize(w, bits=8)
    per_channel = quantize(w, bits=8, shared=(0,))
    quiet = slice(1, None)
    err_tensor = relative_rmse(w[:, quiet], per_tensor.dequantize()[:, quiet])
    err_channel = relative_rmse(
        w[:, quiet], per_channel.dequantize()[:, quiet]
    )
    assert err_channel < err_tensor / 10


def test_small_groups_rescue_four_bit_weights(np_rng):
    w = np_rng.standard_t(df=3, size=(64, 128))  # heavy tails, like LLMs
    coarse = relative_rmse(w, quantize(w, bits=4).dequantize())
    grouped = relative_rmse(w, quantize_groups(w, bits=4).dequantize())
    assert grouped < coarse * 0.7


def test_group_scales_cost_half_a_bit_per_weight(np_rng):
    q = quantize_groups(np_rng.normal(size=(64, 128)), bits=4, group=32)
    assert q.nbytes * 8 / (64 * 128) == 4.5


def test_groups_must_divide_the_row():
    with pytest.raises(ValueError, match="groups of 32"):
        quantize_groups(np.ones((2, 40)))


def test_bfloat16_keeps_range_but_drops_precision():
    x = np.array([1.0, 1 + 2**-8, 1 + 3 * 2**-8, 3.0e38, 0.1])
    y = to_bfloat16(x)
    assert y[0] == 1.0
    assert y[1] == 1.0  # halfway: ties go to the even neighbor
    assert y[2] == 1 + 2**-6
    assert np.isfinite(y[3])  # float16 would overflow here
    assert abs(y[4] - 0.1) / 0.1 < 2**-8


def test_asymmetric_zero_point_uses_all_levels_for_positive_values():
    w = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    codes, scale, zero = quantize_asymmetric(w, bits=8)
    assert zero == 0
    assert codes.max() == 255
    assert np.allclose((codes.astype(float) - zero) * scale, w, atol=scale)


def test_snr_gains_six_decibels_per_bit(np_rng):
    w = np_rng.normal(size=(128, 128))
    low = snr_db(w, quantize(w, 6).dequantize())
    high = snr_db(w, quantize(w, 7).dequantize())
    assert 5.0 < high - low < 7.0
