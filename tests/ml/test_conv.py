import numpy as np

from ai_engineer.ml.conv import (
    clip_global_norm,
    conv1d,
    conv1d_as_matrix,
    conv2d,
    conv_budget,
    conv_output_size,
    dense_budget,
    gru_step,
    lstm_step,
    max_pool2d,
    receptive_field,
    rnn_forward,
    sensitivity,
)


def test_conv1d_matches_the_obvious_loop(np_rng):
    x, k = np_rng.normal(size=20), np_rng.normal(size=3)
    expected = [x[i : i + 3] @ k for i in range(0, 18, 2)]
    assert np.allclose(conv1d(x, k, stride=2), expected)


def test_an_edge_kernel_fires_only_at_the_edge():
    image = np.zeros((6, 8))
    image[:, 4:] = 1.0  # dark left half, bright right half
    vertical_edge = np.array([[-1.0, 1.0]] * 2)
    response = conv2d(image, vertical_edge)
    assert response.shape == (5, 7)
    assert np.flatnonzero(response.any(axis=0)).tolist() == [3]


def test_max_pool_keeps_the_strongest_value_per_tile():
    image = np.arange(16.0).reshape(4, 4)
    assert max_pool2d(image).tolist() == [[5.0, 7.0], [13.0, 15.0]]


def test_one_kernel_replaces_a_dense_layer_of_weights():
    dense = (28 * 28) ** 2  # every output pixel sees every input pixel
    kernel = 3 * 3
    assert dense // kernel == 68_295


def test_rnn_state_depends_on_the_first_input(np_rng):
    d_in, d_h = 3, 4
    w_x = np_rng.normal(size=(d_in, d_h))
    w_h = np_rng.normal(size=(d_h, d_h)) * 0.3
    xs = np_rng.normal(size=(5, d_in))
    states = rnn_forward(xs, w_x, w_h, np.zeros(d_h))
    xs2 = xs.copy()
    xs2[0] += 1.0
    assert states.shape == (5, d_h)
    assert not np.allclose(
        states[-1], rnn_forward(xs2, w_x, w_h, 0 * w_h[0])[-1]
    )


def test_gradients_vanish_over_long_sequences(np_rng):
    d = 8
    w_x = np_rng.normal(size=(d, d)) / np.sqrt(d)
    w_h = np_rng.normal(size=(d, d)) * 0.5 / np.sqrt(d)
    states = rnn_forward(np_rng.normal(size=(40, d)), w_x, w_h, np.zeros(d))
    s = sensitivity(states, w_h)
    assert s[-1] == 1.0
    assert s[0] < 1e-6
    assert np.all(np.diff(s[-10:]) > 0)


def test_output_size_formula_and_the_shapes_conv2d_produces():
    for n, k, pad, stride in [(5, 3, 0, 1), (5, 3, 1, 1), (5, 3, 0, 2)]:
        out = conv2d(np.ones((n, n)), np.ones((k, k)), pad, stride)
        assert out.shape == (conv_output_size(n, k, pad, stride),) * 2
    assert conv_output_size(224, 7, pad=3, stride=2) == 112


def test_padding_keeps_the_size_and_zeros_the_border_sums():
    out = conv2d(np.ones((5, 5)), np.ones((3, 3)), pad=1)
    assert out.shape == (5, 5)
    assert [out[0, 0], out[0, 2], out[2, 2]] == [4.0, 6.0, 9.0]


def test_convolution_flips_the_kernel_and_libraries_do_not():
    x, k = np.array([0.0, 0, 5, 5, 5]), np.array([1.0, 0, -1])
    assert conv1d(x, k).tolist() == [-5.0, -5.0, 0.0]
    assert np.convolve(x, k, mode="valid").tolist() == [5.0, 5.0, 0.0]


def test_three_3x3_layers_see_7x7_and_stride_widens_the_view():
    assert receptive_field([(3, 1)] * 3) == 7
    assert receptive_field([(3, 1), (2, 2), (3, 1)]) == 8


def test_clipping_changes_the_length_not_the_direction():
    g = [np.array([3.0, 0.0]), np.array([0.0, 4.0])]
    clipped, norm = clip_global_norm(g, 1.0)
    assert norm == 5.0
    assert np.allclose(clipped[0], [0.6, 0.0])
    assert np.allclose(clipped[1], [0.0, 0.8])
    same, _ = clip_global_norm(g, 10.0)
    assert same[0].tolist() == [3.0, 0.0]


def test_lstm_cell_state_changes_by_addition(np_rng):
    d = 3
    w_x, w_h = np_rng.normal(size=(2, d * 4)), np_rng.normal(size=(d, d * 4))
    b = np.zeros(d * 4)
    b[d : 2 * d] = 30.0  # forget gate wide open
    b[:d] = -30.0  # input gate shut
    c = np_rng.normal(size=d)
    _, c_new = lstm_step(np_rng.normal(size=2), np.zeros(d), c, w_x, w_h, b)
    assert np.allclose(c_new, c)


def test_gru_update_gate_decides_how_much_state_to_keep(np_rng):
    d = 3
    w_x, w_h = np_rng.normal(size=(2, d * 3)), np_rng.normal(size=(d, d * 3))
    b = np.zeros(d * 3)
    h = np_rng.normal(size=d)
    b[:d] = 30.0  # update gate at 1: keep the old state
    assert np.allclose(gru_step(np.ones(2), h, w_x, w_h, b), h)
    b[:d] = -30.0  # update gate at 0: take the candidate
    assert not np.allclose(gru_step(np.ones(2), h, w_x, w_h, b), h)


def test_a_convolution_is_a_dense_layer_with_locality_and_sharing(np_rng):
    x, k = np_rng.normal(size=6), np.array([1.0, 2.0, 3.0])
    matrix = conv1d_as_matrix(k, 6)
    assert matrix.shape == (4, 6)  # 24 entries, 3 free numbers
    assert np.count_nonzero(matrix) == 12
    assert np.allclose(matrix @ x, conv1d(x, k))


def test_parameter_counts_quoted_in_the_chapter():
    assert 224 * 224 * 3 * 1000 + 1000 == 150_529_000
    assert 64 * (3 * 3 * 3 + 1) == 1_792
    assert (28 * 28) ** 2 == 614_656
    lenet5 = [6 * (25 + 1), 6 * 2, 60 * 25 + 16, 16 * 2]
    lenet5 += [120 * (16 * 25 + 1), 84 * (120 + 1)]
    assert sum(lenet5) == 60_000
    assert 3 * 3 * 3 * 64**2 == 110_592
    assert 7 * 7 * 64**2 == 200_704


def test_a_shifted_input_gives_a_shifted_output():
    x = np.array([0.0, 0, 3, 1, 2, 0, 0, 0, 0, 0])
    k = np.array([1.0, 0, -1])
    shift_then_conv = conv1d(np.roll(x, 2), k)
    conv_then_shift = np.roll(conv1d(x, k), 2)
    assert np.allclose(shift_then_conv, conv_then_shift)


def test_lenet5_budget_puts_the_weights_and_the_arithmetic_apart():
    c1 = conv_budget(out_size=28, k=5, pairs=6, maps_out=6)
    c3 = conv_budget(out_size=10, k=5, pairs=60, maps_out=16)
    c5 = conv_budget(out_size=1, k=5, pairs=16 * 120, maps_out=120)
    f6 = dense_budget(120, 84)
    assert (c1, c3, c5, f6) == (
        (156, 117_600),
        (1_516, 150_000),
        (48_120, 48_000),
        (10_164, 10_080),
    )
    pooling = 6 * 2 + 16 * 2  # a coefficient and a bias per map
    assert sum(b[0] for b in (c1, c3, c5, f6)) + pooling == 60_000
    assert sum(b[1] for b in (c1, c3, c5, f6)) == 325_680


def test_a_first_layer_is_cheap_in_weights_and_dear_in_arithmetic():
    params, macs = conv_budget(out_size=224, k=3, pairs=3 * 64, maps_out=64)
    assert params == 1_792
    assert macs == 86_704_128
    assert macs // 224**2 == 1_728  # every weight is used at 50,176 places


def test_lstm_step_reproduces_the_chapter_example():
    # gates f=0.9, i=0.5, o=0.7 and candidate g=0.8 come from biases alone
    b = np.array([0.0, np.log(9.0), np.arctanh(0.8), np.log(7 / 3)])
    zeros = np.zeros((1, 4))
    c0 = np.array([2.0])
    h, c = lstm_step(np.zeros(1), np.zeros(1), c0, zeros, zeros, b)
    assert np.isclose(c[0], 2.20)
    assert np.isclose(h[0], 0.7 * np.tanh(2.20))
    assert round(float(h[0]), 4) == 0.6830


def test_gru_step_reproduces_the_chapter_example():
    # update gate z=0.9 and candidate -0.4 come from biases alone
    b = np.array([np.log(9.0), 0.0, np.arctanh(-0.4)])
    zeros = np.zeros((1, 3))
    h = gru_step(np.zeros(1), np.array([0.8]), zeros, zeros, b)
    assert np.isclose(h[0], 0.68)
