import numpy as np
import pytest

from ai_engineer.ml.linear import LogisticRegression
from ai_engineer.ml.mlp import (
    Adam,
    forward,
    init_mlp,
    loss_and_grads,
    predict,
    train,
)
from ai_engineer.ml.numerics import numerical_gradient, relative_error


def rings(rng, n=400):
    """Class 0 inside radius 1, class 1 on a ring from 1.5 to 2.5."""
    y = rng.integers(0, 2, n)
    radius = np.where(y == 0, rng.uniform(0, 1, n), rng.uniform(1.5, 2.5, n))
    angle = rng.uniform(0, 2 * np.pi, n)
    x = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return x, y


@pytest.mark.parametrize("name", ["W0", "b0", "W1", "b1", "W2", "b2"])
def test_backprop_matches_numerical_gradient(np_rng, name):
    params = init_mlp([3, 5, 4, 3], np_rng)
    x = np_rng.normal(size=(6, 3))
    y = np_rng.integers(0, 3, 6)
    _, grads = loss_and_grads(params, x, y)

    def loss(_: np.ndarray) -> float:
        return loss_and_grads(params, x, y)[0]

    numeric = numerical_gradient(loss, params[name])
    assert relative_error(grads[name], numeric) < 1e-6


def test_hidden_layers_solve_what_a_linear_model_cannot(np_rng):
    x, y = rings(np_rng)
    linear = LogisticRegression().fit(x, y.astype(np.float64))
    assert linear[-1] > 0.6  # no straight line separates rings
    params = init_mlp([2, 16, 16, 2], np_rng)
    history = train(
        params, x, y, epochs=60, batch_size=32, opt=Adam(lr=0.02), rng=np_rng
    )
    assert history[-1] < 0.1
    assert np.mean(predict(params, x) == y) > 0.97


def test_training_is_reproducible_with_a_seed():
    def run() -> list[float]:
        rng = np.random.default_rng(5)
        x, y = rings(rng, 100)
        params = init_mlp([2, 8, 2], rng)
        return train(
            params, x, y, epochs=5, batch_size=16, opt=Adam(), rng=rng
        )

    assert run() == run()


def test_worked_example_in_the_chapter_is_exact():
    params = {
        "W0": np.array([[2.0, -1.0], [-1.0, 1.0]]),
        "b0": np.array([1.0, -2.0]),
        "W1": np.array([[1.0, -1.0], [2.0, 1.0]]),
        "b1": np.array([0.0, 0.5]),
    }
    x, y = np.array([[1.0, 2.0]]), np.array([1])
    assert np.allclose(forward(params, x)[0], [[1.0, -0.5]])
    loss, grads = loss_and_grads(params, x, y)
    assert loss == pytest.approx(1.7014, abs=1e-4)
    assert np.allclose(grads["W0"], [[1.6351, 0.0], [3.2703, 0.0]], atol=1e-4)
    assert np.allclose(grads["W1"], [[0.8176, -0.8176], [0.0, 0.0]], atol=1e-4)


def test_zero_init_leaves_only_the_last_bias_with_a_gradient(np_rng):
    x, y = rings(np_rng, 50)
    zeros = {
        k: np.zeros_like(v) for k, v in init_mlp([2, 8, 8, 2], np_rng).items()
    }
    _, grads = loss_and_grads(zeros, x, y)
    assert [k for k, g in grads.items() if np.any(g)] == ["b2"]


def test_he_init_keeps_the_activation_scale_over_twenty_layers(np_rng):
    params = init_mlp([256] * 21, np_rng)
    h = np_rng.normal(size=(100, 256))
    for i in range(20):
        h = np.maximum(h @ params[f"W{i}"] + params[f"b{i}"], 0.0)
    assert 0.3 < np.sqrt(np.mean(h**2)) < 3.0


def test_adam_first_step_has_the_size_of_the_learning_rate(np_rng):
    x, y = rings(np_rng)
    params = init_mlp([2, 16, 16, 2], np_rng)
    _, grads = loss_and_grads(params, x, y)
    before = {k: v.copy() for k, v in params.items()}
    Adam(lr=0.01).step(params, grads)
    for k, g in grads.items():
        moved = np.abs(params[k] - before[k])[np.abs(g) > 1e-5]
        assert np.allclose(moved, 0.01, rtol=1e-2)


def test_adam_ignores_the_scale_of_the_loss(np_rng):
    x, y = rings(np_rng)
    base = init_mlp([2, 16, 16, 2], np_rng)
    _, grads = loss_and_grads(base, x, y)
    small = {k: v.copy() for k, v in base.items()}
    large = {k: v.copy() for k, v in base.items()}
    Adam(lr=0.01).step(small, grads)
    Adam(lr=0.01).step(large, {k: 1000.0 * g for k, g in grads.items()})
    for k, g in grads.items():
        keep = np.abs(g) > 1e-5
        assert np.allclose(small[k][keep], large[k][keep], atol=2e-5)


def test_a_tiny_set_is_memorized_to_near_zero_loss():
    x, y = rings(np.random.default_rng(2030), 400)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        net = init_mlp([2, 16, 16, 2], rng)
        opt = Adam(lr=0.01)
        history = train(
            net, x[:8], y[:8], epochs=300, batch_size=8, opt=opt, rng=rng
        )
        assert min(history) < 1e-3
