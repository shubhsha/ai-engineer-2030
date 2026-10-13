import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ai_engineer.ml.numerics import (
    cosine_similarity,
    cross_entropy,
    logsumexp,
    numerical_gradient,
    relative_error,
    softmax,
)

finite = st.floats(-50, 50)


@given(arrays(np.float64, 6, elements=finite), finite)
def test_softmax_is_a_distribution_and_ignores_shifts(x, c):
    p = softmax(x)
    assert math.isclose(p.sum(), 1.0)
    assert np.all(p >= 0)
    assert np.allclose(p, softmax(x + c))


def test_large_scores_do_not_overflow():
    with np.errstate(over="raise"):
        p = softmax(np.array([1000.0, 1001.0, 1002.0]))
        lse = logsumexp(np.array([1000.0, 1000.0]))
    assert np.allclose(p, softmax(np.array([0.0, 1.0, 2.0])))
    assert math.isclose(float(lse), 1000 + math.log(2))


def test_uniform_scores_cost_log_of_the_class_count():
    logits = np.zeros((4, 10))
    assert math.isclose(cross_entropy(logits, np.arange(4)), math.log(10))


def test_numerical_gradient_matches_calculus(np_rng):
    x = np_rng.normal(size=(3, 4))
    grad = numerical_gradient(lambda v: float(np.sum(v**3)), x)
    assert relative_error(grad, 3 * x**2) < 1e-8


def test_cosine_similarity_measures_angle_not_length():
    a = np.array([[1.0, 0.0], [3.0, 3.0]])
    b = np.array([[5.0, 0.0], [0.0, 2.0]])
    assert np.allclose(cosine_similarity(a, b), [[1, 0], [0.5**0.5] * 2])
