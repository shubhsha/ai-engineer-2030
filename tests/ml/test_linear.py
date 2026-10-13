import numpy as np

from ai_engineer.ml.linear import (
    LogisticRegression,
    classification_scores,
    group_split,
)


def blobs(rng, n=400, gap=2.5):
    y = rng.integers(0, 2, n)
    x = rng.normal(size=(n, 2)) + gap * y[:, None]
    return x, y.astype(np.float64)


def test_logistic_regression_learns_separable_blobs(np_rng):
    x, y = blobs(np_rng)
    model = LogisticRegression()
    history = model.fit(x[:300], y[:300])
    assert history[-1] < history[0] / 2
    accuracy = np.mean(model.predict(x[300:]) == y[300:])
    assert accuracy > 0.9


def test_group_split_never_puts_one_group_on_both_sides():
    groups = [f"customer-{i % 17}" for i in range(200)]
    train, test = group_split(groups, test_fraction=0.25, seed=7)
    assert len(train) + len(test) == 200
    assert not {groups[i] for i in train} & {groups[i] for i in test}
    again = group_split(groups, test_fraction=0.25, seed=7)
    assert np.array_equal(test, again[1])


def test_accuracy_flatters_a_model_that_never_finds_anything():
    y_true = np.array([1] * 5 + [0] * 95)
    always_no = np.zeros(100, dtype=np.int64)
    scores = classification_scores(y_true, always_no)
    assert scores.accuracy == 0.95
    assert scores.recall == 0.0
    assert scores.f1 == 0.0
