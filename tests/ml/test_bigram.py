import math

import numpy as np
import pytest

from ai_engineer.ml.bigram import BigramModel
from ai_engineer.ml.numerics import softmax

CYCLE = "the cat sat. " * 20


def test_counts_of_a_known_string_are_exact():
    model = BigramModel.fit("abab", smoothing=0.0)
    a, b = model.chars.index("a"), model.chars.index("b")
    assert model.counts[a, b] == 2
    assert model.counts[b, a] == 1
    assert model.counts[a, a] == 0


def test_every_row_is_a_probability_distribution():
    model = BigramModel.fit(CYCLE)
    for ch in model.chars:
        assert math.isclose(softmax(model.logits(ch)).sum(), 1.0)


def test_smoothing_moves_mass_to_unseen_pairs():
    sharp = BigramModel.fit(CYCLE, smoothing=0.01)
    soft = BigramModel.fit(CYCLE, smoothing=1.0)
    unseen = "zq"  # never occurs; both models need a pair to score
    text = CYCLE + unseen
    with pytest.raises(ValueError, match="substring"):
        sharp.nll(text)
    t, h = "t", "h"
    assert (
        soft.logits(t)[soft.chars.index("c")]
        > sharp.logits(t)[sharp.chars.index("c")]
    )
    assert (
        soft.logits(t)[soft.chars.index(h)]
        < sharp.logits(t)[sharp.chars.index(h)]
    )


def test_greedy_generation_falls_into_a_loop():
    model = BigramModel.fit(CYCLE)
    out = model.generate("t", 24, np.random.default_rng(0), temperature=0)
    assert "cat cat cat" in out
