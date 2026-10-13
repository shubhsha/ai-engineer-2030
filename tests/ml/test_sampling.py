import math

import numpy as np

from ai_engineer.ml.bigram import BigramModel
from ai_engineer.ml.sampling import (
    entropy_bits,
    perplexity,
    sample,
    top_k_filter,
    top_p_filter,
)

LOGITS = np.log(np.array([0.5, 0.3, 0.15, 0.05]))


def kept(z: np.ndarray) -> list[int]:
    return np.flatnonzero(np.isfinite(z)).tolist()


def test_greedy_always_takes_the_top_token(np_rng):
    assert {sample(LOGITS, np_rng, temperature=0) for _ in range(20)} == {0}


def test_top_k_and_top_p_cut_the_tail():
    assert kept(top_k_filter(LOGITS, 2)) == [0, 1]
    assert kept(top_p_filter(LOGITS, 0.3)) == [0]
    assert kept(top_p_filter(LOGITS, 0.75)) == [0, 1]
    assert kept(top_p_filter(LOGITS, 0.85)) == [0, 1, 2]


def test_sampling_follows_the_distribution(np_rng):
    draws = [sample(LOGITS, np_rng) for _ in range(20_000)]
    freq = np.bincount(draws, minlength=4) / len(draws)
    assert np.allclose(freq, [0.5, 0.3, 0.15, 0.05], atol=0.015)


def test_temperature_trades_certainty_for_variety():
    cold, normal, hot = (entropy_bits(LOGITS / t) for t in (0.5, 1.0, 2.0))
    assert cold < normal < hot <= 2.0  # 4 tokens: at most 2 bits


def test_perplexity_is_the_effective_number_of_choices():
    assert math.isclose(perplexity(math.log(4)), 4.0)


TEXT = "the cat sat on the mat. the rat sat on the cat. " * 30


def test_a_bigram_model_prefers_text_like_its_training_data():
    model = BigramModel.fit(TEXT)
    scrambled = "".join(sorted(TEXT[:200]))
    assert model.nll(TEXT[:200]) < model.nll(scrambled)


def test_generation_is_reproducible_and_uses_known_characters():
    model = BigramModel.fit(TEXT)
    a = model.generate("t", 40, np.random.default_rng(3), temperature=0.7)
    b = model.generate("t", 40, np.random.default_rng(3), temperature=0.7)
    assert a == b
    assert set(a) <= set(model.chars)
