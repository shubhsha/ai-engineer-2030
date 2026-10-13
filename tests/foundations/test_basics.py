import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai_engineer.foundations.basics import (
    add_tag,
    add_tag_buggy,
    parse_pairs,
    top_words,
)


def test_parse_pairs_strips_and_skips_empty_fragments():
    assert parse_pairs(" user=ada ;region = eu;; ") == {
        "user": "ada",
        "region": "eu",
    }


def test_parse_pairs_names_the_bad_fragment():
    with pytest.raises(ValueError, match="'oops'"):
        parse_pairs("user=ada; oops")


keys = st.text("abcdefgh", min_size=1, max_size=6)
values = st.text("xyz0123", max_size=6)


@given(st.dictionaries(keys, values, max_size=5))
def test_parse_pairs_round_trips(pairs):
    line = "; ".join(f"{k}={v}" for k, v in pairs.items())
    assert parse_pairs(line) == pairs


def test_top_words_casefolds_and_strips_punctuation():
    text = "The cache. the CACHE! a miss."
    assert top_words(text, 2) == [("the", 2), ("cache", 2)]


def test_mutable_default_is_shared_between_calls():
    assert add_tag_buggy("a") == ["a"]
    assert add_tag_buggy("b") == ["a", "b"]  # the trap, demonstrated


def test_add_tag_never_mutates_its_input():
    tags = ["x"]
    assert add_tag("y", tags) == ["x", "y"]
    assert tags == ["x"]
    assert add_tag("a") == ["a"]
    assert add_tag("b") == ["b"]
