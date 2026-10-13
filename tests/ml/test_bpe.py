import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai_engineer.ml.bpe import PIECES, BPETokenizer, bytes_per_token

CORPUS = (
    "The model reads the invoice. The model reads the ledger. "
    "Readers of invoices read ledgers, and ledgers record invoices. "
) * 20

TOKENIZER = BPETokenizer.train(CORPUS, vocab_size=300)


def test_the_most_frequent_pair_is_merged_first():
    tok = BPETokenizer.train("ab ab ab cd", vocab_size=258)
    assert tok.merges == ((ord("a"), ord("b")), (ord(" "), 256))


def test_training_stops_at_the_requested_vocabulary():
    assert len(TOKENIZER.merges) == 300 - 256
    assert BPETokenizer.train(CORPUS, vocab_size=300) == TOKENIZER


def test_merges_compress_text_like_the_training_data():
    text = "The model reads the ledger."
    assert len(TOKENIZER.encode(text)) < len(text.encode()) / 3


def test_unfamiliar_text_costs_more_tokens_per_character():
    familiar = "the ledger records invoices"
    unfamiliar = "zygote quark vexing jambs"
    rate = [len(TOKENIZER.encode(t)) / len(t) for t in (familiar, unfamiliar)]
    assert rate[1] > 2 * rate[0]


@given(st.text())
def test_any_text_round_trips(text):
    assert "".join(PIECES.findall(text)) == text
    assert TOKENIZER.decode(TOKENIZER.encode(text)) == text


def test_bytes_per_token_counts_bytes_not_characters():
    no_merges = BPETokenizer.train("", vocab_size=256)
    assert bytes_per_token(no_merges, "abc") == 1.0
    assert bytes_per_token(no_merges, "\u20ac") == 1.0  # 3 bytes, 3 tokens


def test_bytes_per_token_is_higher_for_familiar_text():
    familiar = bytes_per_token(TOKENIZER, "the ledger records invoices")
    unfamiliar = bytes_per_token(TOKENIZER, "zygote quark vexing jambs")
    assert familiar > 3 * unfamiliar > 0


def test_bytes_per_token_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        bytes_per_token(TOKENIZER, "")
