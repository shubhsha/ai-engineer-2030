import pytest

from ai_engineer.llm.context import (
    ContextOverflow,
    Item,
    assemble,
    compact_history,
)
from ai_engineer.llm.model import Reply, assistant, tool_result, tool_use, user


def words(text):
    return len(text.split())


ITEMS = [
    Item("system", "You answer questions about the handbook.", pinned=True),
    Item("history", "the user asked about travel " * 4, priority=1),
    Item("doc_a", "travel policy text " * 10, priority=3),
    Item("doc_b", "leave policy text " * 10, priority=2),
]


def test_items_are_admitted_by_priority_and_kept_in_order():
    out = assemble(ITEMS, budget=75, count=words)
    assert out.included == ("system", "doc_a", "doc_b")
    assert out.dropped == ("history",)
    assert out.tokens == 72
    assert out.text.startswith("<system>\n")


def test_pinned_items_that_do_not_fit_are_an_error():
    with pytest.raises(ContextOverflow, match="pinned items need 8"):
        assemble(ITEMS, budget=5, count=words)


def test_item_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        assemble([Item("a", "x"), Item("a", "y")], budget=100)


CONVERSATION = [
    user("Where is order 7?"),
    assistant(tool_use("t1", "lookup", id=7)),
    user([tool_result("t1", "shipped")]),
    assistant(Reply("Order 7 has shipped.")),
    user("And order 8?"),
    assistant(tool_use("t2", "lookup", id=8)),
    user([tool_result("t2", "pending")]),
]


def test_compaction_keeps_tool_calls_with_their_results():
    out = compact_history(
        CONVERSATION,
        keep_last=2,
        summarize=lambda old: f"{len(old)} earlier messages",
    )
    assert out[0] == user("<summary>\n4 earlier messages\n</summary>")
    assert out[1:] == CONVERSATION[4:]


def test_compaction_without_a_clean_cut_changes_nothing():
    out = compact_history(
        CONVERSATION[:3], keep_last=2, summarize=lambda old: "unused"
    )
    assert out == CONVERSATION[:3]


FIVE = [
    Item("system", "rules " * 6, pinned=True),
    Item("doc_a", "alpha " * 30, priority=3),
    Item("doc_b", "beta " * 30, priority=2),
    Item("history", "turn " * 20, priority=1),
    Item("faq", "q " * 10, priority=2),
]


def test_five_item_walkthrough_drops_history_and_faq():
    out = assemble(FIVE, budget=75, count=words)
    assert out.included == ("system", "doc_a", "doc_b")
    assert out.dropped == ("history", "faq")
    assert out.tokens == 72


def test_greedy_is_not_optimal_on_the_knapsack_counterexample():
    items = [
        Item("A", "w " * 4, priority=10),
        Item("B", "w " * 3, priority=6),
        Item("C", "w " * 3, priority=6),
    ]
    out = assemble(items, budget=10, count=words)
    assert out.included == ("A",)
    assert out.dropped == ("B", "C")
