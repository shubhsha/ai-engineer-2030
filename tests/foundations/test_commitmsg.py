from pathlib import Path

import pytest

from ai_engineer.foundations.commitmsg import check_commit_message, main


@pytest.mark.parametrize(
    "message",
    [
        "feat(rag): add reciprocal rank fusion",
        "fix: handle empty tool results\n\nThe loop crashed on [].",
        "refactor(sql-gate)!: reject multi-statement input",
    ],
)
def test_good_messages_pass(message):
    assert check_commit_message(message) == []


@pytest.mark.parametrize(
    ("message", "fragment"),
    [
        ("", "empty"),
        ("Fixed stuff", "type(scope): subject"),
        ("feature: add x", "unknown type"),
        ("fix: add x.", "period"),
        ("fix: " + "x" * 80, "limit is 72"),
        ("fix: add x\nbody right away", "blank line"),
    ],
)
def test_bad_messages_explain_themselves(message, fragment):
    problems = check_commit_message(message)
    assert any(fragment in p for p in problems), problems


def test_comment_lines_are_ignored():
    assert check_commit_message("# Please enter...\nfix: x") == []


def test_main_reads_the_file_git_passes(tmp_path: Path):
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text("wip\n")
    assert main([str(msg)]) == 1
    msg.write_text("docs: explain the hook\n")
    assert main([str(msg)]) == 0
