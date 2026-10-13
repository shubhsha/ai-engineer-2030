import pytest

from ai_engineer.agents.hooks import (
    Decision,
    ask_before_writes,
    block_destructive,
    decision_counts,
    guarded,
    recursive_force_rm,
    redact_results,
)
from ai_engineer.agents.loop import run_agent
from ai_engineer.llm.model import (
    ScriptedModel,
    ToolCall,
    tool_result,
    tool_use,
)
from ai_engineer.llm.tools import Tool, ToolRegistry


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "rm -fr build",
        "rm -r -f build",
        "rm -Rf /",
        "rm --recursive --force build",
        "sudo rm -rf /var/lib/app",
        "make clean && rm -rf dist",
        "cd /tmp; /bin/rm -rf cache",
        "FOO=1 rm -rf build",
        "find . -name '*.pyc' | xargs rm -rf",
    ],
)
def test_recursive_forced_deletes_are_caught(command):
    assert recursive_force_rm(command)


@pytest.mark.parametrize(
    "command",
    [
        "rm notes.txt",
        "rm -r build",
        "rm -f build/app.log",
        "echo 'rm -rf is dangerous'",
        "grep -rf patterns.txt src",
        "git rm -rf --cached build",
    ],
)
def test_ordinary_commands_pass(command):
    assert not recursive_force_rm(command)


def shell(command):
    return ToolCall("t1", "shell", {"command": command})


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("DROP TABLE orders;", "drops a table"),
        ("drop schema staging cascade", "drops a table"),
        ("TRUNCATE orders", "empties a table"),
        ("DELETE FROM orders;", "DELETE without WHERE"),
        ("DELETE FROM orders", "DELETE without WHERE"),
        ("git push --force origin main", "force-pushes"),
        ("git push origin +main", "force-pushes"),
    ],
)
def test_destructive_commands_are_denied(command, reason):
    decision = block_destructive(shell(command))
    assert decision is not None
    assert decision.action == "deny"
    assert reason in decision.reason


@pytest.mark.parametrize(
    "command",
    [
        "DELETE FROM orders WHERE id = 7;",
        "DELETE FROM orders\nWHERE status = 'test'",
        "SELECT * FROM dropped_tables",
        "git push origin main",
    ],
)
def test_safe_commands_are_not_flagged(command):
    assert block_destructive(shell(command)) is None


def test_commands_nested_anywhere_in_the_input_are_found():
    steps = {"steps": [{"cmd": "ls"}, {"cmd": "rm -rf /"}]}
    decision = block_destructive(ToolCall("t1", "runbook", steps))
    assert decision == Decision("deny", "recursive forced delete (rm -rf)")


PATH = {"type": "object", "properties": {"path": {"type": "string"}}}
COMMAND = {"type": "object", "properties": {"command": {"type": "string"}}}


def tools_that_record(ran):
    def run_shell(command):
        ran.append(command)
        return f"ran: {command}"

    return ToolRegistry(
        [
            Tool("read_file", "Read a file.", PATH, lambda path: "contents"),
            Tool("shell", "Run a command.", COMMAND, run_shell, False),
        ]
    )


def yes(call, reason):
    return True


def test_only_tools_that_change_state_need_a_person():
    hook = ask_before_writes(tools_that_record([]))
    assert hook(ToolCall("t1", "read_file", {"path": "a.txt"})) is None
    assert hook(shell("ls")) == Decision("ask", "changes state")
    assert hook(ToolCall("t2", "unknown", {})) is None


def test_deny_beats_ask_and_nothing_runs():
    ran = []
    tools = tools_that_record(ran)
    pre = [ask_before_writes(tools), block_destructive]
    result = guarded(tools.call, pre=pre, approve=yes)(shell("rm -rf /"))
    assert result["is_error"] is True
    assert result["content"] == (
        "Blocked by policy: recursive forced delete (rm -rf). Do not retry."
    )
    assert ran == []


def test_ask_runs_the_call_only_with_approval():
    ran = []
    tools = tools_that_record(ran)
    pre = [ask_before_writes(tools)]
    declined = guarded(tools.call, pre=pre)(shell("ls"))
    assert declined["content"] == (
        "A person declined this action (changes state)."
    )
    assert ran == []
    approved = guarded(tools.call, pre=pre, approve=yes)(shell("ls"))
    assert approved["content"] == "ran: ls"
    assert ran == ["ls"]


def test_post_hooks_redact_what_the_model_would_see():
    call = ToolCall("t1", "read_file", {"path": "crm.csv"})
    leaked = tool_result("t1", "Contact: ana@example.com")
    assert redact_results(call, leaked)["content"] == "Contact: [EMAIL]"
    clean = tool_result("t1", "nothing to hide")
    assert redact_results(call, clean) is clean


def test_an_agent_under_policy_cleans_up_safely():
    ran, audit = [], []
    tools = tools_that_record(ran)
    execute = guarded(
        tools.call,
        pre=[block_destructive, ask_before_writes(tools)],
        post=[redact_results],
        approve=lambda call, reason: call.input["command"].startswith("rm "),
        audit=audit,
    )
    model = ScriptedModel(
        [
            tool_use("t1", "shell", command="rm -rf build"),
            tool_use("t2", "shell", command="rm build/app.log"),
            "Removed the old log file and left the rest of build.",
        ]
    )
    result = run_agent(model, "Free some disk space.", tools, execute=execute)
    assert result.stop == "end_turn"
    assert ran == ["rm build/app.log"]
    assert [entry["decision"] for entry in audit] == ["deny", "ask"]
    assert [entry["approved"] for entry in audit] == [False, True]
    [blocked] = result.messages[2]["content"]
    assert blocked["content"].startswith("Blocked by policy")


def test_decision_counts_summarize_an_audit_trail():
    ran, audit = [], []
    tools = tools_that_record(ran)
    execute = guarded(
        tools.call,
        pre=[block_destructive, ask_before_writes(tools)],
        approve=lambda call, reason: call.input["command"] == "ls",
        audit=audit,
    )
    for command in ("ls", "touch a", "rm -rf /"):
        execute(shell(command))
    execute(ToolCall("t2", "read_file", {"path": "a.txt"}))
    assert decision_counts(audit) == {
        "allow": 1,
        "ask": 2,
        "deny": 1,
        "approved": 1,
    }
