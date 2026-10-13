from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from ai_engineer.foundations.proc import (
    NOT_EXECUTABLE,
    NOT_FOUND,
    TIMED_OUT,
    run_command,
)

ROOT = Path(__file__).resolve().parents[2]


def test_success_and_output() -> None:
    result = run_command([sys.executable, "-c", "print('hi')"])
    assert result.ok
    assert result.stdout == "hi\n"


def test_exit_code_is_preserved() -> None:
    result = run_command([sys.executable, "-c", "raise SystemExit(3)"])
    assert result.returncode == 3
    assert not result.ok


def test_missing_program_is_127() -> None:
    assert run_command(["no-such-program-xyz"]).returncode == NOT_FOUND


def test_timeout_is_124() -> None:
    # The child waits on an event that nobody sets: it never finishes.
    stuck = [
        sys.executable,
        "-c",
        "import threading; threading.Event().wait()",
    ]
    assert run_command(stuck, timeout=0.2).returncode == TIMED_OUT


def test_file_without_execute_permission_is_126(tmp_path: Path) -> None:
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o644)  # rw-r--r--: readable, not executable
    assert run_command([str(script)]).returncode == NOT_EXECUTABLE


def test_arguments_reach_the_program_unparsed() -> None:
    name = "notes.txt; echo INJECTED"
    echo = [sys.executable, "-c", "import sys; print(sys.argv[1])", name]
    assert run_command(echo).stdout == name + "\n"


def test_death_by_signal_is_negative() -> None:
    kill_self = "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
    assert run_command([sys.executable, "-c", kill_self]).returncode == -15


def test_child_leads_its_own_process_group() -> None:
    probe = "import os; print(os.getpgid(0) == os.getpid())"
    assert run_command([sys.executable, "-c", probe]).stdout == "True\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_triage_script_groups_error_signatures(tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    log.write_text(
        "2026-10-01T10:00:01 INFO started\n"
        "2026-10-01T10:00:02 ERROR timeout calling bw after 3000 ms\n"
        "2026-10-01T10:00:03 ERROR timeout calling bw after 4500 ms\n"
        "2026-10-01T10:00:04 ERROR row deadbeef01 failed check 7\n"
    )
    result = run_command(
        ["bash", str(ROOT / "scripts" / "triage.sh"), str(log), "5"]
    )
    assert result.ok, result.stderr
    lines = [ln.split(None, 1) for ln in result.stdout.splitlines()]
    assert lines[0] == ["2", "ERROR timeout calling bw after <n> ms"]
    assert lines[1] == ["1", "ERROR row <id> failed check <n>"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_triage_script_succeeds_when_there_are_no_errors(
    tmp_path: Path,
) -> None:
    log = tmp_path / "quiet.log"
    log.write_text("2026-10-01T10:00:01 INFO all good\n")
    result = run_command(
        ["bash", str(ROOT / "scripts" / "triage.sh"), str(log)]
    )
    assert result.ok, result.stderr
    assert result.stdout == ""
