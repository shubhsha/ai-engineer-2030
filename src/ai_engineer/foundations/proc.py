"""Chapter 6: run other programs without a shell, and believe exit codes.

POSIX only: the child gets its own process group, so that a timeout can
stop everything it started.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

TIMED_OUT = 124  # the convention used by GNU timeout(1)
NOT_EXECUTABLE = 126  # found, but not executable: POSIX shells
NOT_FOUND = 127  # the convention used by POSIX shells


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    args: Sequence[str], *, timeout: float = 30.0, cwd: str | None = None
) -> CommandResult:
    """Run ``args`` directly (no shell) and capture text output.

    The child leads a new session, so on timeout one ``killpg`` stops it
    and every process it started. Timeouts, missing programs and files
    without execute permission come back as 124, 127 and 126, so callers
    handle one shape of failure instead of four. A child killed by
    signal N keeps Python's convention: return code -N.
    """
    argv = tuple(args)
    try:
        child = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=True,
        )
    except FileNotFoundError:
        return CommandResult(argv, NOT_FOUND, "", f"not found: {argv[0]}")
    except PermissionError:
        return CommandResult(argv, NOT_EXECUTABLE, "", f"denied: {argv[0]}")
    with child:
        try:
            out, err = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            note = f"timed out after {timeout:g}s"
            return CommandResult(argv, TIMED_OUT, "", note)
    return CommandResult(argv, child.returncode, out, err)
