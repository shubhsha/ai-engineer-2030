---
name: explain
description: Explain a function, class, method or listing from the book. Find it in the code, find the tests that pin its behavior, run them, and walk the reader through both. Use when a reader asks how a piece of the book's code works or why it is written that way.
argument-hint: "[name, such as SqliteIdempotencyStore.execute]"
allowed-tools: Read Grep Glob Bash(.venv/bin/pytest *)
---

# Explain code from the book

The reader asked about: $ARGUMENTS

1. Find the definition with Grep under `src/` and `tests/`. The module
   docstring names the chapter that explains it; mention that chapter.
2. Find the tests that exercise it, and run only those with
   `.venv/bin/pytest -q <file>::<test>`. Show the result.
3. Explain in this order: what problem the code solves, the decisions
   visible in it and what each one prevents, then how the tests prove
   the behavior. Quote short excerpts with file and line, not whole
   files.
4. Suggest one experiment the reader can run: a change that should make
   a specific test fail, and why it fails. Do not make the change.
