---
name: add-module
description: Add or extend a module for one of the book's chapters, with tests, in this repository's conventions. Use when creating a file under src/ai_engineer or adding a public function that a chapter will print.
argument-hint: "[package/module] [chapter]"
---

# Add a module for a chapter

The request: $ARGUMENTS (for example `rag/rerank 29`).

1. Read two existing modules in the same package and their tests, and
   match them. Read `.claude/rules/src.md` and `.claude/rules/tests.md`.
2. Create or extend `src/ai_engineer/<package>/<module>.py`:
   - The docstring's first line is `Chapter N: <what it does>.`
   - `from __future__ import annotations` follows the docstring.
   - Time, sleep, randomness, models and storage come in as parameters,
     with the real implementation as the default. Nothing touches the
     network or the disk at import time.
3. Write the tests beside the code, in the test folder for that part of
   the book: `tests/foundations/` (Volume I), `tests/ml/` (Volume II),
   `tests/agentic/` (Volume III), or a new folder named after the
   package for later volumes. Pin examples with the arithmetic in a
   comment, state properties with Hypothesis where a law exists, and use
   fakes rather than mocks.
4. If the chapter will print only part of a function or test, wrap that
   part in `# region kebab-name` and `# endregion`.
5. Run `make check` and fix every finding. Never weaken an assertion to
   make a test pass; if a test is wrong, say why.
6. Finish with a short report: the public names you added, since the
   book may quote them, and what each new test pins down.
