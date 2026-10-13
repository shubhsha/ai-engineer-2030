---
paths:
  - "src/**/*.py"
---

# Source modules

- The module docstring starts with `Chapter N:` and says what the module
  is for. Keep the chapter right when code moves.
- Before you rename, move or delete a function, class, method or
  `# region`, search `docs/book-listings.txt`. A listed name is printed in
  the book: keep it, or say in the pull request which listing changes.
  Entries of the form `path:a-b sha256:...` are printed line ranges, so
  editing those lines changes a page. `tests/tooling/test_book_listings.py`
  fails in both cases.
- Inject what varies: time (`Clock`, `WallClock`), waiting (a `sleep`
  parameter), randomness (`random.Random`), models (`Model`) and storage,
  each with the production default. Nothing reads the clock, the network,
  the disk or the environment at import time.
- Values are frozen, slotted dataclasses that validate themselves in
  `__post_init__`. Raise a specific exception whose message says how to
  fix the input.
- Library code logs with `logging.getLogger(__name__)`; only command-line
  entry points print.
- `anthropic`, `mcp` and `fastapi` are imported only where CLAUDE.md
  allows, so that the core installs with NumPy alone.
