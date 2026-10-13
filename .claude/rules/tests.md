---
paths:
  - "tests/**/*.py"
---

# Tests

- Offline and deterministic: no network, no `time.sleep`, no real clock,
  no real keys, no unseeded randomness. Use the `clock` fixture
  (`FakeClock`), a fake `sleep`, the `rng` fixture (seeded with 2030),
  `ScriptedModel`, and SQLite under `tmp_path`.
- A test that needs real time, such as a timeout that kills a process,
  keeps its durations in milliseconds and says in a comment why.
- Prefer fakes to mocks, and assert on outcomes rather than on calls.
- Name each test as a sentence about one behavior. Test boundaries
  exactly, and write the arithmetic behind expected numbers in a comment.
- One condition per `assert` (ruff PT018); `pytest.raises(..., match=...)`
  for errors; `pytest.importorskip` for optional extras.
- The book prints some tests. Wrap the printed part in
  `# region kebab-name` and `# endregion`, and check
  `docs/book-listings.txt` before renaming a test or a region.
- `tests/foundations/legacy_pricing.py` is frozen: it is the oracle of a
  characterization test.
