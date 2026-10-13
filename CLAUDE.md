# AI Engineer 2030: companion code

This repository holds the code for the book *AI Engineer 2030: The
Architect's Field Guide*. Each module backs one chapter, and its docstring
names the chapter. The book prints listings straight from these files, so
treat names as a public interface.

## Commands

- `make install`: create `.venv`, install with dev tools, enable the git hooks
- `make install-all`: add the `claude`, `mcp` and `api` extras
- `make check`: lint, types and tests, exactly what CI runs; run it before
  you say a change is done
- `make fmt`: format and apply safe fixes
- `.venv/bin/pytest tests/foundations -q -k idempotency`: one area, one topic
- `.venv/bin/aie doctor`: check the local toolchain

## Layout

```
src/ai_engineer/
  foundations/  Vol I: values, clocks, retries, idempotency, SQL, logs
  service/      Vol I: the FastAPI quote service
  ml/           Vol II: NumPy from logistic regression to a transformer
  llm/          Vol III: the Model protocol, prompts, structure, tools
  rag/          Vol III: chunking, BM25, fusion and retrieval metrics
  agents/       Vol III: loop, plans, workflows, memory, multi-agent
  claude/       Vol III: the Claude adapter, catalog, routing and cost
  evals/        Vol III: harness, metrics and model-graded checks
  guardrails/   Vol III: redaction, provenance and the lethal trifecta
  mcp_server/   Vol III: a read-only MCP server over SQLite
  enterprise/   Vol IV: warehouse port, evidence and release gate,
                reconciliation, the SQL guard and the AI Test Engineer
  devops/       Vol IV: JUnit evidence, DORA keys, failure triage and
                the Azure DevOps publisher
  systems/      Vol V: capacity, clocks, quorums, the outbox, Kubernetes
                checks, the AI gateway and architecture fitness tests
  leadership/   Vol V: tools for staff work: decisions, strategy, teams,
                flow, incidents and engineering economics
  future/       Vol VI: AI-native delivery, coding-agent configuration,
                autonomy limits, the software factory, multimodal,
                simulation and human-AI teams
  career/       Vol VI: tools for the engineer's own growth: a working
                philosophy, the week, learning, portfolio and transitions
tests/          mirrors src: foundations/, ml/, agentic/, enterprise/,
                devops/, systems/, leadership/, future/, career/, tooling/
configs/        models.toml: model ids and list prices, with a checked date
data/synthetic/ synthetic data only; examples write to data/local/
docs/adr/       architecture decision records
docs/book-listings.txt  every name and line range the book prints
```

## Rules that are easy to break

1. **The book quotes code by name.** A listing refers to a module, a
   top-level function or class, a `Class.method`, or a block between
   `# region name` and `# endregion`. Renaming, moving or deleting one
   breaks a printed page. `docs/book-listings.txt`, written by the
   book's build, lists them all with a hash of each printed line range,
   and `tests/tooling/test_book_listings.py` fails when a change breaks
   one. If a change must, say which listing changes in the pull request.
2. **Tests are offline and deterministic.** Never call a network, sleep,
   read the real clock or use real keys in a test. Inject instead: the
   `clock` fixture (`FakeClock`), the `rng` fixture (seeded with 2030),
   `ScriptedModel` for models and SQLite in `tmp_path` for storage. A
   real-time test must be tiny and say why it needs real time.
3. **`tests/foundations/legacy_pricing.py` is frozen.** It is the oracle
   for a characterization test. A hook refuses edits to it.
4. **Optional dependencies stay optional.** The core needs only NumPy.
   `fastapi` is imported only by `service/`, and `anthropic` and `mcp`
   only inside the functions that build a client or a server. Tests that
   need an extra call `pytest.importorskip`, and the main CI job runs
   without the `claude` and `mcp` extras to prove it.
5. **No secrets, no client data.** `.env` is ignored and `.env.example`
   lists the variables. Data in the repository is synthetic.

## Style

- Python 3.11+, with `from __future__ import annotations` in every
  module that has code.
- Values are `@dataclass(frozen=True, slots=True)` and validate themselves
  in `__post_init__`. Boundaries are `Protocol`s.
- Money is `Decimal`. Durations come from a monotonic `Clock`; stored
  timestamps from `WallClock`.
- Ruff at 79 columns, and mypy `--strict` over `src` and `.claude/hooks`.
  The code has no `# type: ignore`; fix the type instead.
- Docstrings say what and why, in plain sentences. Comments explain
  decisions, not syntax.
- Errors are data: raise specific exceptions and return RFC 9457 problem
  documents from HTTP endpoints.

## Commits and pull requests

Conventional Commits, with a header of at most 72 characters, such as
`fix(retry): cap the jitter at the maximum delay`. The commit-msg hook
checks this locally and CI checks every commit in a pull request. Keep
each pull request to one idea, with tests, and run `make check` first.

## Where to look next

- `.claude/rules/` holds rules for `src/`, `tests/` and security.
- `/add-module` walks through adding code for a chapter.
- `/new-adr Title` records an architecture decision in `docs/adr/`.
- `/explain Name` walks a reader through a listing and its tests.
- The `reviewer` and `test-writer` subagents in `.claude/agents/`.
