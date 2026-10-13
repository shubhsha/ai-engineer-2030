# Changelog

All notable changes to this project are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The names and line ranges that the book prints, listed in
`docs/book-listings.txt`, are part of the public interface: from 1.0.0
on, changing one is a breaking change.

## [Unreleased]

### Added

- `enterprise`, `devops`, `systems`, `leadership`, `future` and `career`:
  the code for Volumes IV to VI.
- The README now links to the free complete edition of the book.

## [0.1.0] - 2026-10-04

First public release, with the code for Volumes I to III.

### Added

- `foundations`: money and other value objects, clocks for tests,
  retries with backoff and jitter, idempotent writes with a SQLite store
  that is safe under concurrent attempts, structured logs, settings,
  SQL migrations, subprocess handling, graceful shutdown and a
  commit-message checker.
- `service`: a FastAPI quote service with idempotency keys and RFC 9457
  problem details.
- `ml`: NumPy implementations from logistic regression to a small
  transformer, with tokenization, sampling, LoRA and quantization.
- `llm`, `rag`, `agents`, `claude`, `evals`, `guardrails` and
  `mcp_server`: the model port and a scripted model for tests, prompts,
  structured output, context and tool use, retrieval, agent loops and
  workflows, the Claude adapter with routing and cost estimates, an
  evaluation harness, guardrails and a read-only MCP server.
- `enterprise`: the warehouse port and its SQLite adapter.
- Tooling: `make` entry points, CI on Python 3.11 to 3.14, a commit-msg
  hook, Claude Code settings, hooks, skills, subagents and rules,
  architecture decision records, and a test that keeps every printed
  listing in step with the code.

[Unreleased]: https://github.com/shubhsha/ai-engineer-2030/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shubhsha/ai-engineer-2030/releases/tag/v0.1.0
