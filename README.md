# AI Engineer 2030: companion code

[![CI](https://github.com/shubhsha/ai-engineer-2030/actions/workflows/ci.yml/badge.svg)](https://github.com/shubhsha/ai-engineer-2030/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11 to 3.14](https://img.shields.io/badge/python-3.11%20to%203.14-blue.svg)

This is the code for *AI Engineer 2030: The Architect's Field Guide* by
Shubham Sharma, an Anthropic Certified Claude Architect Professional.
The book's listings are printed from these files, so the code on the
page is the same code that passes CI here.

## Read the book, free

The complete edition (six volumes, 79 chapters, about 1,600 pages) is
free to download from the
[latest release](https://github.com/shubhsha/ai-engineer-2030/releases/latest).
It is free for this launch. More focused books will follow.

It is written to be read. Modules are small and strictly typed. The
tests run offline in a few seconds and give the same result on every
run, because the clock, randomness, storage and the model are all
injected.

## Quick start

You need Python 3.11 or newer, `make` and `git` on macOS or Linux. On
Windows, use WSL 2.

```bash
git clone https://github.com/shubhsha/ai-engineer-2030.git
cd ai-engineer-2030
make install      # .venv, the package with dev tools, and the git hooks
make check        # ruff, strict mypy and the tests, exactly as CI runs them
.venv/bin/aie doctor
```

`make install-all` adds three optional extras: the Anthropic SDK
(`claude`), the MCP SDK (`mcp`) and the HTTP service (`api`). The core
package needs only NumPy. `make help` lists every target.

To skip local setup, open the repository in its dev container
(`.devcontainer/`). The container runs Python 3.14, installs every extra
and includes Claude Code.

## What is here

```text
src/ai_engineer/
  foundations/  Vol I    values, clocks, retries, idempotency, SQL, logs
  service/      Vol I    a FastAPI quote service with idempotency keys
  enterprise/   Vol I    the warehouse port and its SQLite adapter
  ml/           Vol II   NumPy, from logistic regression to a transformer
  llm/          Vol III  the Model protocol, prompts, structure, tools
  rag/          Vol III  chunking, BM25, rank fusion, retrieval metrics
  agents/       Vol III  loop, plans, workflows, memory, multi-agent
  claude/       Vol III  the Claude adapter, catalog, routing and cost
  evals/        Vol III  harness, metrics and model-graded checks
  guardrails/   Vol III  redaction, provenance, the lethal trifecta
  mcp_server/   Vol III  a read-only MCP server over SQLite
  enterprise/   Vol IV   evidence, reconciliation, safe SQL, data quality,
                         SLOs, threat models, AI governance, FinOps
  devops/       Vol IV   JUnit to evidence, DORA keys, failure triage
  systems/      Vol V    capacity, clocks, quorums, outbox, k8s, AI gateway
  leadership/   Vol V    decisions, flow, growth plans, incidents, strategy
  future/       Vol VI   agent config linter, autonomy guards, the factory
  career/       Vol VI   learning, portfolio, weekly planning, transitions
tests/          mirrors src: foundations/, ml/, agentic/, tooling/
configs/        models.toml: model ids and list prices, with a date
data/synthetic/ invented data; examples write to data/local/ (ignored)
docs/adr/       architecture decision records
scripts/        the commit-msg hook and a log triage script
.claude/        settings, hooks, rules, skills and subagents for Claude Code
.github/        CI, Dependabot, issue and pull request templates
```

## Chapter map

The first line of each module's docstring names its chapter.

| Ch. | Topic | Code |
|---:|---|---|
| 1 | Your engineering workbench | `foundations/doctor.py`, `aie doctor` |
| 2 | Python from zero | `foundations/basics.py` |
| 3 | Python abstractions | `foundations/abstractions.py` |
| 4 | Professional Python | `foundations/settings.py`, `foundations/logs.py`, `cli.py` |
| 5 | Concurrent Python | `foundations/concurrency.py` |
| 6 | Linux and the command line | `foundations/proc.py`, `foundations/shutdown.py`, `scripts/triage.sh` |
| 7 | Git and collaboration | `foundations/commitmsg.py`, `scripts/hooks/commit-msg` |
| 8 | SQL and data modeling | `foundations/sqlkit.py` |
| 9 | APIs and contracts | `foundations/idempotency.py`, `service/app.py` |
| 10 | Networking | `foundations/retry.py` |
| 11 | Testing | `tests/conftest.py`, `tests/foundations/` |
| 12 | Clean code | `foundations/pricing.py`, `tests/foundations/legacy_pricing.py` |
| 13 | Architecture | `enterprise/warehouse.py`, `docs/adr/` |
| 14 | Math in code | `ml/numerics.py` |
| 15 | Machine learning | `ml/linear.py` |
| 16 | Neural networks | `ml/mlp.py` |
| 17 | CNNs and RNNs | `ml/conv.py` |
| 18 | Transformers | `ml/transformer.py` |
| 19 | Attention | `ml/attention.py` |
| 20 | Tokens and embeddings | `ml/bpe.py`, `ml/embed.py` |
| 21 | Inside an LLM | `ml/bigram.py`, `ml/sampling.py` |
| 22 | Fine-tuning | `ml/lora.py` |
| 23 | Quantization | `ml/quant.py` |
| 24 | Inference economics | `ml/inference.py` |
| 25 | Local models | `ml/local.py` |
| 26 | Prompts as specifications | `llm/prompt.py` |
| 27 | Structured output | `llm/structured.py` |
| 28 | Context engineering | `llm/context.py` |
| 29 | Retrieval | `rag/` |
| 30 | Tools and the agent loop | `llm/model.py`, `llm/tools.py`, `llm/budget.py`, `agents/loop.py` |
| 31 | Memory and durable state | `agents/memory.py`, `agents/durable.py` |
| 32 | Planning | `agents/plan.py` |
| 33 | Workflows | `agents/workflows.py` |
| 34 | Trustworthy agents | `agents/hooks.py`, `.claude/` |
| 35 | Multi-agent systems | `agents/multi.py` |
| 36 | Model Context Protocol | `mcp_server/warehouse.py`, `.mcp.json` |
| 37 | The Claude platform | `claude/` |
| 38 | Evaluations | `evals/` |
| 39 | Guardrails | `guardrails/` |

## The book quotes this code by name

A listing in the book names a module, a function, a class, a method or a
block between `# region name` and `# endregion`, and some print a fixed
range of lines. `docs/book-listings.txt` records every one, with a hash
for each line range. `tests/tooling/test_book_listings.py` fails if a
change renames a printed name or alters printed lines, so the book and
the code cannot drift apart unnoticed. If a change has to break a
listing, say which one in the pull request.

## Working with Claude Code

The repository carries the Claude Code setup the book recommends, and
every part of it is checked in, so you can read it:

- `CLAUDE.md` holds the commands, the layout and the rules that are
  easy to break.
- `.claude/settings.json` allows the `make` targets and read-only git.
  It asks before a commit, a push, an install or a network tool, and it
  denies reads of `.env`, `secrets/` and key files.
- Two hooks run on every edit. `protect_files.py` refuses edits to
  secrets and to the frozen legacy oracle. `ruff_after_edit.py` fixes
  and formats each edited Python file, then reports whatever it could
  not fix. Both need `python3` on your `PATH`.
- `.claude/rules/` loads rules for `src/` and `tests/` when Claude works
  there. The security rule loads in every session.
- The skills are `/add-module`, `/new-adr` and `/explain`. The
  subagents are `reviewer` and `test-writer`.
- `.mcp.json` starts the read-only warehouse server from Chapter 36. It
  runs over a synthetic SQLite database and needs `make install-all`.

## Model calls and prices

No test calls a model. The suite uses `ScriptedModel`, and
`tests/agentic/test_model_contract.py` checks that it behaves like the
real adapter. To call the API yourself, install the `claude` extra and
set `ANTHROPIC_API_KEY` in your environment. To keep it in `.env`, which
git ignores, load the file into your shell with `set -a; . ./.env; set +a`.
`.env.example` lists the variables the code reads.

`configs/models.toml` holds list prices in US dollars per million
tokens, as published on 2026-10-04. Prices change. Before you use the
cost tools for a real decision, check the provider's pricing page and
update both the prices and the date.

## Status

The code covers all six volumes. The first line of each module's
docstring names its chapter; the table above maps Volumes I to III, and
the layout lists the packages for Volumes IV to VI. `CHANGELOG.md`
records every change. From 1.0.0, the names the book prints are a
stable interface.

## Contributing, errata and security

- [CONTRIBUTING.md](CONTRIBUTING.md) covers setup, commit messages and
  the listing contract.
- If you find a mistake in the book, report it with the **Erratum**
  issue form.
- Report vulnerabilities privately, as [SECURITY.md](SECURITY.md)
  describes. Never use a public issue for them.

## License

The code is released under the [MIT License](LICENSE). The book's text
and illustrations are not part of this repository and the MIT License
does not cover them. They are © Shubham Sharma, all rights reserved. The
free PDF is for your own reading; please share the link to this page
rather than the file.

This is an independent project. Anthropic, PBC has not endorsed it and
is not affiliated with it. Claude is a trademark of Anthropic, PBC.
