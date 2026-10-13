# Contributing

Thank you for helping. The most useful contributions are corrections:
a bug in the code, a test that misses a case, or a place where the book
and the code disagree. For a larger change, open an issue first so that
we can agree on the approach before you spend time on it.

## Set up

You need Python 3.11 or later, `make` and `git`. The tooling runs on
macOS and Linux; on Windows, use WSL 2.

```bash
git clone https://github.com/shubhsha/ai-engineer-2030.git
cd ai-engineer-2030
make install   # .venv, the package with dev tools, and the git hooks
make check     # lint, types and tests: exactly what CI runs
```

`make install-all` adds the optional extras for the Claude adapter, the
MCP server and the HTTP service. `make help` lists every target.

## Make a change

1. Branch from `main`.
2. Make the change with its tests. Tests are offline and deterministic;
   `docs/adr/0002-offline-deterministic-tests.md` explains why and how.
3. Run `make fmt`, then `make check`, and fix everything it reports.
4. Commit with a [Conventional Commits](https://www.conventionalcommits.org/)
   message, such as `fix(retry): cap the jitter at the maximum delay`.
   The header is at most 72 characters. The commit-msg hook checks this
   locally, and CI checks every commit in a pull request.
5. Open a pull request with one idea in it, and fill in the template.

The conventions are written down in `CLAUDE.md` and `.claude/rules/`.
They are addressed to Claude Code, but they are the same rules for
people.

## The book prints this code

Many functions, classes and tests appear in the book by name, and some
files by line range. `docs/book-listings.txt` lists every one, and
`tests/tooling/test_book_listings.py` fails when a change renames,
moves or deletes a printed name or edits a printed range. If your change
has to do that, say which listing changes in the pull request. The
change may still be right, but it then needs an erratum or a new
edition, so it is decided on purpose.

## Report a mistake in the book

Open an issue with the **Erratum** form. Give the chapter, the page
number and the edition, quote the text, and say what it should be.

## Using Claude Code here

The repository ships its own Claude Code configuration: settings,
hooks that protect a few files and run ruff after each edit, skills
such as `/add-module` and `/explain`, and `reviewer` and `test-writer`
subagents. The hooks need `python3` on your `PATH`. Changes made with
an assistant are held to the same standard: you are the author, so read
and understand every line before you open the pull request.

## License

By contributing, you agree that your contribution is licensed under the
MIT License that covers this repository.
