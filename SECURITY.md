# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue. Use
GitHub's private reporting: open the repository's **Security** tab and
choose **Report a vulnerability**. Describe the problem, the affected
file or command, and the smallest steps that reproduce it.

Never include real secrets, API keys or personal data in a report. If a
key was exposed, revoke it with its provider first; removing it from a
file does not remove it from git history.

The maintainer will acknowledge the report, agree a fix and a disclosure
date with you, and credit you in the changelog unless you prefer not to
be named.

## Scope

In scope: the code in `src/`, `tests/` and `scripts/`, the CI workflow,
and the Claude Code configuration in `.claude/` (settings, hooks, skills
and agents). The repository is teaching code, but people copy it, so a
pattern that would be unsafe in their systems is worth reporting too.

Out of scope: vulnerabilities in third-party packages, which belong with
their maintainers, and mistakes in the book's text, which go in an
erratum issue.

## Supported versions

Fixes go to the `main` branch and the latest release.
