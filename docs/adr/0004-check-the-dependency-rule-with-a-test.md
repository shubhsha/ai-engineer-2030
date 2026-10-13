# 0004. Check the dependency rule with a test

- Status: Accepted
- Date: 2026-10-07
- Deciders: maintainer

## Context

On the day of this record, `src/ai_engineer` has 70 modules and 59
imports between them. No two packages import each other, and one module,
`claude/adapter.py`, imports the Anthropic SDK (ADR 0003). Nothing keeps
it that way except review. One convenient import can close a cycle or let
a vendor type into the core, and the cost shows months later, when a
package cannot be tested or moved without dragging another along.

## Options

1. Rely on review. Free, but a reviewer skims imports in a large diff,
   and the rule is written down nowhere that a machine reads.
2. Adopt an import-linting tool with its own contract file. Strong, but
   it adds a dependency and a second configuration language for two rules.
3. Write the two rules as ordinary tests that parse the source with
   `ast`: no import cycle between packages, and only `claude/adapter.py`
   imports the SDK.

## Decision

Option 3. The rules live in `tests/foundations/test_warehouse.py`, run in
the normal offline suite on every push, and fail with the name of the
cycle or of the module that broke the rule. A new rule is a new test. A
rule that no longer fits is replaced by a new ADR that supersedes this one.

## Because

The tests and their helpers are about 75 lines, need no new dependency and
run in under 0.1 seconds. A failing test names the file to fix, which is more than a
review comment can promise on the tenth pull request of the day.

## Consequences

Easier: the dependency rule is executable, so it survives staff changes
and fast-moving code.

Harder: the tests see only `import` statements. A dynamic
`importlib.import_module` call is invisible to them, and the package is
the smallest unit they check, so a cycle between two modules of one
package passes.

## Revisit when

There are more than about five rules, or a rule has to name modules that
change every month. That is the point to move to a tool.
