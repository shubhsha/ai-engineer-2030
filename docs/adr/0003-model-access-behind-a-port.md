# 0003. Put model access behind a port

- Status: Accepted
- Date: 2026-10-04
- Deciders: maintainer

## Context

Agents, evaluations and reports all call models. Providers, model
versions and prices change several times a year, and the tests must run
offline (ADR 0002).

## Options

1. Call the provider SDK directly wherever a model is needed. No
   indirection, but every caller depends on one vendor's types, and
   every test needs a network or a patch.
2. Wrap the SDK in a thin port that the domain owns, with one real
   adapter and one scripted adapter for tests.

## Decision

A `Model` protocol in `ai_engineer.llm.model` with a single method,
`complete(request) -> Reply`. `ClaudeModel` in `ai_engineer.claude`
adapts the Anthropic SDK, and imports it only when it has to build its
own client. `ScriptedModel` replays prepared replies for tests.

## Because

Tests become deterministic, a change of model or provider touches one
adapter, and cost and latency can be measured in one place.

## Consequences

New provider features reach the rest of the code only when they are
added to the port on purpose, which slows their adoption a little. The
scripted adapter must behave like the real one, so both run the same
contract tests in `tests/agentic/test_model_contract.py`.

## Revisit when

The port starts to mirror every SDK feature, or a second provider is
never needed.
