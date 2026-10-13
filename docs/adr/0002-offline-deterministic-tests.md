# 0002. Keep every test offline and deterministic

- Status: Accepted
- Date: 2026-10-04
- Deciders: maintainer

## Context

Readers run the suite on laptops, in containers and behind corporate
proxies, often without an API key. The code depends on time (expiry,
retries, rate limits), on randomness (sampling, splits, jitter), on
storage and on language models. A test that touches any of these for
real is slow, can fail for reasons unrelated to the code, and teaches
people to rerun failures instead of reading them.

## Options

1. Test against real services, with keys in CI secrets. Realistic, but
   slow, costly, impossible without a key, and flaky.
2. Patch globals in each test with mocks and libraries that freeze time.
   Quick to write, but it couples tests to implementation details and
   cannot reach code in other processes.
3. Inject every source of variation through a parameter with a
   production default, and give tests fakes that they control.

## Decision

Code takes time from a `Clock` (or `WallClock` for stored timestamps),
waits through an injected `sleep`, draws randomness from an injected
generator, calls models through the `Model` port and stores data behind
a protocol. Tests use the shared fixtures in `tests/conftest.py`:
`clock` (a `FakeClock` at zero), `rng` and `np_rng` (seeded with 2030).
They use `ScriptedModel` for models and SQLite under `tmp_path` for
storage. No test opens a network connection.

## Because

The suite runs in seconds anywhere and gives the same answer every time,
so a failure always means something. Fake time also reaches cases that
real time cannot, such as the exact instant an entry expires or a key
that lives for a day.

## Consequences

Each dependency needs a parameter and a fake, which is a little more
code. Fakes can drift from the real thing, so they are held to the same
contract tests as the real adapters (`tests/agentic/test_model_contract.py`).
Behavior that only real time can show, such as a child process killed
at a timeout, gets a few tests with millisecond durations that say why
they need real time. What a model actually answers is measured by
evaluations, not by these tests.

## Revisit when

A class of bug reaches readers that only a test against a real service
would have caught.
