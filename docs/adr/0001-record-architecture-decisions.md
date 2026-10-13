# 0001. Record architecture decisions

- Status: Accepted
- Date: 2026-10-04
- Deciders: maintainer

## Context

This repository is read by people who were not in the room. Readers of
the book copy its patterns into their own systems, and contributors
change code that a printed page explains. When the reason for a design
is lost, the design is either defended by habit or quietly undone by
someone who never knew why it was there.

## Options

1. Keep decisions in pull request descriptions. Cheap, but scattered and
   hard to find a year later.
2. Keep a design document that is updated in place. One place to read,
   but it shows only the latest thinking and hides what changed and why.
3. Keep architecture decision records: short, numbered files in
   `docs/adr/`, one per decision, as Michael Nygard proposed.

## Decision

Record each significant decision as an ADR in `docs/adr/`, from
`template.md`, numbered in sequence and listed in `README.md`. A
decision is significant when it is expensive to reverse, when it
constrains other code, or when a reasonable engineer could have chosen
differently. The `/new-adr` skill starts one.

## Because

Records kept with the code are reviewed with the code and found with it.
One decision per file keeps each record small, and numbering preserves
the order in which the thinking happened.

## Consequences

An accepted record is not edited, apart from its status. A change of
mind is a new record that supersedes the old one and links back to it,
so the history survives. Writing a record takes about half an hour,
which is why only significant decisions get one.

## Revisit when

Records pile up that nobody reads, or decisions are made that nobody
records.
