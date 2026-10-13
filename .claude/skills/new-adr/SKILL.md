---
name: new-adr
description: Record an architecture decision as the next numbered ADR in docs/adr.
argument-hint: "[decision title]"
disable-model-invocation: true
---

# Record an architecture decision

Title: $ARGUMENTS

1. List `docs/adr/` and take the next four-digit number.
2. Copy `docs/adr/template.md` to `docs/adr/NNNN-kebab-case-title.md`.
3. Fill every section from this conversation: the context and forces,
   the options with what each would cost (doing nothing included), the
   decision, why it beats the others, the consequences good and bad, and
   the signal that would make it worth revisiting. Use the status
   `Proposed` unless the user has said the decision is made.
4. Add the record to the index table in `docs/adr/README.md`.
5. Do not invent facts. Where the conversation gives no answer, write
   `TODO(author):` and the question, so the gap is visible in review.
