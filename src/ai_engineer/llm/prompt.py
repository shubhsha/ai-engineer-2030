"""Chapter 26: a prompt as a versioned, testable specification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def tagged(tag: str, body: str) -> str:
    """Wrap ``body`` in XML-style tags.

    A closing tag inside the body is broken up, so pasted text cannot
    end its section early and pose as instructions.
    """
    safe = body.replace(f"</{tag}>", f"< /{tag}>")
    return f"<{tag}>\n{safe}\n</{tag}>"


@dataclass(frozen=True, slots=True)
class Example:
    input: str
    output: str


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """The parts of an instruction, kept apart so each can be reviewed.

    ``render`` produces the system prompt: stable text that belongs at
    the front of every call, where a prompt cache can reuse it. Data
    for one call goes in the user turn (see ``user_turn``).
    """

    name: str
    role: str
    task: str
    rules: tuple[str, ...] = ()
    output: str = ""
    examples: tuple[Example, ...] = ()

    def render(self) -> str:
        parts = [self.role, tagged("task", self.task)]
        if self.rules:
            rules = "\n".join(f"- {rule}" for rule in self.rules)
            parts.append(tagged("rules", rules))
        if self.examples:
            shots = "\n".join(
                tagged(
                    "example",
                    tagged("input", e.input)
                    + "\n"
                    + tagged("output", e.output),
                )
                for e in self.examples
            )
            parts.append(tagged("examples", shots))
        if self.output:
            parts.append(tagged("output_format", self.output))
        return "\n\n".join(parts)

    @property
    def version(self) -> str:
        """A content hash. Log it with every call, so that any output
        can be traced to the exact prompt that produced it."""
        digest = hashlib.sha256(self.render().encode()).hexdigest()
        return f"{self.name}@{digest[:10]}"


def user_turn(**sections: str) -> str:
    """The data for one call, each part in its own tagged section."""
    return "\n\n".join(tagged(name, text) for name, text in sections.items())


# region prompt_sizes
def estimated_tokens(text: str) -> int:
    """A rough token count: about four characters per token.

    An estimate for budgeting, never for billing. The API reports the
    real count.
    """
    return -(-len(text) // 4)


def part_sizes(spec: PromptSpec) -> dict[str, int]:
    """Characters in each tagged part of the rendered system prompt."""
    sizes = {"role": len(spec.role), "task": len(tagged("task", spec.task))}
    if spec.rules:
        rules = "\n".join(f"- {rule}" for rule in spec.rules)
        sizes["rules"] = len(tagged("rules", rules))
    if spec.examples:
        shots = "\n".join(
            tagged(
                "example",
                tagged("input", e.input) + "\n" + tagged("output", e.output),
            )
            for e in spec.examples
        )
        sizes["examples"] = len(tagged("examples", shots))
    if spec.output:
        sizes["output_format"] = len(tagged("output_format", spec.output))
    return sizes


# endregion


# region example_balance
def majority_share(spec: PromptSpec) -> float:
    """The share of examples that carry the most common output.

    1.0 means every example gives the same answer, so the model learns
    to give it too (see "Calibrate Before Use").
    """
    if not spec.examples:
        return 0.0
    counts: dict[str, int] = {}
    for example in spec.examples:
        counts[example.output] = counts.get(example.output, 0) + 1
    return max(counts.values()) / len(spec.examples)


# endregion
