"""Chapter 39: the lethal trifecta, checked when an agent is configured."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Capability:
    tool: str
    private_data: bool = False  # reads data the attacker should not see
    untrusted_input: bool = False  # reads text an attacker could write
    outbound: bool = False  # can send data somewhere an attacker can read


def lethal_trifecta(capabilities: Sequence[Capability]) -> list[str]:
    """Explain the risk if one agent holds all three capabilities.

    With private data, untrusted input and a way out, an injected
    instruction can make the agent leak the data (Willison, 2025). An
    empty list means at least one leg is missing. Remove a leg; no
    prompt reliably defends a configuration with all three.
    """
    legs = {
        "private data": [c.tool for c in capabilities if c.private_data],
        "untrusted input": [c.tool for c in capabilities if c.untrusted_input],
        "outbound channel": [c.tool for c in capabilities if c.outbound],
    }
    if not all(legs.values()):
        return []
    return [f"{leg}: {', '.join(tools)}" for leg, tools in legs.items()]
