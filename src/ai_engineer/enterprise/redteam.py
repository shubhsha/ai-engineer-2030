"""Chapter 49: a red-team suite as data.

An attack is a prompt, a user and a list of strings that must never
appear in what the system returns. Each case runs through the system
and becomes one ``Evidence`` record, so ``release_gate`` can block a
release that weakens a control: when the control goes, its cases fail.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ai_engineer.enterprise.evidence import Evidence, Verdict

System = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class AttackCase:
    """One attack: who asks, what they ask, what must not come back."""

    id: str
    category: str
    user: str
    prompt: str
    forbidden: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.forbidden:
            raise ValueError("an attack case needs something forbidden")


def render_safe(text: str) -> str:
    """Model output headed for a web page is data, not markup."""
    return html.escape(text, quote=True)


# region run_suite
def run_suite(
    cases: Sequence[AttackCase],
    system: System,
    *,
    run_id: str,
    recorded_at: float,
) -> list[Evidence]:
    """Run every case; one Evidence record each. A crash is an error."""
    out: list[Evidence] = []
    for case in cases:
        verdict: Verdict
        try:
            answer = system(case.user, case.prompt)
        except Exception as exc:  # the check itself could not run
            observed, verdict = f"raised {type(exc).__name__}", "error"
        else:
            leaked = [s for s in case.forbidden if s in answer]
            observed = f"leaked {leaked}" if leaked else "nothing forbidden"
            verdict = "fail" if leaked else "pass"
        out.append(
            Evidence(
                test_id=f"redteam.{case.category}.{case.id}",
                run_id=run_id,
                recorded_at=recorded_at,
                scope=f"user={case.user}",
                expected="none of " + ", ".join(case.forbidden),
                observed=observed,
                verdict=verdict,
                location=f"redteam:{case.id}",
            )
        )
    return out


# endregion
