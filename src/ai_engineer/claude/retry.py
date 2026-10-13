"""Chapter 37: which Claude API errors are worth retrying."""

from __future__ import annotations

from ai_engineer.foundations.retry import RETRYABLE_STATUS

# 500 (api_error) and 529 (overloaded_error) are on top of the generic set.
CLAUDE_RETRYABLE = RETRYABLE_STATUS | frozenset({500, 529})
SPEND_LIMIT = "enforced_spend_limit_reached"


def should_retry(status: int, error_code: str | None = None) -> bool:
    """True when waiting and trying again can succeed.

    A spend-limit 429 looks like a rate limit but carries no
    ``retry-after`` header and lasts until access is restored, so
    retrying it only burns attempts.
    """
    if error_code == SPEND_LIMIT:
        return False
    return status in CLAUDE_RETRYABLE
