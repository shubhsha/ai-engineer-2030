"""Chapter 4: configuration that fails fast and explains itself."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})


class SettingsError(ValueError):
    """Carries every problem found, not just the first one."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("invalid settings:\n  - " + "\n  - ".join(problems))
        self.problems = problems


@dataclass(frozen=True, slots=True)
class Settings:
    model_config: Path
    warehouse: Path
    log_level: str = "INFO"
    api_key: str | None = field(default=None, repr=False)  # never logged

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read ``AIE_*`` variables, validate all of them, then build."""
        env = os.environ if env is None else env
        problems = []
        level = env.get("AIE_LOG_LEVEL", "INFO").upper()
        if level not in LOG_LEVELS:
            problems.append(
                f"AIE_LOG_LEVEL must be one of "
                f"{sorted(LOG_LEVELS)}, got {level!r}"
            )
        config = Path(env.get("AIE_MODEL_CONFIG", "configs/models.toml"))
        if config.suffix != ".toml":
            problems.append(
                f"AIE_MODEL_CONFIG must be a .toml file, got {config}"
            )
        warehouse = Path(env.get("AIE_WAREHOUSE", "data/local/warehouse.db"))
        if problems:
            raise SettingsError(problems)
        key = env.get("ANTHROPIC_API_KEY") or None
        return cls(config, warehouse, level, key)


def parse_env_file(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines from a .env file; skip blanks, comments."""
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, eq, value = line.partition("=")
        if not eq or not key.strip().isidentifier():
            raise ValueError(f"line {number}: expected KEY=value")
        values[key.strip()] = value.strip().strip("'\"")
    return values
