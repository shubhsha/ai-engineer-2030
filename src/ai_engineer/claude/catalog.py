"""Chapter 37: the model catalog, read from ``configs/models.toml``.

Prices and limits live in configuration, not in code, because they
change: keep the file current and record the date you checked it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

TIERS = ("small", "medium", "large", "frontier")


@dataclass(frozen=True, slots=True)
class LongPrompt:
    """Higher prices for a whole call whose prompt exceeds ``above``."""

    above: int
    input_per_mtok: Decimal
    output_per_mtok: Decimal


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    tier: str
    context_window: int
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_multiplier: Decimal = Decimal("0.1")
    long_prompt: LongPrompt | None = None

    def prices(self, prompt_tokens: int) -> tuple[Decimal, Decimal]:
        """Input and output price per MTok for a prompt of this length."""
        long = self.long_prompt
        if long is not None and prompt_tokens > long.above:
            return long.input_per_mtok, long.output_per_mtok
        return self.input_per_mtok, self.output_per_mtok


@dataclass(frozen=True, slots=True)
class Catalog:
    models: tuple[ModelInfo, ...]
    prices_checked: str

    def get(self, model_id: str) -> ModelInfo:
        for model in self.models:
            if model.id == model_id:
                return model
        raise KeyError(f"unknown model {model_id!r}")


def parse_catalog(text: str) -> Catalog:
    data = tomllib.loads(text)
    models = tuple(_model(entry) for entry in data["model"])
    return Catalog(models, str(data.get("prices_checked", "unknown")))


def load_catalog(path: str | Path = "configs/models.toml") -> Catalog:
    return parse_catalog(Path(path).read_text(encoding="utf-8"))


def _model(entry: dict[str, Any]) -> ModelInfo:
    if entry["tier"] not in TIERS:
        raise ValueError(f"{entry['id']}: tier must be one of {TIERS}")
    info = ModelInfo(
        id=entry["id"],
        tier=entry["tier"],
        context_window=int(entry["context_window"]),
        input_per_mtok=Decimal(str(entry["input_per_mtok"])),
        output_per_mtok=Decimal(str(entry["output_per_mtok"])),
        cache_read_multiplier=Decimal(
            str(entry.get("cache_read_multiplier", "0.1"))
        ),
        long_prompt=_long_prompt(entry.get("long_prompt")),
    )
    if min(info.input_per_mtok, info.output_per_mtok) <= 0:
        raise ValueError(f"{info.id}: prices must be positive")
    return info


def _long_prompt(entry: dict[str, Any] | None) -> LongPrompt | None:
    if entry is None:
        return None
    return LongPrompt(
        above=int(entry["above"]),
        input_per_mtok=Decimal(str(entry["input_per_mtok"])),
        output_per_mtok=Decimal(str(entry["output_per_mtok"])),
    )
