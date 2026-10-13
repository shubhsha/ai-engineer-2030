"""Shared fixtures. Every test is offline and deterministic."""

from __future__ import annotations

import random

import numpy as np
import pytest

from ai_engineer.foundations.abstractions import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def rng() -> random.Random:
    return random.Random(2030)


@pytest.fixture
def np_rng() -> np.random.Generator:
    return np.random.default_rng(2030)
