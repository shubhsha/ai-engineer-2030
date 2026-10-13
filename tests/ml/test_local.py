"""Tests for chapter 25: planning, the local adapter, safe parsing."""

from __future__ import annotations

import json
import struct
from typing import Any

import pytest

from ai_engineer.llm.model import Request, user
from ai_engineer.ml.local import (
    LocalChatModel,
    Machine,
    ModelShape,
    plan_local,
    read_safetensors_header,
)

M8 = ModelShape("8B", 8.03e9, 32, 8, 128)


def test_plan_fits_and_ceiling() -> None:
    plan = plan_local(M8, Machine("m1", 16, 68.25), bits=4.5, context=8192)
    assert plan.fits
    assert round(plan.weights_gb, 2) == 4.52
    assert round(plan.kv_gb, 2) == 1.07
    assert round(plan.max_tokens_per_s, 1) == 15.1


def test_plan_refuses_16_bit_on_16gb() -> None:
    plan = plan_local(M8, Machine("m1", 16, 68.25), bits=16, context=8192)
    assert not plan.fits
    assert plan.headroom_gb < 0


def test_adapter_translates_request_and_reply() -> None:
    seen: list[dict[str, Any]] = []

    def post(body: dict[str, Any]) -> dict[str, Any]:
        seen.append(body)
        return {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        }

    reply = LocalChatModel(post).complete(
        Request(system="be brief", messages=(user("hello"),))
    )
    assert reply.text == "hi"
    assert reply.usage.total == 9
    assert seen[0]["messages"][0] == {
        "role": "system",
        "content": "be brief",
    }
    assert seen[0]["messages"][1]["content"] == "hello"


def _file(header: dict[str, Any], data: bytes) -> bytes:
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + data


GOOD = {
    "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
    "b": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
}


def test_header_parses() -> None:
    header = read_safetensors_header(_file(GOOD, bytes(12)))
    assert header["w"]["shape"] == [2]


def test_header_rejects_holes_and_trailing_bytes() -> None:
    with pytest.raises(ValueError, match="indexed"):
        read_safetensors_header(_file(GOOD, bytes(16)))
    holey = {"w": {**GOOD["w"], "data_offsets": [4, 12]}}
    with pytest.raises(ValueError, match="hole"):
        read_safetensors_header(_file(holey, bytes(12)))


def test_header_rejects_absurd_length() -> None:
    with pytest.raises(ValueError, match="range"):
        read_safetensors_header(struct.pack("<Q", 10**12) + b"{}")
