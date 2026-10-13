"""Chapter 25: will this model run on this machine, and how fast?

Two numbers answer most of the question. Memory decides whether the
weights and the KV cache fit at all. Memory bandwidth bounds decoding
speed, because generating each token streams every weight once.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_engineer.llm.model import Reply, Request, Usage
from ai_engineer.ml.inference import kv_cache_bytes, weight_bytes

GB = 1e9


@dataclass(frozen=True)
class ModelShape:
    name: str
    params: float
    layers: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class Machine:
    name: str
    memory_gb: float
    bandwidth_gb_s: float
    reserved_gb: float = 4.0  # operating system, runtime, other apps


@dataclass(frozen=True)
class LocalPlan:
    weights_gb: float
    kv_gb: float
    headroom_gb: float
    max_tokens_per_s: float

    @property
    def fits(self) -> bool:
        return self.headroom_gb >= 0


def plan_local(
    model: ModelShape,
    machine: Machine,
    *,
    bits: float,
    context: int,
    kv_bits: float = 16,
) -> LocalPlan:
    """Memory plan and decode-speed ceiling for one local user.

    The speed is an upper bound: real runtimes reach a fraction of the
    memory bandwidth, and long contexts add KV-cache reads per token.
    """
    weights = weight_bytes(model.params, bits) / GB
    kv = (
        kv_cache_bytes(
            layers=model.layers,
            kv_heads=model.kv_heads,
            head_dim=model.head_dim,
            tokens=context,
            bytes_per_value=kv_bits / 8,
        )
        / GB
    )
    headroom = machine.memory_gb - machine.reserved_gb - weights - kv
    return LocalPlan(weights, kv, headroom, machine.bandwidth_gb_s / weights)


# region local_chat_model
Post = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class LocalChatModel:
    """An adapter from the Model port to a local chat server.

    ``post`` sends one JSON body to the server and returns its JSON
    answer. Injecting it keeps the adapter testable without a network.
    """

    post: Post
    model: str = "local"

    def complete(self, request: Request) -> Reply:
        body = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": "system", "content": request.system},
                *request.messages,
            ],
        }
        answer = self.post(body)
        usage = answer.get("usage", {})
        return Reply(
            text=answer["choices"][0]["message"]["content"],
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
        )


# endregion


# region safetensors_header
def read_safetensors_header(blob: bytes) -> dict[str, Any]:
    """Parse a safetensors header without running or trusting anything.

    Raises ValueError if the length is absurd or the ranges leave holes.
    """
    if len(blob) < 8:
        raise ValueError("file shorter than the length prefix")
    (n,) = struct.unpack("<Q", blob[:8])
    if n > 100_000_000 or 8 + n > len(blob):
        raise ValueError("header length is out of range")
    header: dict[str, Any] = json.loads(blob[8 : 8 + n])
    spans = sorted(
        tuple(v["data_offsets"])
        for k, v in header.items()
        if k != "__metadata__"
    )
    end = 0
    for begin, stop in spans:
        if begin != end:
            raise ValueError("tensor ranges leave a hole or overlap")
        end = stop
    if 8 + n + end != len(blob):
        raise ValueError("data buffer is not fully indexed")
    return header


# endregion
