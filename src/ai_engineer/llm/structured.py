"""Chapter 27: structured output, validated in code and repaired in a loop.

Constrained decoding (the API's structured outputs) makes replies parse
and match a schema's shape, but it cannot check business rules, and a
refusal or a truncated reply can still break the shape. So the code
that receives the output validates it, always, and asks the model to
repair what fails.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ai_engineer.llm.model import (
    JSON,
    Model,
    Request,
    Usage,
    assistant,
    user,
)

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}
_LIMITS = {
    str: ("minLength", "maxLength"),
    list: ("minItems", "maxItems"),
}


def _is(value: Any, kind: str) -> bool:
    if isinstance(value, bool) and kind in ("number", "integer"):
        return False  # in Python, True is an int; in JSON it is not
    return isinstance(value, _TYPES[kind])


def validate(value: Any, schema: JSON, path: str = "$") -> list[str]:
    """Check ``value`` against a subset of JSON Schema.

    Supports type, enum, required, properties, additionalProperties
    (false), items and the min/max keywords. Returns every error, each
    with a path, phrased so that a person or a model can fix it.
    """
    kind = schema.get("type")
    if kind is not None and not _is(value, kind):
        return [f"{path}: expected {kind}, got {type(value).__name__}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}: missing required {key!r}")
        for key, item in value.items():
            if key in props:
                errors += validate(item, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors + _bounds(value, schema, path)


def _bounds(value: Any, schema: JSON, path: str) -> list[str]:
    size: float
    if isinstance(value, bool):
        return []
    if isinstance(value, (str, list)):
        size, (low, high) = len(value), _LIMITS[type(value)]
    elif isinstance(value, (int, float)):
        size, (low, high) = value, ("minimum", "maximum")
    else:
        return []
    errors = []
    if low in schema and size < schema[low]:
        errors.append(f"{path}: {low} is {schema[low]}, got {size}")
    if high in schema and size > schema[high]:
        errors.append(f"{path}: {high} is {schema[high]}, got {size}")
    return errors


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json(text: str) -> tuple[Any, list[str]]:
    """Parse a reply as JSON, tolerating a Markdown code fence."""
    body = text.strip()
    if fenced := _FENCE.match(body):
        body = fenced.group(1)
    try:
        return json.loads(body), []
    except json.JSONDecodeError as exc:
        where = f"line {exc.lineno}, column {exc.colno}"
        reason = exc.msg.removesuffix(" at")
        return None, [f"not valid JSON: {reason} at {where}"]


Check = Callable[[Any], list[str]]


@dataclass(frozen=True, slots=True)
class Validated:
    value: Any
    attempts: int
    usage: Usage


class StructuredOutputError(Exception):
    def __init__(self, reason: str, errors: list[str], attempts: int):
        detail = "; ".join(errors[:3]) or "no usable output"
        super().__init__(f"{reason} after {attempts} attempt(s): {detail}")
        self.reason, self.errors, self.attempts = reason, errors, attempts


def generate_validated(
    model: Model,
    request: Request,
    schema: JSON,
    *,
    check: Check | None = None,
    max_attempts: int = 3,
) -> Validated:
    """Ask for JSON, validate it and feed the errors back until it passes.

    Schema errors and business-rule errors (from ``check``) are repaired
    the same way: the reply and the list of errors go back to the model.
    A refusal or a reply cut off by ``max_tokens`` is not retried, since
    the same request would fail the same way.
    """
    messages, usage = list(request.messages), Usage()
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        reply = model.complete(replace(request, messages=tuple(messages)))
        usage += reply.usage
        if reply.stop_reason in ("refusal", "max_tokens"):
            raise StructuredOutputError(reply.stop_reason, [], attempt)
        value, errors = parse_json(reply.text)
        if not errors:
            errors = validate(value, schema) or (check(value) if check else [])
        if not errors:
            return Validated(value, attempt, usage)
        messages += [assistant(reply), user(repair_prompt(errors))]
    raise StructuredOutputError("invalid output", errors, max_attempts)


def repair_prompt(errors: list[str]) -> str:
    listed = "\n".join(f"- {error}" for error in errors)
    return (
        f"Your JSON failed validation:\n{listed}\n"
        "Reply with the corrected JSON only."
    )
