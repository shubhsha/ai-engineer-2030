import pytest

from ai_engineer.llm.model import Reply, Request, ScriptedModel, user
from ai_engineer.llm.structured import (
    StructuredOutputError,
    generate_validated,
    parse_json,
    validate,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["bug", "question", "billing"]},
        "priority": {"type": "integer", "minimum": 1, "maximum": 4},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["label", "priority"],
    "additionalProperties": False,
}
REQUEST = Request("Label the ticket. Reply with JSON.", (user("Blank page"),))


def test_a_valid_value_has_no_errors():
    value = {"label": "bug", "priority": 2, "tags": ["ui"]}
    assert validate(value, SCHEMA) == []


def test_each_error_names_its_path():
    value = {"label": "outage", "priority": 9, "tags": ["a", 1], "x": 0}
    errors = validate(value, SCHEMA)
    assert (
        "$.label: 'outage' is not one of ['bug', 'question', 'billing']"
        in errors
    )
    assert "$.priority: maximum is 4, got 9" in errors
    assert "$.tags[1]: expected string, got int" in errors
    assert "$: unexpected property 'x'" in errors


def test_missing_fields_are_reported():
    assert validate({"label": "bug"}, SCHEMA) == [
        "$: missing required 'priority'"
    ]


def test_a_boolean_is_not_an_integer():
    assert validate(True, {"type": "integer"}) == [
        "$: expected integer, got bool"
    ]


def test_parse_json_strips_a_fence_and_locates_errors():
    assert parse_json('```json\n{"a": 1}\n```') == ({"a": 1}, [])
    value, errors = parse_json('{"a": 1,}')
    assert value is None
    assert errors[0].startswith("not valid JSON")
    assert "line 1" in errors[0]


def test_invalid_output_is_repaired_with_the_errors():
    model = ScriptedModel(
        [
            '{"label": "outage", "priority": 2}',
            '{"label": "bug", "priority": 2}',
        ]
    )
    out = generate_validated(model, REQUEST, SCHEMA)
    assert out.value == {"label": "bug", "priority": 2}
    assert out.attempts == 2
    assert (
        "'outage' is not one of" in model.requests[1].messages[-1]["content"]
    )
    assert out.usage.output_tokens > 0


def test_business_rules_are_repaired_the_same_way():
    def billing_is_never_urgent(value):
        if value["label"] == "billing" and value["priority"] == 1:
            return ["$.priority: billing tickets are never priority 1"]
        return []

    model = ScriptedModel(
        [
            '{"label": "billing", "priority": 1}',
            '{"label": "billing", "priority": 3}',
        ]
    )
    out = generate_validated(
        model, REQUEST, SCHEMA, check=billing_is_never_urgent
    )
    assert out.value["priority"] == 3


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_refusals_and_truncation_are_not_retried(stop):
    model = ScriptedModel([Reply('{"label": "bu', stop_reason=stop)])
    with pytest.raises(StructuredOutputError) as info:
        generate_validated(model, REQUEST, SCHEMA)
    assert info.value.reason == stop
    assert len(model.requests) == 1


def test_it_gives_up_after_max_attempts():
    model = ScriptedModel(default="Sure! Here is the JSON you asked for.")
    with pytest.raises(StructuredOutputError, match="invalid output after 3"):
        generate_validated(model, REQUEST, SCHEMA)
    assert len(model.requests) == 3
