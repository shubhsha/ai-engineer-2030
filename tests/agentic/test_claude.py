from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_engineer.claude.adapter import ClaudeModel, to_params, to_reply
from ai_engineer.claude.catalog import Catalog, load_catalog, parse_catalog
from ai_engineer.claude.cost import cache_break_even, call_cost
from ai_engineer.claude.retry import should_retry
from ai_engineer.claude.routing import (
    NoAcceptableReply,
    cascade,
    cascade_pays,
    cheapest_fit,
    expected_cascade_cost,
)
from ai_engineer.llm.model import (
    Reply,
    Request,
    ScriptedModel,
    ToolCall,
    Usage,
    ask,
    user,
)

CATALOG = load_catalog(Path(__file__).parents[2] / "configs/models.toml")
HAIKU = CATALOG.get("claude-haiku-4-5-20251001")
HAIKU_55 = CATALOG.get("claude-haiku-5-5")
SONNET = CATALOG.get("claude-sonnet-5-5")


def test_the_catalog_records_when_prices_were_checked():
    assert len(CATALOG.models) == 5
    assert CATALOG.prices_checked == "2026-10-08"
    with pytest.raises(KeyError, match="unknown model 'gpt-9'"):
        CATALOG.get("gpt-9")


@pytest.mark.parametrize(
    ("tier", "price", "message"),
    [("huge", 1, "tier must be one of"), ("small", 0, "must be positive")],
)
def test_a_bad_catalog_entry_is_rejected(tier, price, message):
    text = (
        f'[[model]]\nid = "m"\ntier = "{tier}"\ncontext_window = 1000\n'
        f"input_per_mtok = {price}\noutput_per_mtok = 5\n"
    )
    with pytest.raises(ValueError, match=message):
        parse_catalog(text)


def test_caching_a_shared_prefix_cuts_its_cost_by_ninety_five_percent():
    # 1,000 calls an hour share a 20,000-token prefix (input side only)
    prefix = 20_000
    plain = 1000 * call_cost(SONNET, Usage(input_tokens=prefix))
    write = call_cost(SONNET, Usage(cache_write_tokens=prefix))
    read = call_cost(SONNET, Usage(cache_read_tokens=prefix))
    assert plain == Decimal(40)
    assert write + 999 * read == Decimal("2.048")


def test_output_costs_more_and_a_batch_halves_everything():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert call_cost(HAIKU, usage) == Decimal(6)
    assert call_cost(HAIKU, usage, batch=True) == Decimal(3)


@pytest.mark.parametrize("model", CATALOG.models, ids=lambda m: m.id)
def test_a_cache_pays_for_itself_by_the_second_or_third_call(model):
    assert cache_break_even(model, "5m") == 2
    assert cache_break_even(model, "1h") == 3


def test_routing_picks_the_cheapest_model_whose_window_fits():
    assert cheapest_fit(CATALOG, prompt_tokens=10_000) == HAIKU_55
    older = Catalog((HAIKU, SONNET), CATALOG.prices_checked)
    assert cheapest_fit(older, prompt_tokens=10_000) == HAIKU
    assert cheapest_fit(older, prompt_tokens=500_000) == SONNET
    large = cheapest_fit(CATALOG, prompt_tokens=10_000, min_tier="large")
    assert large.id == "claude-opus-5-5"
    with pytest.raises(LookupError, match="2000000"):
        cheapest_fit(CATALOG, prompt_tokens=2_000_000)


def test_one_token_past_the_step_raises_the_price_of_the_whole_call():
    under = Usage(input_tokens=100_000, output_tokens=1_000)
    over = Usage(input_tokens=100_001, output_tokens=1_000)
    assert call_cost(HAIKU_55, under) == Decimal("0.0105")
    assert call_cost(HAIKU_55, over) == Decimal("0.0525005")


def test_cached_tokens_count_toward_the_prompt_length():
    usage = Usage(input_tokens=1_000, cache_read_tokens=100_000)
    assert call_cost(HAIKU_55, usage) == Decimal("0.0055")


REQUEST = Request("Reply with JSON only.", (user("Classify: blank page"),))


def is_json(reply):
    return reply.text.startswith("{")


def test_a_cascade_stops_at_the_first_acceptable_reply():
    cheap = ScriptedModel([Reply("It looks like a bug.", usage=Usage(500, 9))])
    strong = ScriptedModel([Reply('{"label": "bug"}', usage=Usage(500, 8))])
    out = cascade([("haiku", cheap), ("sonnet", strong)], REQUEST, is_json)
    assert (out.model, out.tried) == ("sonnet", ("haiku", "sonnet"))
    assert out.usage == Usage(1000, 17)  # both attempts are paid for


def test_a_cascade_that_finds_nothing_says_what_it_tried():
    models = [
        ("haiku", ScriptedModel(["no"])),
        ("opus", ScriptedModel(["no"])),
    ]
    with pytest.raises(NoAcceptableReply, match="from haiku, opus"):
        cascade(models, REQUEST, is_json)


def test_to_params_sends_only_what_is_set():
    bare = to_params("m", Request("", (user("hi"),)), cache_ttl=None)
    assert bare == {"model": "m", "max_tokens": 1024, "messages": [user("hi")]}
    schema = {"type": "object"}
    request = Request(
        "Be brief.", (user("hi"),), ({"name": "t"},), output_schema=schema
    )
    params = to_params("m", request, cache_ttl="1h", effort="low")
    assert params["system"] == "Be brief."
    assert params["tools"] == [{"name": "t"}]
    assert params["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert params["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": schema},
    }


def message(*blocks, stop="end_turn", cache_read=None):
    usage = SimpleNamespace(
        input_tokens=50,
        output_tokens=20,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=None,
    )
    return SimpleNamespace(content=blocks, usage=usage, stop_reason=stop)


TEXT = SimpleNamespace(type="text", text="Let me check.")
CALL = SimpleNamespace(
    type="tool_use", id="t1", name="lookup", input={"id": 7}
)


def test_to_reply_reads_text_tool_calls_and_cache_usage():
    reply = to_reply(message(TEXT, CALL, stop="tool_use", cache_read=900))
    assert reply.text == "Let me check."
    assert reply.tool_calls == (ToolCall("t1", "lookup", {"id": 7}),)
    assert reply.stop_reason == "tool_use"
    assert reply.usage == Usage(50, 20, 900, 0)


def test_a_missing_stop_reason_reads_as_end_turn():
    assert to_reply(message(TEXT, stop=None)).stop_reason == "end_turn"


class FakeClient:
    """Stands in for ``anthropic.Anthropic``: ``client.messages.create``."""

    def __init__(self, reply):
        self.messages, self.sent, self._reply = self, [], reply

    def create(self, **params):
        self.sent.append(params)
        return self._reply


def test_claude_model_sends_the_params_to_the_client():
    client = FakeClient(message(TEXT))
    model = ClaudeModel(
        "claude-haiku-4-5-20251001", client=client, effort="low"
    )
    assert ask(model, "hi").text == "Let me check."
    [sent] = client.sent
    assert sent["model"] == "claude-haiku-4-5-20251001"
    assert sent["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert sent["output_config"] == {"effort": "low"}


# region test_cascade_cost_by_hand
@pytest.mark.parametrize(
    ("rate", "expected"),
    [("0.6", "0.0099"), ("0.8", "0.0077"), ("0.95", "0.00605")],
)
def test_a_cascade_costs_the_small_price_plus_the_miss_rate_times_large(
    rate, expected
):
    small, large = Decimal("0.0055"), Decimal("0.011")
    cost = expected_cascade_cost(small, large, Decimal(rate))
    assert cost == Decimal(expected)


def test_a_cascade_loses_money_below_a_pass_rate_of_small_over_large():
    small, large = Decimal("0.0055"), Decimal("0.011")
    assert cascade_pays(small, large, Decimal("0.51"))
    assert not cascade_pays(small, large, Decimal("0.5"))
    assert not cascade_pays(small, large, Decimal("0.2"))


# endregion


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_transient_claude_errors_are_retried(status):
    assert should_retry(status)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_permanent_claude_errors_are_not_retried(status):
    assert not should_retry(status)


def test_a_spend_limit_429_is_not_retried():
    assert should_retry(429, "rate_limit_exceeded")
    assert not should_retry(429, "enforced_spend_limit_reached")
