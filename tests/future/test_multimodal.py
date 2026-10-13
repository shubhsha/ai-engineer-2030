from __future__ import annotations

import json
from decimal import Decimal

import numpy as np

from ai_engineer.claude.catalog import load_catalog
from ai_engineer.future.multimodal import (
    HIGH_RES,
    Extraction,
    count_image_tokens,
    extract_fields,
    image_tokens,
    make_page,
    needs_review,
    normalize_field,
    page_cost,
    patch_grid,
    resized_size,
    score_extraction,
)
from ai_engineer.llm.model import ScriptedModel

CATALOG = load_catalog()


# region test_documented_sizes
def test_documented_sizes() -> None:
    assert count_image_tokens(1000, 1000) == 1296
    assert count_image_tokens(1075, 1520) == 2145
    assert resized_size(1075, 1520) == (924, 1307)
    assert resized_size(1920, 1080) == (1456, 819)
    assert image_tokens(1920, 1080) == 1560
    assert image_tokens(1920, 1080, *HIGH_RES) == 2691
    assert resized_size(3840, 2160, *HIGH_RES) == (2576, 1449)


# endregion


def test_small_images_are_not_resized() -> None:
    assert resized_size(200, 200) == (200, 200)
    assert image_tokens(200, 200) == 64


def test_tall_images_mirror_wide_ones() -> None:
    w, h = resized_size(1080, 1920)
    assert (h, w) == resized_size(1920, 1080)


def test_page_cost_scales_with_pages_and_batch() -> None:
    haiku = CATALOG.get("claude-haiku-5-5")
    one = page_cost(haiku, 1, 1551, 300)
    assert (
        one
        == Decimal(1551) * Decimal("0.10") / 1_000_000
        + Decimal(300) * Decimal("0.50") / 1_000_000
    )
    assert page_cost(haiku, 10_000, 1551, 300) == one * 10_000
    assert page_cost(haiku, 1, 1551, 300, batch=True) == one / 2


def test_patch_grid_counts_a_224_image() -> None:
    image = np.zeros((224, 224), dtype=np.uint8)
    grid = patch_grid(image, 16)
    assert grid.shape == (14, 14, 16, 16)
    assert grid.shape[0] * grid.shape[1] == 196


def test_make_page_is_seeded_and_has_ink() -> None:
    a = make_page(np.random.default_rng(2030), 400, 300)
    b = make_page(np.random.default_rng(2030), 400, 300)
    assert (a == b).all()
    assert 0.05 < (a < 128).mean() < 0.6


def test_normalize_dates_amounts_names() -> None:
    assert normalize_field("date", "03/04/2030") == "2030-04-03"
    assert normalize_field("date", "April 3, 2030") == "2030-04-03"
    assert normalize_field("amount", "$1,204.5") == "1204.50"
    assert normalize_field("name", "  ACME, Ltd. ") == "acme ltd"
    assert normalize_field("text", "   ") is None
    assert normalize_field("date", None) is None


def test_score_counts_wrong_missing_and_invented() -> None:
    kinds = {"total": "amount", "date": "date"}
    want = [
        {"total": "10.00", "date": "2030-01-02", "po": "A1"},
        {"total": "20.00", "date": "2030-01-03"},
        {"total": "30.00", "date": "2030-01-04"},
    ]
    got = [
        {"total": "$10", "date": "02/01/2030", "po": "A1"},
        {"total": "21.00", "date": "2030-01-03", "po": "Z9"},
        {"date": "2030-01-04"},
    ]
    score = score_extraction(want, got, kinds)
    total, date, po = score["total"], score["date"], score["po"]
    assert (total.tp, total.fp, total.fn) == (1, 1, 2)
    assert date.f1 == 1.0
    assert (po.tp, po.fp, po.fn) == (1, 1, 0)
    assert total.precision == 0.5
    assert round(total.recall, 3) == 0.333


def test_extract_fields_reads_json_and_flags_review() -> None:
    reply = json.dumps(
        {
            "total": {"value": "12.50", "confidence": 0.97},
            "date": {"value": "2030-01-02", "confidence": 0.6},
            "po": {"value": None, "confidence": 0.0},
        }
    )
    model = ScriptedModel([reply])
    got = extract_fields(model, "Invoice ...", ["total", "date", "po"])
    assert got.values == {"total": "12.50", "date": "2030-01-02"}
    assert needs_review(got, ["total", "date", "po"]) == ("date", "po")
    assert "<doc>" in str(model.requests[0].messages[0])


def test_extract_fields_survives_bad_replies() -> None:
    for bad in ("not json", "[1, 2]", '{"total": 5}'):
        got = extract_fields(ScriptedModel([bad]), "x", ["total"])
        assert got == Extraction({}, {})
