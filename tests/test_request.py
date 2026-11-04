"""Request validation keeps token content separate from candidate metadata."""

import copy

import pytest

from cxl_recsys.common.request import MAX_CANDIDATE_TOKENS, placeholder, validate


def flat_request():
    return {
        "request_id": "request-flat",
        "user_id": "user-flat",
        "history_token_ids": [0, 42, 42],
        "candidate_token_ids": [7, 0, 2**31 - 1] * 171,
        "candidate_item_ids": ["item-z", "item-a"],
        "timestamp": 1.25,
        "visit_index": 0,
    }


def test_flat_request_preserves_long_suffix_and_metadata_without_aliasing():
    request = flat_request()
    original = copy.deepcopy(request)
    parsed = validate(request)
    assert parsed == original
    assert len(parsed["candidate_token_ids"]) == 513
    parsed["history_token_ids"][0] = 5
    parsed["candidate_token_ids"][0] = 6
    parsed["candidate_item_ids"][0] = "different-item"
    assert request == original


def test_flat_suffix_limit_is_independent_of_candidate_item_count():
    request = flat_request()
    request["candidate_item_ids"] = ["one-item"]
    request["candidate_token_ids"] = [0] * MAX_CANDIDATE_TOKENS
    assert validate(request) == request
    request["candidate_token_ids"].append(0)
    with pytest.raises(ValueError):
        validate(request)


def test_legacy_request_remains_compatible_with_optional_metadata():
    request = placeholder()
    request.update(timestamp=0, visit_index=3)
    original = copy.deepcopy(request)
    parsed = validate(request)
    assert parsed == original
    parsed["candidates"][0]["token_ids"][0] = 5
    assert request == original


@pytest.mark.parametrize("field", ["candidate_token_ids", "candidate_item_ids"])
def test_legacy_and_flat_fields_cannot_be_mixed_even_if_null(field):
    request = placeholder()
    request[field] = None
    with pytest.raises(ValueError, match="cannot be mixed"):
        validate(request)


@pytest.mark.parametrize("field", ["candidate_token_ids", "candidate_item_ids"])
def test_both_flat_candidate_fields_are_required(field):
    request = flat_request()
    del request[field]
    with pytest.raises(ValueError):
        validate(request)


@pytest.mark.parametrize(
    "field, value",
    [
        ("candidate_token_ids", []),
        ("candidate_token_ids", None),
        ("candidate_token_ids", "tokens"),
        ("candidate_token_ids", [True]),
        ("candidate_token_ids", [-1]),
        ("candidate_token_ids", [2**31]),
        ("candidate_token_ids", [1.0]),
        ("candidate_item_ids", []),
        ("candidate_item_ids", None),
        ("candidate_item_ids", "item"),
        ("candidate_item_ids", [""]),
        ("candidate_item_ids", ["x" * 129]),
        ("candidate_item_ids", [True]),
        ("candidate_item_ids", ["duplicate", "duplicate"]),
        ("candidate_item_ids", [f"item-{index}" for index in range(129)]),
        ("timestamp", True),
        ("timestamp", -1),
        ("timestamp", float("inf")),
        ("timestamp", float("-inf")),
        ("timestamp", float("nan")),
        ("timestamp", "1.5"),
        ("timestamp", None),
        ("visit_index", True),
        ("visit_index", -1),
        ("visit_index", 1.0),
        ("visit_index", "1"),
        ("visit_index", None),
        ("unknown_field", "unexpected"),
    ],
)
def test_flat_request_rejects_invalid_content_and_metadata(field, value):
    request = flat_request()
    request[field] = value
    with pytest.raises(ValueError):
        validate(request)


def test_flat_request_honors_custom_candidate_limit():
    with pytest.raises(ValueError, match="candidate count"):
        validate(flat_request(), max_candidates=1)
