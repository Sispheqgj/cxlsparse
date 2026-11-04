"""Shared request contract: immutable history prefix, append-only extension."""

import hashlib
import json
import math

MAX_CANDIDATES = 128
MAX_CANDIDATE_TOKENS = MAX_CANDIDATES * 256


def validate(value, max_history=4096, max_candidates=MAX_CANDIDATES):
    if not isinstance(value, dict):
        raise ValueError("Request must be an object")
    unknown = set(value) - {
        "request_id",
        "user_id",
        "history_token_ids",
        "candidates",
        "candidate_token_ids",
        "candidate_item_ids",
        "timestamp",
        "visit_index",
    }
    if unknown:
        raise ValueError(f"Unknown request fields: {sorted(unknown)}")
    for field in ("request_id", "user_id"):
        if not isinstance(value.get(field), str) or not 0 < len(value[field]) <= 128:
            raise ValueError(f"{field} must be a nonempty string, max 128 characters")

    def tokens(x, limit):
        if not isinstance(x, list) or not 0 < len(x) <= limit:
            raise ValueError(f"Token sequence must have 1..{limit} entries")
        if (set(map(type, x)) != {int} or min(x) < 0 or max(x) >= 2**31):
            raise ValueError("Token IDs must be nonnegative int32")
        return list(x)

    def item_ids(items):
        if not isinstance(items, list) or not 0 < len(items) <= max_candidates:
            raise ValueError("Invalid candidate count")
        if any(not isinstance(item, str) or not 0 < len(item) <= 128 for item in items):
            raise ValueError("Invalid item_id")
        if len(set(items)) != len(items):
            raise ValueError("Candidate item_id values must be unique")
        return list(items)

    result = {
        "request_id": value["request_id"],
        "user_id": value["user_id"],
        "history_token_ids": tokens(value.get("history_token_ids"), max_history),
    }
    flat_fields = {"candidate_token_ids", "candidate_item_ids"} & value.keys()
    if "candidates" in value:
        if flat_fields:
            raise ValueError("candidates cannot be mixed with flat candidate fields")
        candidates = value["candidates"]
        if (
            not isinstance(candidates, list)
            or not 0 < len(candidates) <= max_candidates
        ):
            raise ValueError("Invalid candidate count")
        parsed = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {
                "item_id",
                "token_ids",
            }:
                raise ValueError("Each candidate needs item_id and token_ids")
            parsed.append(
                {
                    "item_id": candidate["item_id"],
                    "token_ids": tokens(candidate["token_ids"], 256),
                }
            )
        item_ids([candidate["item_id"] for candidate in parsed])
        result["candidates"] = parsed
    else:
        result["candidate_token_ids"] = tokens(
            value.get("candidate_token_ids"), MAX_CANDIDATE_TOKENS
        )
        result["candidate_item_ids"] = item_ids(value.get("candidate_item_ids"))

    if "timestamp" in value:
        timestamp = value["timestamp"]
        valid_timestamp = type(timestamp) in (int, float) and timestamp >= 0
        if valid_timestamp:
            try:
                valid_timestamp = math.isfinite(timestamp)
            except OverflowError:
                valid_timestamp = False
        if not valid_timestamp:
            raise ValueError("timestamp must be a finite nonnegative number")
        result["timestamp"] = timestamp
    if "visit_index" in value:
        visit_index = value["visit_index"]
        if type(visit_index) is not int or visit_index < 0:
            raise ValueError("visit_index must be a nonnegative integer")
        result["visit_index"] = visit_index
    return result


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def placeholder(
    user_id="user-0", request_id="request-0", history_length=64, candidates=2
):
    return {
        "request_id": request_id,
        "user_id": user_id,
        "history_token_ids": list(range(1, history_length + 1)),
        "candidates": [
            {"item_id": f"item-{i}", "token_ids": [101 + i, 201 + i]}
            for i in range(candidates)
        ],
    }
