"""One scheduled task contains independent users in a stable request order."""

from .request import validate


def validate_batch(value, max_history=4096, max_batch_size=16):
    if (not isinstance(value, dict)
            or not {"batch_id", "requests"} <= set(value)
            or set(value) - {"batch_id", "requests", "expected_epoch"}):
        raise ValueError("Batch needs batch_id and requests, with optional expected_epoch")
    if "expected_epoch" in value and (
        not isinstance(value["expected_epoch"], str)
        or not 0 < len(value["expected_epoch"]) <= 128
    ):
        raise ValueError("expected_epoch must be a nonempty string, max 128 characters")
    batch_id = value["batch_id"]
    if not isinstance(batch_id, str) or not 0 < len(batch_id) <= 128:
        raise ValueError("batch_id must be a nonempty string, max 128 characters")
    requests = value["requests"]
    if not isinstance(requests, list) or not 0 < len(requests) <= max_batch_size:
        raise ValueError(f"Batch must contain 1..{max_batch_size} requests")
    requests = [validate(request, max_history) for request in requests]
    for field in ("request_id", "user_id"):
        if len({request[field] for request in requests}) != len(requests):
            raise ValueError(f"Batch {field} values must be unique")
    return {"batch_id": batch_id, "requests": requests,
            **({"expected_epoch": value["expected_epoch"]} if "expected_epoch" in value else {})}
