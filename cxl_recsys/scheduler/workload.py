"""Scheduler-owned GR workloads and compatibility loaders for existing traces.

Dataset length-only records explicitly produce placeholder tokens, not guessed
tokenizer output. Existing user prefixes remain immutable across repetitions.
"""

import hashlib
import json
import math
import random
from pathlib import Path

from ..common.request import MAX_CANDIDATES, validate


def records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)
    else:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(
                "JSON workload must be an array; use JSONL for streaming records"
            )
        yield from data


def synthetic_tokens(key, length):
    if type(length) is not int or length <= 0:
        raise ValueError("Token count must be a positive integer")
    seed = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "little")
    rng = random.Random(seed)
    return [rng.randrange(1, 32000) for _ in range(length)]


def dataset_request(record, index, max_history):
    uid = str(record["user_id"])
    history = record.get("history_token_ids")
    if history is None:
        length = record["history_length"]
        if type(length) is not int or not 0 < length <= max_history:
            raise ValueError("Invalid dataset history_length")
        history = synthetic_tokens("user:" + uid, length)
    candidates = record.get("candidates")
    if candidates is None:
        ids = record["candidates_id"]
        groups = record["goods_index"]  # Bi-KV: lists of positions, NOT token IDs.
        if len(ids) != len(groups):
            raise ValueError("candidates_id and goods_index lengths differ")
        candidates = []
        for item, positions in zip(ids, groups):
            if not isinstance(positions, list) or not 0 < len(positions) <= 256:
                raise ValueError("Invalid goods_index group")
            if "input_ids" in record:
                if any(
                    type(p) is not int or not 0 <= p < len(record["input_ids"])
                    for p in positions
                ):
                    raise ValueError("goods_index position out of bounds")
                tokens = [record["input_ids"][p] for p in positions]
            else:
                tokens = synthetic_tokens("item:" + str(item), len(positions))
            candidates.append({"item_id": str(item), "token_ids": tokens})
    return validate(
        {
            "request_id": str(record.get("request_id", f"dataset-{index}")),
            "user_id": uid,
            "history_token_ids": history,
            "candidates": candidates,
        },
        max_history,
    )


def generate(
    count=8,
    users=2,
    history_length=64,
    append_tokens=0,
    candidates=2,
    seed=0,
    sampling="random",
    zipf_alpha=1.0,
):
    if (
        count <= 0
        or users <= 0
        or history_length <= 0
        or append_tokens < 0
        or not 1 <= candidates <= MAX_CANDIDATES
    ):
        raise ValueError("Invalid generated workload dimensions")
    if not math.isfinite(zipf_alpha) or zipf_alpha < 0:
        raise ValueError("zipf_alpha must be finite and nonnegative")
    rng, visits = random.Random(seed), {}
    order = []
    weights = [1 / (u + 1) ** zipf_alpha for u in range(users)]
    for i in range(count):
        if sampling == "sequential":
            uid = i % users
        elif sampling == "random":
            if not order:
                order = rng.sample(range(users), users)
            uid = order.pop(0)
        elif sampling == "zipf":
            uid = rng.choices(range(users), weights=weights, k=1)[0]
        else:
            raise ValueError("Unknown sampling mode")
        n = visits.get(uid, 0)
        visits[uid] = n + 1
        yield {
            "request_id": f"generated-{seed}-{i}",
            "user_id": f"user-{uid}",
            "history_token_ids": synthetic_tokens(
                f"{seed}:user:{uid}", history_length + n * append_tokens
            ),
            "candidates": [
                {
                    "item_id": f"item-{j}",
                    "token_ids": synthetic_tokens(f"{seed}:item:{j}", 2),
                }
                for j in range(candidates)
            ],
        }


def load_workload(args):
    if args.source == "gr":
        from .gr import generate_gr, gr_request

        rows = records(args.input) if args.input else generate_gr(args)
        values = (gr_request(r, i, args.max_history) for i, r in enumerate(rows))
    elif args.source in ("requests", "dataset"):
        if not args.input:
            raise ValueError("--input is required for requests/dataset source")
        values = records(args.input)
        if args.source == "dataset":
            values = (
                dataset_request(r, i, args.max_history) for i, r in enumerate(values)
            )
    else:
        if args.input:
            raise ValueError("--input is not used for generated workloads")
        # Validate worst-case size before allocating token lists.
        if (
            args.history_length + max(0, args.num_requests - 1) * args.append_tokens
            > args.max_history
        ):
            raise ValueError("Generated history could exceed --max-history")
        values = generate(
            args.num_requests,
            args.users,
            args.history_length,
            args.append_tokens,
            args.candidates,
            args.seed,
            args.sampling,
            args.zipf_alpha,
        )
    requests, seen, histories = [], set(), {}
    previous_timestamp = 0.0
    for value in values:
        if len(requests) >= 10000:
            raise ValueError(
                "Workload exceeds service lifetime record capacity (10000)"
            )
        request = validate(value, args.max_history)
        timestamp = request.get("timestamp", previous_timestamp)
        if timestamp < previous_timestamp:
            raise ValueError("Workload timestamps must be nondecreasing")
        previous_timestamp = timestamp
        rid, uid, history = (
            request["request_id"],
            request["user_id"],
            request["history_token_ids"],
        )
        if rid in seen:
            raise ValueError(f"Duplicate request_id in sequence: {rid}")
        old = histories.get(uid, [])
        if len(history) < len(old) or history[: len(old)] != old:
            raise ValueError(f"Dataset history is not append-only for {uid}")
        seen.add(rid)
        histories[uid] = history
        requests.append(request)
    if not requests:
        raise ValueError("Empty workload")
    return requests
