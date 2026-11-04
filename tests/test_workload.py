import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cxl_recsys.common.request import placeholder, validate
from cxl_recsys.scheduler.workload import dataset_request, load_workload


def config(**overrides):
    return SimpleNamespace(
        **{
            **dict(
                source="generated",
                input=None,
                max_history=256,
                num_requests=6,
                users=2,
                history_length=3,
                append_tokens=1,
                candidates=2,
                seed=42,
                sampling="random",
                zipf_alpha=1.0,
            ),
            **overrides,
        }
    )


@pytest.mark.parametrize("sampling", ["sequential", "random", "zipf"])
def test_generated_reproducible_append_only(sampling):
    a = config(sampling=sampling)
    first = load_workload(a)
    assert first == load_workload(a)
    histories = {}
    for request in first:
        uid, history = request["user_id"], request["history_token_ids"]
        if uid in histories:
            assert len(history) == len(histories[uid]) + 1
            assert history[:-1] == histories[uid]
        histories[uid] = history


def test_requests_jsonl_and_validation(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = [
        placeholder(request_id="a", history_length=3),
        placeholder(request_id="b", history_length=4),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert load_workload(config(source="requests", input=str(path))) == rows
    rows[1]["history_token_ids"][0] = 999
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="append-only"):
        load_workload(config(source="requests", input=str(path)))


def test_128_single_token_candidates_survive_request_loading(tmp_path):
    request = placeholder(history_length=256, candidates=128)
    for candidate in request["candidates"]:
        candidate["token_ids"] = candidate["token_ids"][:1]
    assert validate(request) == request
    path = tmp_path / "large-request.json"
    path.write_text(json.dumps([request]))
    assert load_workload(config(source="requests", input=str(path))) == [request]
    generated = load_workload(config(candidates=128))
    assert all(len(item["candidates"]) == 128 for item in generated)
    with pytest.raises(ValueError, match="candidate count"):
        validate(placeholder(candidates=129))
    with pytest.raises(ValueError, match="dimensions"):
        load_workload(config(candidates=129))


def test_bikv_positions_not_tokens():
    row = {
        "user_id": 7,
        "history_length": 3,
        "candidates_id": [42],
        "goods_index": [[2, 3]],
        "input_ids": [99, 98, 10, 11],
    }
    r = dataset_request(row, 0, 256)
    assert r["user_id"] == "7"
    assert r["candidates"] == [{"item_id": "42", "token_ids": [10, 11]}]
    extended = dataset_request({**row, "history_length": 4}, 1, 256)
    assert extended["history_token_ids"][:3] == r["history_token_ids"]


def test_no_client_cli():
    result = subprocess.run(
        [sys.executable, "-m", "cxl_recsys.cli", "client"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
