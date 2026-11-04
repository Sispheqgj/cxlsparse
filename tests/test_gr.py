import copy
import json
from pathlib import Path

import pytest
from test_batching import BatchModel

from cxl_recsys.cli import parse_args
from cxl_recsys.scheduler.gr import generate_gr, gr_request
from cxl_recsys.scheduler.workload import load_workload
from cxl_recsys.worker.inference import execute_batch
from cxl_recsys.worker.prompt import build_prompt


def gr_row(task=0, visit=0):
    # Two instruction tokens, three history tokens, and a complete long suffix.
    ids = [100001, 100002, 11, 12, 13] + list(range(300)) + [100003]
    ids[7] += visit
    return {
        "task_id": task,
        "user_id": 42,
        "visit_index": visit,
        "timestamp": task / 100,
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "instruction_tokens": 2,
        "user_tokens": 3,
        "item_tokens": 303,
        "candidate_suffix_tokens": 301,
        "total_input_tokens": 306,
        "stable_prefix_tokens": 5,
        "history_token_span": [2, 5],
        "candidate_token_span": [5, 306],
        "candidate_item_ids": [80 + visit, 90 + visit],
        # This may include shared candidate text; it must NOT define the cache.
        "common_prefix_tokens": 7 if visit else 0,
    }


@pytest.mark.parametrize("extension", ["json", "jsonl"])
def test_gr_export_replay_preserves_full_tokens_and_stable_boundary(
    tmp_path, extension
):
    rows = [gr_row(), gr_row(1, 1)]
    original = copy.deepcopy(rows)
    path = tmp_path / ("gr." + extension)
    path.write_text(
        json.dumps(rows) if extension == "json" else "\n".join(map(json.dumps, rows))
    )
    args = parse_args(["scheduler", "--input", str(path)])
    assert args.source == "gr"
    requests = load_workload(args)
    assert rows == original
    assert requests[0]["history_token_ids"] == requests[1]["history_token_ids"]
    assert requests[0]["candidate_token_ids"] != requests[1]["candidate_token_ids"]
    for row, request in zip(rows, requests):
        assert request["request_id"] == f"gr-{row['task_id']}"
        assert request["timestamp"] == row["timestamp"]
        assert request["visit_index"] == row["visit_index"]
        prepared = build_prompt(
            request["history_token_ids"],
            candidate_token_ids=request["candidate_token_ids"],
        )
        assert prepared.input_ids == row["input_ids"]
        assert prepared.positions == list(range(306))
        assert prepared.history_length == 5
        assert prepared.candidate_tokens == 301
    # Normalized --save-requests output is independently replayable.
    normalized = tmp_path / "normalized.json"
    normalized.write_text(json.dumps(requests))
    assert (
        load_workload(
            parse_args(
                ["scheduler", "--source", "requests", "--input", str(normalized)]
            )
        )
        == requests
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("stable_prefix_tokens", 3),
        ("stable_prefix_tokens", True),
        ("instruction_tokens", 1),
        ("user_tokens", 4),
        ("item_tokens", 302),
        ("candidate_suffix_tokens", 300),
        ("total_input_tokens", 305),
        ("history_token_span", [2, 4]),
        ("candidate_token_span", [5, 305]),
        ("attention_mask", [1] * 305 + [0]),
        ("attention_mask", [True] * 306),
        ("input_ids", [True] * 306),
        ("task_id", -1),
        ("user_id", True),
        ("candidate_item_ids", [1, 1]),
        ("candidate_item_ids", [True]),
        ("candidate_item_ids", []),
        ("timestamp", float("nan")),
        ("timestamp", 10**400),
        ("visit_index", -1),
    ],
)
def test_gr_rejects_inconsistent_exports(field, value):
    row = gr_row()
    row[field] = value
    with pytest.raises(ValueError):
        gr_request(row, 0, 4096)


def test_gr_history_limit_includes_instruction():
    with pytest.raises(ValueError, match="instruction 2.*user 3"):
        gr_request(gr_row(), 0, 3)


@pytest.mark.parametrize("invalid", ["timestamp", "history", "task_id"])
def test_gr_sequence_validation(tmp_path, invalid):
    rows = [gr_row(1), gr_row(2, 1)]
    if invalid == "timestamp":
        rows[1]["timestamp"] = 0
    elif invalid == "history":
        rows[1]["input_ids"][0] += 1
    else:
        rows[1]["task_id"] = rows[0]["task_id"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        load_workload(parse_args(["scheduler", "--input", str(path)]))


def test_gr_real_tokenizer_generation_and_worker_input(tmp_path):
    tokenizer = Path(parse_args(["scheduler"]).gr_tokenizer)
    if not tokenizer.is_file():
        pytest.skip("Local DeepSeek tokenizer unavailable")
    heat = tmp_path / "users.csv"
    heat.write_text("user_id,pv_share\n42,0.2\n99,0.8\n")
    args = parse_args(
        [
            "scheduler",
            "--gr-heat-source",
            "industrial",
            "--gr-heat-path",
            str(heat),
            "--gr-text-material",
            "synthetic",
            "--gr-sampling",
            "sequential",
            "--gr-arrival",
            "constant",
            "--gr-user-lengths",
            "256",
            "--gr-item-lengths",
            "512",
            "--num-requests",
            "4",
        ]
    )
    raw = list(generate_gr(args))
    requests = load_workload(args)
    assert requests == load_workload(args)
    assert [r["timestamp"] for r in requests] == [0, 0.01, 0.02, 0.03]
    assert [r["visit_index"] for r in requests] == [0, 0, 1, 1]
    for first, revisit in zip(requests[:2], requests[2:]):
        assert first["history_token_ids"] == revisit["history_token_ids"]
        assert first["candidate_token_ids"] != revisit["candidate_token_ids"]
    model = BatchModel()
    for offset in (0, 2):
        batch = requests[offset : offset + 2]
        execute_batch(model, batch)
        ids, positions, indptr, caches = model.calls[-1]
        assert indptr == [0, 768, 1536]
        for i in range(2):
            row = raw[offset + i]
            assert ids[indptr[i] : indptr[i + 1]] == row["input_ids"]
            assert positions[indptr[i] : indptr[i + 1]] == list(range(768))
            assert caches[i].cacheable_prefix_length == row["stable_prefix_tokens"]
    args.gr_user_lengths = [args.max_history]
    with pytest.raises(ValueError, match="instruction tokens exceeds"):
        load_workload(args)


@pytest.mark.parametrize("prompt_format", ["deepseek", "nosa"])
def test_default_curve_generation_needs_no_original_dataset(
    tmp_path, monkeypatch, prompt_format
):
    from GR import input_generator

    args = parse_args(["scheduler"])
    assert (args.source, args.gr_heat_source, args.gr_text_material) == (
        "gr",
        "curve",
        "synthetic",
    )
    assert args.gr_sampling == "weighted"
    args.gr_data_root = str(tmp_path / "missing-dataset")
    args.gr_prompt_format = prompt_format
    if prompt_format == "nosa":
        args.gr_tokenizer = "artifacts/models/NOSA-3B/tokenizer.json"
    if not Path(args.gr_tokenizer).is_file():
        pytest.skip("Local tokenizer unavailable")

    def forbid_original_data(*args, **kwargs):
        raise AssertionError("Default generation must use bundled heat curves")

    monkeypatch.setattr(input_generator.HeatPopulation, "load", forbid_original_data)
    monkeypatch.setattr(input_generator, "load_titles", forbid_original_data)
    raw = list(generate_gr(args))
    requests = load_workload(args)
    assert requests == load_workload(args)
    assert len(requests) == args.num_requests
    for row, request in zip(raw, requests):
        assert row["user_id"] in range(args.users)
        assert row["user_heat_weight"] > 0
        prepared = build_prompt(
            request["history_token_ids"],
            candidate_token_ids=request["candidate_token_ids"],
        )
        assert prepared.input_ids == row["input_ids"]
        assert prepared.history_length == row["stable_prefix_tokens"]
        assert len(row["input_ids"]) == 384


def test_curve_weights_drive_request_frequency_and_preserve_revisits(tmp_path):
    from collections import Counter

    args = parse_args(["scheduler"])
    if not Path(args.gr_tokenizer).is_file():
        pytest.skip("Local DeepSeek tokenizer unavailable")
    curve = tmp_path / "curve.csv"
    curve.write_text(
        "dataset,field,user_fraction,traffic_fraction\n"
        "custom,pv_share,0.5,0.9\n"
        "custom,pv_share,1,1\n"
    )
    args = parse_args(
        [
            "scheduler",
            "--gr-heat-path",
            str(curve),
            "--gr-curve-dataset",
            "custom",
            "--gr-curve-field",
            "pv_share",
            "--num-requests",
            "256",
            "--gr-arrival",
            "constant",
        ]
    )
    raw = list(generate_gr(args))
    weights = {row["user_id"]: row["user_heat_weight"] for row in raw}
    assert sorted(weights.values()) == pytest.approx([0.1, 0.9])
    hot = max(weights, key=lambda uid: weights[uid])
    counts = Counter(row["user_id"] for row in raw)
    assert 0.8 < counts[hot] / len(raw) < 0.98
    requests = load_workload(args)
    assert [r["user_id"] for r in requests] == [str(r["user_id"]) for r in raw]
    histories, previous_suffix, visits = {}, {}, Counter()
    for i, request in enumerate(requests):
        uid = request["user_id"]
        assert request["timestamp"] == i / args.gr_qps
        assert request["visit_index"] == visits[uid]
        if uid in histories:
            assert request["history_token_ids"] == histories[uid]
            assert request["candidate_token_ids"] != previous_suffix[uid]
        histories[uid] = request["history_token_ids"]
        previous_suffix[uid] = request["candidate_token_ids"]
        visits[uid] += 1
    args.gr_sampling = "uniform"
    uniform = list(generate_gr(args))
    assert 0.35 < sum(r["user_id"] == hot for r in uniform) / len(uniform) < 0.65
    args.users = 0
    with pytest.raises(ValueError, match="num_users"):
        list(generate_gr(args))
