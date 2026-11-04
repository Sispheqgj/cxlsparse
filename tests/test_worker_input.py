"""Worker prepares tokens and leaves the model's hidden-state output untouched."""

import copy
import inspect
import json
import weakref

import pytest

from cxl_recsys.common.request import placeholder
from cxl_recsys.models.base import BaseModel, CacheContext
from cxl_recsys.worker import Worker
from cxl_recsys.worker.prompt import build_prompt


class UntouchedHiddenStates:
    def __getattribute__(self, name):
        raise AssertionError(f"Worker must not inspect hidden states: {name}")

    def __getitem__(self, key):
        raise AssertionError("Worker must not select hidden-state rows")


class RecordingModel:
    max_history = 4096
    layers = 2

    def __init__(self):
        self.calls = []
        self.closed = False

    def forward(self, input_ids, positions, *, seq_indptr, caches):
        self.packed_call = (input_ids, positions, seq_indptr, caches)
        self.stats = []
        for index, cache in enumerate(caches):
            start, stop = seq_indptr[index : index + 2]
            self.calls.append((input_ids[start:stop], positions[start:stop], cache))
            self.stats.append(
                {
                    "input_tokens": stop - start,
                    "kv_length": cache.cacheable_prefix_length,
                    "cache": {"queries": (stop - start) * self.layers},
                }
            )
        hidden_states = UntouchedHiddenStates()
        self.hidden_ref = weakref.ref(hidden_states)
        return hidden_states

    def batch_forward_stats(self):
        return [stats.copy() for stats in self.stats]

    def snapshot(self):
        return {}

    def close(self):
        self.closed = True


def test_worker_passes_tokens_and_does_not_process_or_retain_hidden_states():
    model = RecordingModel()
    worker = Worker(model)
    request = placeholder(history_length=3)
    original = copy.deepcopy(request)
    try:
        assert worker.submit(request)["accepted"]
        worker.jobs.join()
        reply = worker.get({"request_id": request["request_id"]})
        assert reply["status"] == "SUCCEEDED"
        assert len(model.calls) == 1
        input_ids, positions, cache = model.calls[0]
        prepared = build_prompt(request["history_token_ids"], request["candidates"])
        assert input_ids == prepared.input_ids == [1, 2, 3, 101, 201, 102, 202]
        assert all(type(token) is int for token in input_ids)
        assert positions == list(range(7))
        assert cache == CacheContext("user-0", 3)
        assert request == original
        assert model.hidden_ref() is None
        result = reply["result"]
        assert result["request_id"] == request["request_id"]
        assert result["postprocessing"] == "not_implemented"
        assert result["prompt"] == {
            "history_tokens": 3,
            "candidate_tokens": 4,
            "input_tokens": 7,
            "last_input_position": 6,
        }
        assert result["cache"]["candidate_kv_tokens"] == 8
        assert result["cache"]["queries"] == 14
        assert "prompt_kv_tokens" not in result["cache"]
        assert (
            not {
                "hidden_states",
                "token_logits",
                "input_tokens",
                "output_position",
                "output_kind",
                "output_text",
                "output_token_ids",
                "output_tokenizer",
                "selected_candidate",
                "candidate_logits",
            }
            & result.keys()
        )
        json.dumps(reply)
    finally:
        worker.close()
    assert model.closed


def test_worker_packs_flat_and_legacy_inputs_without_passing_item_metadata_to_model():
    model = RecordingModel()
    worker = Worker(model)
    suffix = [7, 0, 2**31 - 1] * 171
    flat = {
        "request_id": "flat-request",
        "user_id": "flat-user",
        "history_token_ids": [0, 42, 42],
        "candidate_token_ids": suffix,
        "candidate_item_ids": ["item-z", "item-a"],
        "timestamp": 1.25,
        "visit_index": 4,
    }
    legacy = placeholder(history_length=2)
    batch = {"batch_id": "mixed-batch", "requests": [flat, legacy]}
    original = copy.deepcopy(batch)
    try:
        assert worker.submit_batch(batch)["accepted"]
        worker.jobs.join()
        flat_reply = worker.get({"request_id": flat["request_id"]})
        legacy_reply = worker.get({"request_id": legacy["request_id"]})
        assert flat_reply["status"] == legacy_reply["status"] == "SUCCEEDED"
        input_ids, positions, seq_indptr, caches = model.packed_call
        assert input_ids == [0, 42, 42] + suffix + [1, 2, 101, 201, 102, 202]
        assert positions == list(range(516)) + list(range(6))
        assert seq_indptr == [0, 516, 522]
        assert caches == [CacheContext("flat-user", 3), CacheContext("user-0", 2)]
        assert len(model.calls) == 2
        assert model.hidden_ref() is None
        assert batch == original
        result = flat_reply["result"]
        assert result["candidate_item_ids"] == ["item-z", "item-a"]
        assert result["timestamp"] == 1.25
        assert result["visit_index"] == 4
        assert result["prompt"] == {
            "history_tokens": 3,
            "candidate_tokens": 513,
            "input_tokens": 516,
            "last_input_position": 515,
        }
        assert result["cache"]["candidate_kv_tokens"] == 1026
        assert "candidate_item_ids" not in legacy_reply["result"]
        json.dumps(flat_reply, allow_nan=False)
    finally:
        worker.close()
    assert model.closed


@pytest.mark.gpu
def test_model_forward_returns_tensor_and_reuses_plain_token_prefix():
    import torch

    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel

    model = TestModel(MemoryPool(4, 64), max_users=1, max_history=128)
    try:
        for interface in (BaseModel, TestModel):
            assert not hasattr(interface, "infer")
            assert not hasattr(interface, "forward_batch")
            assert (
                "output_token_ids"
                not in inspect.signature(interface.forward).parameters
            )
        context = CacheContext("generic-sequence", 2)
        first = model.forward([10, 20, 30], [0, 1, 2], cache=context)
        assert isinstance(first, torch.Tensor)
        assert first.shape == (3, model.dim)
        assert first.device.type == "cuda"
        assert first.dtype == torch.bfloat16
        assert not first.requires_grad
        assert model.forward_stats()["input_tokens"] == 3
        assert model.forward_stats()["mode"] == "extend"
        assert model.users["generic-sequence"]["tokens"] == [10, 20]

        repeated = model.forward([10, 20, 30], [0, 1, 2], cache=context)
        assert repeated.shape == (1, model.dim)
        assert torch.equal(repeated, first[2:])
        assert model.forward_stats()["mode"] == "reuse"
        assert model.forward_stats()["cache"]["new_tokens"] == 0

        extended = model.forward(
            [10, 20, 40, 30],
            [0, 1, 2, 3],
            cache=CacheContext("generic-sequence", 3),
        )
        assert extended.shape == (2, model.dim)
        assert model.forward_stats()["mode"] == "extend"
        assert model.users["generic-sequence"]["tokens"] == [10, 20, 40]
        json.dumps(model.forward_stats())
        output_ref = weakref.ref(extended)
        del extended
        assert output_ref() is None  # Model diagnostics must not keep the Tensor.
    finally:
        model.close()


@pytest.mark.gpu
@pytest.mark.parametrize("invalid_input", ["token", "positions", "prefix", "cache_key"])
def test_invalid_forward_input_does_not_allocate_or_mutate(invalid_input):
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel

    model = TestModel(MemoryPool(4, 64), max_users=1, max_history=128)
    args = {
        "input_ids": [10, 20, 30],
        "positions": [0, 1, 2],
        "cache": CacheContext("generic-sequence", 2),
    }
    if invalid_input == "token":
        args["input_ids"] = [True, 20, 30]
    elif invalid_input == "positions":
        args["positions"] = [1, 2, 3]
    elif invalid_input == "prefix":
        args["cache"] = CacheContext("generic-sequence", 4)
    else:
        args["cache"] = CacheContext("", 2)
    try:
        with pytest.raises(ValueError):
            model.forward(**args)
        assert model.snapshot() == {}
        assert model.forward_stats() == {}
        assert not model.pool.allocated
        assert model.pool.write_bytes == 0
    finally:
        model.close()
