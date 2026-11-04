"""Real HiSparse with DSA layer-order, causality and in-place update assertions."""

import pytest
import torch

from cxl_recsys.common.request import placeholder
from cxl_recsys.mempool import MemoryPool
from cxl_recsys.models.simulation.model import TestModel
from cxl_recsys.worker.inference import execute_request


@pytest.mark.gpu
def test_layer_writes_then_all_queries_before_next_layer(monkeypatch):
    pool = MemoryPool(8, 64)
    model = TestModel(pool, max_users=1, max_history=128)
    events = []
    selections = []
    select = model.ops.select_batch

    def select_layer(positions, hidden, qr, layer, sequences):
        selected = select(positions, hidden, qr, layer, sequences)
        selections.append((positions.cpu(), selected.cpu()))
        return selected

    def trace(stage, **fields):
        if stage == "embed":
            assert model.snapshot() == {}
        else:
            assert model.snapshot()["user-0"]["kv_length"] == 3
        events.append((stage, fields))

    monkeypatch.setattr(model, "event", trace)
    monkeypatch.setattr(model.ops, "select_batch", select_layer)
    try:
        request = placeholder(history_length=3)
        result = execute_request(model, request)
        queries = result["prompt"]["input_tokens"]
        names = [stage for stage, _ in events]
        assert names[0] == "embed"
        assert names[-2:] == ["final_norm", "forward_finished"]
        assert names.count("final_norm") == 1
        assert names.index("layer_finished") < names.index("layer_started", 2)
        for layer in range(2):
            block = [stage for stage, f in events if f.get("layer") == layer]
            assert block[:4] == [
                "layer_started",
                "q_kv_projected",
                "mla_cache_written",
                "query_absorbed",
            ]
            assert block[4:-2] == [
                "index_cache_written",
                "index_topk",
                "attention_consumed",
            ]
            assert block[-2:] == ["ffn_finished", "layer_finished"]
        assert len(selections) == model.layers
        for positions, selected in selections:
            assert positions.tolist() == list(range(queries))
            for position, indices in zip(positions.tolist(), selected.tolist()):
                valid = [index for index in indices if index >= 0]
                assert all(index <= position for index in valid)
                assert len(valid) == min(8, position + 1)
            assert positions[-1] == result["prompt"]["last_input_position"]
        assert result["cache"]["queries"] == queries * 2
        assert result["cache"]["hits"] == result["cache"]["misses"] == 0
        assert result["cache"]["new_pages"] == 2
        assert result["cache"]["new_tokens"] == 6
        assert model.snapshot()["user-0"]["kv_length"] == 3
    finally:
        model.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "old_length,new_length", [(0, 65), (63, 64), (63, 65), (65, 65)]
)
@pytest.mark.parametrize("failure", ["index", "consume", "final_norm"])
def test_layer_failure_leaves_live_state_and_propagates(
    monkeypatch, old_length, new_length, failure
):
    pool = MemoryPool(8, 64)
    model = TestModel(pool, max_users=1, max_history=128)
    try:
        if old_length:
            execute_request(model, placeholder(history_length=old_length))
        model.cache.invalidate()
        current_layer = -1

        def trace(stage, **fields):
            nonlocal current_layer
            if stage == "layer_started":
                current_layer = fields["layer"]

        operator = {
            "index": "select_batch",
            "consume": "consume",
            "final_norm": "final_norm",
        }[failure]
        original = getattr(model.ops, operator)

        def fail(*args, **kwargs):
            if current_layer == 1:
                raise OSError("second-layer or final normalization failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(model, "event", trace)
        monkeypatch.setattr(model.ops, operator, fail)
        with pytest.raises(OSError, match="second-layer"):
            execute_request(model, placeholder(history_length=new_length))
        assert model.snapshot()["user-0"]["kv_length"] == new_length
        assert model.forward_stats() == {}
        assert len(pool.allocated) == 2 * ((new_length + 63) // 64)
        assert all(len(slots) == new_length for slots in model.users["user-0"]["slots"])
        assert bool((model.cache.host_locs[:, :new_length] >= 0).all())
        # No retry: these live tables may describe only partially completed work.
    finally:
        model.close()


@pytest.mark.gpu
def test_second_layer_pool_read_failure_keeps_partial_live_state(monkeypatch):
    pool = MemoryPool(8, 64)
    model = TestModel(pool, max_users=1, max_history=128)
    try:
        execute_request(model, placeholder(history_length=63))
        model.cache.invalidate()
        current_layer = -1

        def trace(stage, **fields):
            nonlocal current_layer
            if stage == "layer_started":
                current_layer = fields["layer"]

        read = pool.read_many

        def fail(slot, tensor):
            if current_layer == 1:
                raise OSError("second-layer read failure")
            read(slot, tensor)

        # Force old-prefix I/O after the live history and slot tables are extended.
        monkeypatch.setattr(
            model.ops,
            "select_batch",
            lambda p, hidden, qr, layer, sequences: torch.zeros(
                (len(p), 1), device=p.device, dtype=torch.int64
            ),
        )
        monkeypatch.setattr(model, "event", trace)
        monkeypatch.setattr(pool, "read_many", fail)
        with pytest.raises(OSError, match="read failure"):
            execute_request(model, placeholder(history_length=65))
        assert model.snapshot()["user-0"]["kv_length"] == 65
        assert all(len(slots) == 65 for slots in model.users["user-0"]["slots"])
        assert len(pool.allocated) == 4
        # Planning no longer publishes hot tags before backing IO finishes.
        # Model history/page tables are still live; failure invalidates this run.
        assert model.cache.planner_calls > 0
        assert model.forward_stats() == {}
    finally:
        model.close()
