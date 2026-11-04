"""Physical-span coalescing and layer-wide HiSparse gathering regressions."""

from types import SimpleNamespace

import pytest
import torch
from kv_helpers import LengthOnlySlots

from cxl_recsys.common.request import placeholder
from cxl_recsys.mempool import MemoryPool


def test_batch_io_merges_across_pages_and_preserves_order_and_gaps():
    pool = MemoryPool(3, 4)
    pool.allocate(3)
    slots = [130, 63, 64, 129, 62, 0]
    source = torch.arange(24).reshape(6, 4).bfloat16()
    target = torch.empty_like(source)
    try:
        pool.write_many(slots, source)
        pool.read_many(slots, target)
        assert torch.equal(source, target)
        # [0], [62..64] (crosses a page), [129..130]; never transfer the gaps.
        assert pool.write_operations == pool.read_operations == 3
        assert pool.write_bytes == pool.read_bytes == source.numel() * 2
        gap = torch.empty(4, dtype=torch.bfloat16)
        pool.read_into(65, gap)
        assert bool((gap == 0).all())
    finally:
        pool.close()


@pytest.mark.parametrize(
    "invalid",
    ["shape", "dtype", "noncontiguous"],
)
def test_invalid_batch_tensor_is_rejected_before_any_io(invalid):
    pool = MemoryPool(2, 4)
    pool.allocate(1)
    slots = [0, 1]
    data = torch.zeros(2, 4, dtype=torch.bfloat16)
    if invalid == "shape":
        data = data.reshape(4, 2)
    elif invalid == "dtype":
        data = data.float()
    else:
        data = torch.zeros(2, 8, dtype=torch.bfloat16)[:, ::2]
    try:
        with pytest.raises(ValueError, match="Batch IO"):
            pool.write_many(slots, data)
        with pytest.raises(ValueError, match="Batch IO"):
            pool.read_many(slots, data)
        assert pool.write_bytes == pool.read_bytes == 0
        assert pool.write_operations == pool.read_operations == 0
    finally:
        pool.close()


@pytest.mark.parametrize("slots", [[0, 0], [0, 64], [0, -1], [0, 1.5], [0, True]])
def test_invalid_cpu_read_slots_still_rejected_before_io(slots):
    pool = MemoryPool(2, 4)
    pool.allocate(1)
    try:
        with pytest.raises(ValueError, match="Batch IO"):
            pool.read_many(slots, torch.empty((2, 4), dtype=torch.bfloat16))
        assert pool.read_bytes == pool.read_operations == 0
    finally:
        pool.close()


@pytest.mark.parametrize("count", [0, 2, 4096])
def test_batch_tensor_validation_never_scans_slots(count):
    pool = MemoryPool(1, 4)
    try:
        # No pages are allocated: only the vector length may be inspected here.
        slots = LengthOnlySlots(count)
        data = torch.empty((count, 4), dtype=torch.bfloat16)
        pool._validate_batch(slots, data, device_type="cpu")
    finally:
        pool.close()


def test_empty_batch_performs_no_transfer():
    pool = MemoryPool(1, 4)
    try:
        empty = torch.empty((0, 4), dtype=torch.bfloat16)
        pool.write_many([], empty)
        pool.read_many([], empty)
        assert pool.write_operations == pool.read_operations == 0
    finally:
        pool.close()


def test_contiguous_io_splits_only_at_registered_staging_capacity(monkeypatch):
    pool = MemoryPool(1, 4)
    pool.allocate(1)
    storage = torch.zeros((64, 4), dtype=torch.bfloat16)
    transfers = []

    def transfer(offset, tensor, write):
        transfers.append((offset, len(tensor), write))
        start = offset // pool.token_bytes
        if write:
            storage[start : start + len(tensor)] = tensor
        else:
            tensor.copy_(storage[start : start + len(tensor)])

    monkeypatch.setattr(
        pool,
        "remote",
        SimpleNamespace(
            region=SimpleNamespace(
                endpoint=SimpleNamespace(size=3 * pool.token_bytes), transfer=transfer
            )
        ),
    )
    try:
        source = torch.arange(28).reshape(7, 4).bfloat16()
        target = torch.empty_like(source)
        pool.write_many(list(range(7)), source)
        pool.read_many(list(range(7)), target)
        assert torch.equal(source, target)
        assert [n for _, n, _ in transfers] == [3, 3, 1, 3, 3, 1]
    finally:
        pool.remote = None
        pool.close()


@pytest.mark.gpu
def test_layer_union_larger_than_hot_is_read_once_and_survives_eviction(monkeypatch):
    from cxl_recsys.mempool.hisparse import HiSparseCache

    pool = MemoryPool(2, 4)
    pool.allocate(2)
    source = torch.arange(128 * 4).reshape(128, 4).bfloat16()
    pool.write_many(list(range(128)), source)
    cache = HiSparseCache(pool, rows=2, max_history=128, hot=32, topk=8, dim=4)
    cache.bind(1, list(range(128)))
    # Resident tokens near the tail would be evicted by a naive sorted union.
    cache.fetch(1, list(range(72, 80)), 128)
    first = torch.arange(80, device="cuda").reshape(10, 8)
    selected = torch.cat((first, first.flip(0)))  # 160 refs, 80 unique tokens.
    current = torch.zeros((20, 4), device="cuda", dtype=torch.bfloat16)
    reads = []
    read_many = pool.read_many

    def read_once(slots, tensor):
        reads.append(list(slots))
        read_many(slots, tensor)

    monkeypatch.setattr(pool, "read_many", read_once)
    before_bytes, before_ops = pool.read_bytes, pool.read_operations
    try:
        batch, layer_counters = cache.fetch_layer(
            [
                {
                    "row": 1,
                    "indices": selected,
                    "history_length": 128,
                    "cached_length": 128,
                    "current_kv": current,
                }
            ]
        )
        assert isinstance(layer_counters, list)
        (stats,) = layer_counters
        expected = source[selected.cpu()]
        assert torch.equal(batch.values[batch.indices].cpu(), expected)
        assert len(reads) == 1 and sorted(reads[0]) == list(range(72))
        assert pool.read_bytes - before_bytes == 72 * pool.token_bytes
        assert pool.read_operations - before_ops == 1
        assert stats["hits"] == 8 and stats["misses"] == 72
        assert stats["old_references"] == 160
        assert stats["deduplicated_references"] == 80
        assert stats["h2d_transfers"] == 1
        assert len(cache.tags[1, :32]) == 32
        assert bool((cache.tags[0] == -1).all())  # Another row is untouched.
        # A later fetch may replace all hot slots, but cannot corrupt the layer buffer.
        cache.fetch(1, list(range(100, 108)), 128)
        assert torch.equal(batch.values[batch.indices].cpu(), expected)
        valid_tags = cache.tags[1, :32].cpu()
        valid = valid_tags >= 0
        assert valid_tags[valid].tolist() == list(range(100, 108))
        locations = cache.locs[1, :32].cpu()[valid].long().cuda()
        assert torch.equal(
            cache.data[locations].cpu(), source[valid_tags[valid].long()]
        )
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("invalid", ["future", "padding", "all_padding"])
def test_invalid_layer_topk_does_not_touch_pool_or_publish_tags(invalid):
    from cxl_recsys.mempool.hisparse import HiSparseCache

    pool = MemoryPool(1, 4)
    pool.allocate(1)
    cache = HiSparseCache(pool, rows=1, max_history=64, hot=4, topk=2, dim=4)
    cache.bind(0, list(range(4)))
    selected = torch.tensor([[0, 4], [1, 5]], device="cuda")
    if invalid == "future":
        selected[0, 1] = 5
    elif invalid == "padding":
        selected[0, 1] = -2
    else:
        selected[0] = -1
    try:
        with pytest.raises(ValueError, match="Top-K"):
            cache.fetch_layer(
                [
                    {
                        "row": 0,
                        "indices": selected,
                        "history_length": 4,
                        "cached_length": 4,
                        "current_kv": torch.zeros(
                            (2, 4), device="cuda", dtype=torch.bfloat16
                        ),
                    }
                ]
            )
        assert pool.read_bytes == 0
        assert bool((cache.tags == -1).all())
    finally:
        pool.close()


@pytest.mark.gpu
def test_layer_model_calls_write_index_fetch_consume_once_per_layer(monkeypatch):
    from cxl_recsys.models.simulation.model import TestModel
    from cxl_recsys.worker.inference import execute_request

    pool = MemoryPool(8, 64)
    model = TestModel(pool, max_users=1, max_history=256)
    calls = []
    for owner, name in (
        (pool, "write_many"),
        (model.ops, "select_batch"),
        (model.cache, "fetch_layer"),
        (model.ops, "consume"),
    ):
        original = getattr(owner, name)

        def record(*args, _name=name, _call=original, **kwargs):
            calls.append(_name)
            return _call(*args, **kwargs)

        monkeypatch.setattr(owner, name, record)
    try:
        result = execute_request(model, placeholder(history_length=128))
        assert calls == ["write_many", "select_batch", "fetch_layer", "consume"] * 2
        assert result["pool"]["write_operations"] == 2
        assert result["pool"]["write_bytes"] == 128 * 2 * 128
        calls.clear()
        reuse = execute_request(model, placeholder(history_length=128))
        assert calls == ["select_batch", "fetch_layer", "consume"] * 2
        assert reuse["pool"]["write_operations"] == 2
        assert reuse["cache"]["h2d_transfers"] <= 2
    finally:
        model.close()


@pytest.mark.gpu
def test_layer_mixes_old_and_current_kv_without_reading_padding_or_new_history():
    from cxl_recsys.mempool.hisparse import HiSparseCache

    pool = MemoryPool(1, 4)
    pool.allocate(1)
    source = torch.arange(16).reshape(4, 4).bfloat16()
    pool.write_many(list(range(4)), source)
    cache = HiSparseCache(pool, rows=1, max_history=64, hot=4, topk=4, dim=4)
    cache.bind(0, list(range(5)))
    current = torch.full((3, 4), 77, device="cuda", dtype=torch.bfloat16)
    selected = torch.tensor(
        [[0, 4, -1, -1], [0, 4, 5, -1], [1, 4, 5, 6]], device="cuda"
    )
    try:
        batch, layer_counters = cache.fetch_layer(
            [
                {
                    "row": 0,
                    "indices": selected,
                    "history_length": 5,
                    "cached_length": 4,
                    "current_kv": current,
                }
            ]
        )
        assert isinstance(layer_counters, list)
        (stats,) = layer_counters
        full = torch.cat((source, current.cpu()))
        valid = selected >= 0
        assert torch.equal(
            batch.values[batch.indices[valid]].cpu(), full[selected[valid].cpu()]
        )
        assert torch.equal(batch.indices == -1, selected == -1)
        assert batch.logical_positions == [0, 1, 4, 5, 6]
        assert pool.read_bytes == 2 * pool.token_bytes
        assert pool.read_operations == 1
        assert stats["old_references"] == 3
        assert stats["deduplicated_references"] == 1
        assert stats["local_tokens"] == 6
        assert pool.write_operations == 1
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("failure", ["plan", "io"])
def test_large_layer_failure_propagates_without_retry(monkeypatch, failure):
    from cxl_recsys.mempool.hisparse import HiSparseCache

    pool = MemoryPool(2, 4)
    pool.allocate(2)
    source = torch.arange(512).reshape(128, 4).bfloat16()
    pool.write_many(list(range(128)), source)
    cache = HiSparseCache(pool, rows=1, max_history=128, hot=32, topk=8, dim=4)
    cache.bind(0, list(range(128)))
    cache.fetch(0, list(range(72, 80)), 128)
    selections = torch.arange(0, 80, 2, device="cuda").reshape(5, 8)
    current = torch.zeros((5, 4), device="cuda", dtype=torch.bfloat16)
    owner, name = (
        (cache, "_plan_layer") if failure == "plan" else (pool, "_transfer_span")
    )
    original = getattr(owner, name)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == (1 if failure == "plan" else 2):
            raise OSError("layer batch failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, fail)
    try:
        with pytest.raises(OSError, match="layer batch failure"):
            cache.fetch_layer(
                [
                    {
                        "row": 0,
                        "indices": selections,
                        "history_length": 128,
                        "cached_length": 128,
                        "current_kv": current,
                    }
                ]
            )
        assert bool((cache.tags != -1).any())
        # The failed run must be discarded, not invalidated and retried.
    finally:
        pool.close()


@pytest.mark.gpu
def test_batched_consumer_matches_per_query_reference():
    from cxl_recsys.models.simulation.backbone import DSAStub

    ops = DSAStub(64, 8)
    packed = ops.embed(list(range(8)))
    q = ops.norm(packed)
    indices = torch.full((8, 8), -1, device="cuda", dtype=torch.int64)
    expected = torch.empty_like(packed)
    for row in range(8):
        indices[row, : row + 1] = torch.arange(row + 1, device="cuda")
        key = packed[: row + 1, : ops.latent_dim].float()
        positional = packed[: row + 1, ops.latent_dim :].float()
        latent_output = key.mean(dim=0) + q[row, : ops.latent_dim].float() / 8
        latent_output += key.mean(dim=0) / 16
        positional_output = (
            positional.mean(dim=0) + q[row, ops.latent_dim :].float() / 8
        )
        expected[row] = torch.cat((latent_output, positional_output)).bfloat16()
    args = (q[:, : ops.latent_dim], q[:, ops.latent_dim :], packed, indices)
    assert torch.equal(ops.consume(*args), expected)
