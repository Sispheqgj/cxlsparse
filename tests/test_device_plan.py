"""Compare device-resident completion with the checked compact cache path."""

import pytest
import torch

from cxl_recsys.mempool import MemoryPool
from cxl_recsys.mempool.device_plan import DeferredLayerCounters
from cxl_recsys.mempool.hisparse import HiSparseCache


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dram", "cxl"])
def test_device_plan_cold_hot_ragged_updates_and_accounting(backend, monkeypatch):
    # Anonymous registered host memory exercises the same direct-copy kernels
    # without touching a DAX device. It is not a CXL performance measurement.
    pool = MemoryPool(4, 128)
    pool.allocate(4)
    payload = torch.arange(256).bfloat16()[:, None].expand(-1, 128).contiguous()
    pool.write_many(list(range(256)), payload)
    pool.backend = backend
    reference = HiSparseCache(pool, rows=2, max_history=64, hot=32, topk=8, dim=128)
    fast = HiSparseCache(
        pool, rows=2, max_history=64, hot=32, topk=8, dim=128, device_resident=True
    )
    for cache in (reference, fast):
        cache.bind(0, list(range(64)))
        cache.bind(1, list(range(128, 192)))
    old_batch = None
    old_values = None
    try:
        for iteration in range(4):
            fetches = []
            for row, queries in ((0, 7), (1, 3)):
                ids = torch.tensor(
                    [
                        [0, 1, 2, 3, 20 + iteration, 31, 32 + q, -1]
                        for q in range(queries)
                    ],
                    device="cuda",
                )
                current = payload[row * 128 + 32 : row * 128 + 32 + queries].cuda()
                fetches.append(
                    dict(
                        row=row,
                        indices=ids,
                        history_length=35,
                        cached_length=32,
                        current_kv=current,
                    )
                )
            before = pool.read_bytes
            expected, expected_stats = reference.fetch_layer(fetches)
            assert isinstance(expected_stats, list)
            expected_bytes = pool.read_bytes - before
            before = pool.read_bytes

            def forbid_host_plan(*args, **kwargs):
                raise AssertionError("Device planning must not download or stage KV")

            with monkeypatch.context() as guard:
                guard.setattr(torch.Tensor, "tolist", forbid_host_plan)
                guard.setattr(pool, "read_many", forbid_host_plan)
                actual, deferred = fast.fetch_layer(fetches)
            assert isinstance(deferred, DeferredLayerCounters)
            stats = deferred.materialize(deferred.statistics.tolist())
            if backend == "dram":
                assert sum(r["cxl_read_kernel_launches"] for r in stats) == 0
                assert sum(r["h2d_transfers"] for r in stats) == int(
                    any(r["misses"] for r in stats)
                )
            pool.flush_read_stats()
            assert pool.read_bytes - before == expected_bytes
            assert actual.logical_positions == expected.logical_positions
            assert actual.sequence_indptr == expected.sequence_indptr
            assert actual.cached_tokens == expected.cached_tokens
            assert torch.equal(actual.indices, expected.indices)
            count = len(expected.values)
            assert torch.equal(actual.values[:count], expected.values)
            assert torch.equal(fast.tags, reference.tags)
            valid = fast.tags[:, :32] >= 0
            assert torch.equal(
                fast.data.view(2, 33, 128)[:, :32][valid],
                reference.data.view(2, 33, 128)[:, :32][valid],
            )
            for left, right in zip(stats, expected_stats):
                for key in (
                    "hits",
                    "misses",
                    "old_references",
                    "deduplicated_references",
                    "local_tokens",
                ):
                    assert left[key] == right[key]
            if old_batch is not None:
                assert old_values is not None
                assert torch.equal(old_batch.values[: len(old_values)], old_values)
            old_batch, old_values = actual, actual.values[:count].clone()
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("invalid", ["future", "negative", "unbound"])
@pytest.mark.parametrize("backend", ["dram", "cxl"])
def test_invalid_plan_defers_error_without_invalid_payload_access(invalid, backend):
    pool = MemoryPool(1, 128)
    pool.allocate(1)
    pool.backend = backend
    cache = HiSparseCache(
        pool, rows=1, max_history=64, hot=32, topk=4, dim=128, device_resident=True
    )
    cache.bind(0, [0])
    bad = {"future": 999999, "negative": -2, "unbound": 2}[invalid]
    fetches = [
        dict(
            row=0,
            indices=torch.tensor([[0, bad]], device="cuda"),
            history_length=3,
            cached_length=3,
            current_kv=torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16),
        )
    ]
    try:
        actual, deferred = cache.fetch_layer(fetches)
        assert isinstance(deferred, DeferredLayerCounters)
        with pytest.raises(ValueError, match="Top-K"):
            deferred.materialize(deferred.statistics.tolist())
        pool.flush_read_stats()
        assert pool.read_bytes == 0
        assert (actual.indices == -1).all()
    finally:
        pool.close()


@pytest.mark.gpu
def test_device_descriptor_refresh_and_temporary_only_hot_eviction():
    pool = MemoryPool(1, 128)
    pool.allocate(1)
    source = torch.arange(5).bfloat16()[:, None].expand(-1, 128).contiguous()
    pool.write_many([0, 1, 2], source[:3])
    pool.backend = "cxl"
    cache = HiSparseCache(
        pool, rows=1, max_history=8, hot=32, topk=4, dim=128, device_resident=True
    )
    cache.bind(0, [0, 1, 2])
    try:
        first = dict(
            row=0,
            cached_length=3,
            history_length=3,
            current_kv=source[3:4].cuda(),
            indices=torch.tensor([[0, 1, 2]], device="cuda"),
        )
        _, counts = cache.fetch_layer([first])
        assert isinstance(counts, DeferredLayerCounters)
        counts.materialize(counts.statistics.tolist())
        assert cache.tags[0, :3].tolist() == [0, 1, 2]
        second = dict(
            row=0,
            cached_length=3,
            history_length=3,
            current_kv=source[3:5].cuda(),
            indices=torch.tensor([[3, -1], [3, 4]], device="cuda"),
        )
        caller = torch.cuda.current_stream()
        other = torch.cuda.Stream()
        other.wait_stream(caller)
        with torch.cuda.stream(other):
            batch, counts = cache.fetch_layer([second])
            assert isinstance(counts, DeferredLayerCounters)
        caller.wait_stream(other)
        stats = counts.materialize(counts.statistics.tolist())
        assert stats[0]["misses"] == 0
        assert batch.logical_positions == [3, 4]
        assert torch.equal(batch.values[:2], source[3:5].cuda())
        assert bool((cache.tags == -1).all())
    finally:
        pool.close()
