"""Real upstream sparse-copy kernel tests on registered anonymous mappings.

These exercise CUDA and slot semantics without claiming to use physical CXL.
The end-to-end devdax smoke separately exercises the actual device mapping.
"""

import pytest
import torch

from cxl_recsys.common.request import placeholder
from cxl_recsys.mempool import MemoryPool
from cxl_recsys.mempool.cuda_transfer import RegisteredCudaMapping
from cxl_recsys.mempool.hisparse import HiSparseCache
from cxl_recsys.models.base import CacheContext
from cxl_recsys.models.simulation.model import TestModel
from cxl_recsys.worker.prompt import build_prompt


def forbid_staging(*args, **kwargs):
    raise AssertionError("CXL KV must not pass through CPU staging")


@pytest.fixture
def pool():
    instance = MemoryPool(4, 64)
    instance.backend = "cxl"
    instance.allocate(1)
    yield instance
    instance.close()


@pytest.mark.gpu
@pytest.mark.parametrize("dim", [64, 576])
def test_sparse_read_first_scattered_rows_bitwise_cross_page(dim, monkeypatch):
    pool = MemoryPool(4, dim)
    pool.backend = "cxl"
    pool.allocate(4)
    # Include all kinds of BF16 bit patterns, including NaNs. Comparisons below
    # operate on int16 bits, not floating-point equality or a runtime KV digest.
    source = (torch.arange(256 * dim) * 7).to(torch.int16).view(torch.bfloat16)
    source = source.reshape(256, dim)
    pool.write_many(list(range(256)), source)
    slots = list(range(62, 194))[::-1] + [0, 255]
    rows = list(range(0, len(slots) * 2, 2))[::-1]
    expected = torch.full((len(slots) * 2 + 1, dim), -99, dtype=torch.bfloat16)
    expected[rows] = source[slots]
    stream = torch.cuda.Stream()
    try:
        assert pool.cuda_mapping is None
        with torch.cuda.stream(stream):
            target = torch.full_like(expected, -99, device="cuda")
            with monkeypatch.context() as guard:
                guard.setattr(pool, "read_many", forbid_staging)
                guard.setattr(pool, "_transfer_span", forbid_staging)
                guard.setattr(torch.Tensor, "cpu", forbid_staging)
                pool.read_cuda(slots, target, rows)
            assert stream.query()
        assert torch.equal(target.cpu().view(torch.int16), expected.view(torch.int16))
        assert pool.read_bytes == len(slots) * dim * 2
        assert pool.read_operations == pool.read_kernel_launches == 1
        # A read-first registration is reused for a later direct write and read.
        registered = pool.cuda_mapping
        pool.write_many(
            [255], torch.full((1, dim), 7, device="cuda", dtype=torch.bfloat16)
        )
        pool.read_cuda([255], target)
        assert bool((target[0] == 7).all())
        assert pool.cuda_mapping is registered
        pool.close()
        assert registered is not None and registered.view is None
        with pytest.raises(RuntimeError, match="closed"):
            pool.read_cuda([0], target)
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate_source",
        "negative_source",
        "unallocated",
        "capacity",
        "float_source",
        "duplicate_destination",
        "negative_destination",
        "destination_bounds",
        "destination_int32",
        "destination_length",
        "float_destination",
        "shape",
        "dtype",
        "noncontiguous",
        "cpu",
        "backend",
    ],
)
def test_bad_read_rejected_before_registration_or_launch(pool, invalid):
    slots, rows = [0, 1], [0, 1]
    target = torch.zeros((2, 64), dtype=torch.bfloat16, device="cuda")
    if invalid == "duplicate_source":
        slots = [0, 0]
    elif invalid == "negative_source":
        slots = [0, -1]
    elif invalid == "unallocated":
        slots = [0, 64]
    elif invalid == "capacity":
        slots = [0, 256]
    elif invalid == "float_source":
        slots = [0, 1.5]
    elif invalid == "duplicate_destination":
        rows = [0, 0]
    elif invalid == "negative_destination":
        rows = [0, -1]
    elif invalid == "destination_bounds":
        rows = [0, 2]
    elif invalid == "destination_int32":
        rows = [0, 2**31]
    elif invalid == "destination_length":
        rows = [0]
    elif invalid == "float_destination":
        rows = [0, 1.5]
    elif invalid == "shape":
        target = target.reshape(1, 128)
    elif invalid == "dtype":
        target = target.float()
    elif invalid == "noncontiguous":
        target = torch.zeros((2, 128), dtype=torch.bfloat16, device="cuda")[:, ::2]
    elif invalid == "cpu":
        target = target.cpu()
    else:
        pool.backend = "rdma"
    with pytest.raises(ValueError):
        pool.read_cuda(slots, target, rows)
    assert pool.cuda_mapping is None
    assert pool.read_bytes == pool.read_operations == pool.read_kernel_launches == 0


@pytest.mark.gpu
def test_empty_read_does_not_register(pool):
    pool.read_cuda([], torch.empty((0, 64), dtype=torch.bfloat16, device="cuda"))
    assert pool.cuda_mapping is None
    assert pool.read_kernel_launches == pool.read_bytes == 0


@pytest.mark.gpu
@pytest.mark.parametrize("dim", [64, 576])
def test_whole_layer_cxl_misses_one_kernel_despite_hot_evictions(dim, monkeypatch):
    pool = MemoryPool(2, dim)
    pool.backend = "cxl"
    pool.allocate(2)
    source = torch.arange(128 * dim).reshape(128, dim).bfloat16()
    pool.write_many(list(range(128)), source)
    cache = HiSparseCache(pool, rows=2, max_history=128, hot=32, topk=8, dim=dim)
    cache.bind(1, list(range(128)))
    cache.fetch(1, list(range(72, 80)), 128)
    selected = torch.arange(80, device="cuda").reshape(10, 8).repeat(2, 1)
    current = torch.full((20, dim), -7, device="cuda", dtype=torch.bfloat16)
    # Mix local KV into the same layer and preserve the large old working set.
    selected = torch.cat(
        (selected, torch.tensor([[128, 129, -1, -1, -1, -1, -1, -1]], device="cuda"))
    )
    current = torch.cat((current, current[:1]))
    reads = []
    read_cuda = pool.read_cuda

    def record(slots, target, rows=None, **kwargs):
        counts = kwargs["counts"].tolist()
        reads.append(
            [
                slot
                for i, count in enumerate(counts)
                for slot in slots[i, :count].tolist()
            ]
        )
        return read_cuda(slots, target, rows, **kwargs)

    try:
        before = pool.read_kernel_launches
        with monkeypatch.context() as guard:
            guard.setattr(pool, "read_many", forbid_staging)
            guard.setattr(pool, "_transfer_span", forbid_staging)
            guard.setattr(torch.Tensor, "cpu", forbid_staging)
            guard.setattr(pool, "read_cuda", record)
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
        assert len(reads) == 1 and sorted(reads[0]) == list(range(72))
        assert pool.read_kernel_launches - before == 1
        assert stats["hits"] == 8 and stats["misses"] == 72
        assert stats["h2d_transfers"] == 0
        assert stats["cxl_read_kernel_launches"] == 1
        assert stats["deduplicated_references"] == 80
        assert stats["local_tokens"] == 2
        assert torch.equal(batch.values[:80].cpu(), source[:80])
        assert torch.equal(batch.values[80:], current[:2])
        assert torch.equal(
            batch.values[batch.indices[:20]].cpu(), source[selected[:20].cpu()]
        )
        assert bool((cache.tags[0] == -1).all())
        # The compact attention buffer survives a subsequent hot-set replacement.
        cache.fetch(1, list(range(100, 108)), 128)
        assert torch.equal(batch.values[:80].cpu(), source[:80])
        tags = cache.tags[1, :32].long()
        valid = tags >= 0
        assert tags[valid].tolist() == list(range(100, 108))
        assert torch.equal(
            cache.data[cache.locs[1, :32][valid].long()].cpu(),
            source[tags[valid].cpu()],
        )
        before = pool.read_kernel_launches
        hits = torch.arange(100, 108, device="cuda").reshape(1, 8)
        _, layer_counters = cache.fetch_layer(
            [
                {
                    "row": 1,
                    "indices": hits,
                    "history_length": 128,
                    "cached_length": 128,
                    "current_kv": current[:1],
                }
            ]
        )
        assert isinstance(layer_counters, list)
        (stats,) = layer_counters
        assert (
            stats["misses"]
            == stats["h2d_transfers"]
            == stats["cxl_read_kernel_launches"]
            == 0
        )
        assert pool.read_kernel_launches == before
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("dim", [64, 576])
def test_cxl_and_dram_forward_hidden_states_equal_without_staging(dim, monkeypatch):
    pool = MemoryPool(16, dim)
    pool.backend = "cxl"
    model = TestModel(pool, max_users=1, max_history=256, dim=dim)
    reference = TestModel(MemoryPool(16, dim), max_users=1, max_history=256, dim=dim)
    try:
        for length in (3, 65, 128, 128):
            request = placeholder(history_length=length)
            prepared = build_prompt(request["history_token_ids"], request["candidates"])
            context = CacheContext(request["user_id"], prepared.history_length)
            expected = reference.forward(
                prepared.input_ids, prepared.positions, cache=context
            )
            with monkeypatch.context() as guard:
                guard.setattr(pool, "read_many", forbid_staging)
                guard.setattr(pool, "_transfer_span", forbid_staging)
                guard.setattr(torch.Tensor, "cpu", forbid_staging)
                actual = model.forward(
                    prepared.input_ids, prepared.positions, cache=context
                )
            assert torch.equal(actual, expected)
            stats = model.forward_stats()["cache"]
            assert stats["h2d_transfers"] == 0
            assert stats["cxl_read_kernel_launches"] <= model.layers
        assert pool.read_kernel_launches > 0
    finally:
        model.close()
        reference.close()


@pytest.mark.gpu
@pytest.mark.parametrize("failure", ["registration", "kernel"])
def test_read_failure_propagates_without_staging_or_completed_counters(
    pool, monkeypatch, failure
):
    def fail(*args, **kwargs):
        raise RuntimeError("injected direct read failure")

    if failure == "registration":
        monkeypatch.setattr(RegisteredCudaMapping, "__init__", fail)
    else:
        monkeypatch.setattr(RegisteredCudaMapping, "read", fail)
    monkeypatch.setattr(pool, "read_many", forbid_staging)
    monkeypatch.setattr(pool, "_transfer_span", forbid_staging)
    target = torch.empty((2, 64), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="injected direct read failure"):
        pool.read_cuda([0, 1], target)
    assert pool.read_bytes == pool.read_operations == pool.read_kernel_launches == 0


@pytest.mark.gpu
def test_dram_direct_write_scattered_pages_and_accounting(monkeypatch):
    pool = MemoryPool(5, 128, page_size=16)
    try:
        pages = pool.allocate(5)
        pages = [pages[i] for i in (3, 0, 4, 1, 2)]
        slots = pool.token_slots(pages, 13, 67)
        assert slots == [pool.token_slot(pages[i // 16], i % 16) for i in range(13, 67)]
        payload = torch.randn(len(slots), 128, device="cuda", dtype=torch.bfloat16)
        expected = payload.cpu()
        with monkeypatch.context() as guard:
            guard.setattr(pool, "_transfer_many", forbid_staging)
            guard.setattr(torch.Tensor, "cpu", forbid_staging)
            pool.write_many(slots, payload)
        actual = torch.empty_like(expected)
        pool.read_many(slots, actual)
        assert torch.equal(actual, expected)
        assert pool.write_bytes == len(slots) * 256
        assert pool.write_operations == pool.write_kernel_launches == 1
        assert pool.token_slots(pages, 80, 80) == []
        with pytest.raises(ValueError):
            pool.token_slots(pages, 0, 81)
    finally:
        pool.close()


@pytest.mark.gpu
def test_registered_dram_async_write_then_device_read(monkeypatch):
    """Stream-ordered persistent writes must not fall back to CPU KV staging."""
    pool = MemoryPool(4, 256)
    pool.allocate(4)
    source = torch.randn(130, 256, device="cuda", dtype=torch.bfloat16)
    destination = torch.empty_like(source)

    def forbid_cpu(*args, **kwargs):
        raise AssertionError("CUDA writes must not stage payload through CPU")

    monkeypatch.setattr(pool, "_transfer_many", forbid_cpu)
    try:
        pool.write_many(range(130), source, asynchronous=True)
        pool.read_cuda(list(range(130)), destination)
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
        assert pool.write_bytes == 130 * 512
        assert pool.write_kernel_launches == 1
    finally:
        pool.close()
