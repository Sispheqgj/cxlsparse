"""Whole-layer plans, independent attention storage and per-layer frequency."""

import random
import subprocess
from collections import Counter

import pytest
import torch
import tvm_ffi.cpp

from cxl_recsys.mempool import MemoryPool
from cxl_recsys.mempool.cuda_transfer import RegisteredCudaMapping
from cxl_recsys.mempool.hisparse import HiSparseCache


class GatherSnapshot:
    """Inspect plans and unfinished values BEFORE the backing read/hot update."""

    def __init__(self, module):
        self.module = module
        self.snapshot = {}

    def __getattr__(self, name):
        return getattr(self.module, name)

    def plan_layer(self, *args):
        self.current_address = args[6].data_ptr()
        self.module.plan_layer(*args)

    def gather_layer(self, *args):
        (
            indices,
            descriptors,
            ranks,
            stats,
            offsets,
            hbm_sources,
            hbm_destinations,
            copy_sources,
            copy_destinations,
            values,
            mapped,
            miss_destinations,
            count,
            metadata_items,
        ) = args
        values.fill_(-7)
        self.module.gather_layer(*args)
        self.snapshot = {
            "count": count,
            "stats": stats.clone(),
            "offsets": offsets.clone(),
            "local_sources": hbm_sources.clone(),
            "local_destinations": hbm_destinations.clone(),
            "sources": copy_sources.flatten()[:count].clone(),
            "destinations": copy_destinations.flatten()[:count].clone(),
            "miss_destinations": miss_destinations.clone(),
            "values": values.clone(),
            "mapped": mapped.clone(),
        }


@pytest.mark.gpu
@pytest.mark.parametrize("dim", [4, 64, 128, 130, 576])
def test_dense_hbm_plan_contains_only_valid_jobs_and_leaves_miss_rows_unwritten(dim):
    pool = MemoryPool(2, dim)
    pool.allocate(2)
    source_a = torch.arange(100, 107).bfloat16()[:, None].expand(-1, dim).contiguous()
    source_b = torch.arange(200, 205).bfloat16()[:, None].expand(-1, dim).contiguous()
    pool.write_many(list(range(6)), source_a[:6])
    pool.write_many(list(range(64, 68)), source_b[:4])
    cache = HiSparseCache(pool, rows=2, max_history=8, hot=4, topk=3, dim=dim)
    cache.bind(0, list(range(6)))
    cache.bind(1, list(range(64, 68)))
    cache.tags[:] = torch.tensor([[0, 2, 1, -1, -1], [1, 0, -1, -1, -1]], device="cuda")
    cache.data[[0, 1, 2, 5, 6]] = torch.cat(
        (source_a[[0, 2, 1]], source_b[[1, 0]])
    ).cuda()
    recorder = GatherSnapshot(cache.module)
    cache.module = recorder
    inputs = [
        layer_input(
            0,
            torch.tensor([[1, 3, 4], [1, 4, 5], [3, 5, 6]], device="cuda"),
            source_a[4:].cuda(),
            cached=4,
        ),
        layer_input(
            1,
            torch.tensor([[0, 2, 3], [0, 3, 4]], device="cuda"),
            source_b[3:].cuda(),
            cached=3,
        ),
    ]
    inputs[0]["history_length"] = 6
    inputs[1]["history_length"] = 4
    try:
        batch, counters = cache.fetch_layer(inputs)
        assert isinstance(counters, list)
        snapshot = recorder.snapshot
        assert "hits" not in cache.workspace
        assert snapshot["count"] == 7
        assert snapshot["stats"].tolist() == [
            [5, 2, 1, 1, 4, 5, 0],
            [4, 2, 1, 1, 3, 3, 0],
        ]
        assert snapshot["offsets"].tolist() == [[0, 0], [5, 4]]
        assert snapshot["local_destinations"][0, :4].tolist() == [0, 2, 3, 4]
        assert snapshot["local_destinations"][1, :3].tolist() == [0, 2, 3]
        assert snapshot["destinations"].tolist() == [0, 2, 3, 4, 5, 7, 8]
        stride = dim * 2
        hot_address = cache.data.data_ptr()
        current_address = recorder.current_address
        assert snapshot["sources"].tolist() == [
            hot_address + 2 * stride,
            current_address,
            current_address + stride,
            current_address + 2 * stride,
            hot_address + 6 * stride,
            current_address + 3 * stride,
            current_address + 4 * stride,
        ]
        assert snapshot["miss_destinations"][:, 0].tolist() == [1, 6]
        expected = torch.cat((source_a[[1, 3, 4, 5, 6]], source_b[[0, 2, 3, 4]]))
        before_read = expected.clone()
        before_read[[1, 6]] = -7
        assert torch.equal(snapshot["values"].cpu(), before_read)
        assert torch.equal(batch.values.cpu(), expected)
        assert batch.indices.tolist() == [
            [0, 1, 2],
            [0, 2, 3],
            [1, 3, 4],
            [5, 6, 7],
            [5, 7, 8],
        ]
        assert batch.sequence_indptr == [0, 5, 9]
        assert [item["misses"] for item in counters] == [1, 1]
        assert pool.read_bytes == 2 * stride
    finally:
        pool.close()


@pytest.mark.gpu
def test_hbm_plan_handles_all_miss_then_all_hit_and_zero_cached_history():
    pool = MemoryPool(1, 128)
    pool.allocate(1)
    source = torch.arange(8 * 128).reshape(8, 128).to(torch.int16).view(torch.bfloat16)
    pool.write_many(list(range(8)), source)
    cache = HiSparseCache(pool, rows=1, max_history=8, hot=4, topk=4, dim=128)
    cache.bind(0, list(range(8)))
    recorder = GatherSnapshot(cache.module)
    cache.module = recorder
    selected = torch.tensor([[0, 2, 4, 6]], device="cuda")
    item = layer_input(
        0, selected, torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16), cached=8
    )
    try:
        first, _ = cache.fetch_layer([item])
        assert recorder.snapshot["count"] == 0
        assert recorder.snapshot["sources"].numel() == 0
        assert bool((recorder.snapshot["values"] == -7).all())
        assert torch.equal(
            first.values.cpu().view(torch.int16), source[[0, 2, 4, 6]].view(torch.int16)
        )
        read_bytes = pool.read_bytes
        second, _ = cache.fetch_layer([item])
        assert recorder.snapshot["count"] == 4
        assert pool.read_bytes == read_bytes
        assert torch.equal(
            second.values.cpu().view(torch.int16), first.values.cpu().view(torch.int16)
        )
        # Reuse the larger plan workspace with a smaller current-only request.
        current = source[:3].cuda()
        item = layer_input(
            0, torch.tensor([[0, -1], [0, 1], [1, 2]], device="cuda"), current, cached=0
        )
        item["history_length"] = 2
        third, _ = cache.fetch_layer([item])
        assert recorder.snapshot["count"] == 3
        assert recorder.snapshot["destinations"].tolist() == [0, 1, 2]
        assert pool.read_bytes == read_bytes
        assert third.indices.tolist() == [[0, -1], [0, 1], [1, 2]]
        assert torch.equal(
            third.values.cpu().view(torch.int16), source[:3].view(torch.int16)
        )
    finally:
        pool.close()


def layer_input(row, selected, current, cached=512):
    return {
        "row": row,
        "indices": selected,
        "history_length": cached,
        "cached_length": cached,
        "current_kv": current,
    }


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dram", "cxl"])
def test_layer_frequency_matches_reference_across_hits_and_evictions(backend):
    pool = MemoryPool(8, 64)
    pool.backend = backend
    pool.allocate(8)
    source = torch.arange(512 * 64).reshape(512, 64).to(torch.int16)
    pool.write_many(list(range(512)), source.view(torch.bfloat16))
    cache = HiSparseCache(pool, rows=1, max_history=512, hot=4, topk=4, dim=64)
    cache.bind(0, list(range(512)))
    reference = []
    generator = random.Random(19)
    selections = [[0, 1], [2], [0], [3, 5, 7], list(range(12)), [9, 11]]
    selections += [[0] * 20 + [7] * 10 + [4] * 10 + [12] * 5 + [511], [511], [2, 1]]
    selections.extend(generator.choices(range(24), k=size) for size in range(1, 48))
    try:
        for selected in selections:
            unique = sorted(set(selected))
            expected_hits = len(set(unique) & set(reference))
            counts = Counter(selected)
            reference = sorted(counts, key=lambda token: (-counts[token], token))[:4]
            # One selected token per query allows a union much larger than hot.
            indices = torch.tensor(selected, device="cuda").reshape(-1, 1)
            current = torch.zeros(
                (len(selected), 64), device="cuda", dtype=torch.bfloat16
            )
            before = cache.planner_calls
            batch, layer_counters = cache.fetch_layer(
                [layer_input(0, indices, current)]
            )
            assert isinstance(layer_counters, list)
            (stats,) = layer_counters
            assert cache.planner_calls == before + 1
            assert stats["planner_calls"] == 1
            assert stats["hits"] == expected_hits
            assert stats["misses"] == len(unique) - expected_hits
            assert torch.equal(
                batch.values[batch.indices[:, 0]].cpu().view(torch.int16),
                source[selected],
            )
            ordered = cache.tags[0, :4].tolist()
            assert [token for token in ordered if token >= 0] == reference
            assert not hasattr(cache, "lru")
            tags = cache.tags[0, :4].long()
            valid = tags >= 0
            assert torch.equal(
                cache.data[cache.locs[0, :4][valid].long()].cpu().view(torch.int16),
                source[tags[valid].cpu()],
            )
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dram", "cxl"])
def test_new_history_is_admitted_but_candidates_and_unselected_entries_are_not(backend):
    pool = MemoryPool(1, 64)
    pool.backend = backend
    pool.allocate(1)
    source = torch.arange(10 * 64).reshape(10, 64).bfloat16()
    pool.write_many(list(range(6)), source[:6])
    # hot exceeds the workspace extent: only selected valid slots may be filled.
    cache = HiSparseCache(pool, rows=1, max_history=8, hot=32, topk=4, dim=64)
    cache.bind(0, list(range(6)))
    selected = torch.tensor(
        [
            [0, 1, 4, -1],
            [0, 4, 5, -1],
            [1, 4, 5, 6],
            [0, 5, 6, 7],
            [4, 6, 7, 8],
            [6, 7, 8, 9],
        ],
        device="cuda",
    )
    item = layer_input(0, selected, source[4:].cuda(), cached=4)
    item["history_length"] = 6
    try:
        batch, layer_counters = cache.fetch_layer([item])
        assert isinstance(layer_counters, list)
        (stats,) = layer_counters
        assert stats["misses"] == 2
        assert cache.tags[0, :4].tolist() == [4, 0, 5, 1]
        assert bool((cache.tags[0, 4:] == -1).all())
        tags = cache.tags[0, :4].long()
        assert torch.equal(
            cache.data[cache.locs[0, :4].long()].cpu(), source[tags.cpu()]
        )
        mask = selected >= 0
        assert torch.equal(
            batch.values[batch.indices[mask]].cpu(), source[selected[mask].cpu()]
        )
        read_bytes = pool.read_bytes
        values, stats = cache.fetch(0, [4, 5], 6)
        assert stats == {"hits": 2, "misses": 0}
        assert pool.read_bytes == read_bytes
        assert torch.equal(values.cpu(), source[[4, 5]])
        assert cache.tags[0, :4].tolist() == [4, 5, -1, -1]
        # A candidate-only selection leaves the persistent hot set empty, rather
        # than carrying an unselected hot entry from the previous invocation.
        candidates = layer_input(
            0,
            torch.tensor([[6, -1], [6, 7]], device="cuda"),
            source[6:8].cuda(),
            cached=6,
        )
        cache.fetch_layer([candidates])
        assert bool((cache.tags == -1).all())
        assert pool.read_bytes == read_bytes
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("hot", [128, 256])
def test_frequency_top128_multiuser_matches_cpu_reference(hot, monkeypatch):
    pool = MemoryPool(32, 128)
    pool.backend = "cxl"
    pool.allocate(32)
    source = torch.arange(2048 * 128).reshape(2048, 128)
    source = (source + (torch.arange(2048) // 1024).reshape(-1, 1) * 31).to(torch.int16)
    pool.write_many(list(range(2048)), source.view(torch.bfloat16))
    cache = HiSparseCache(pool, rows=2, max_history=768, hot=hot, topk=128, dim=128)
    fetches = []
    expected = []
    for row in range(2):
        base = row * 1024
        cache.bind(row, list(range(base, base + 768)))
        # Generate the fixture on CPU so sanitizer checks focus on the cache,
        # not unrelated scoring kernels. The real Indexer has separate coverage
        # in test_indexer.py and the E2E; this fixture only tests cache semantics.
        generator = random.Random(19 + row)
        indices = torch.tensor(
            [
                sorted(generator.sample(range(position + 1), 128))
                for position in range(512, 896)
            ],
            device="cuda",
            dtype=torch.int64,
        )
        item = layer_input(
            row,
            indices,
            source[base + 512 : base + 896].view(torch.bfloat16).cuda(),
            cached=512,
        )
        item["history_length"] = 768
        fetches.append(item)
        counts = Counter(indices[(indices >= 0) & (indices < 768)].tolist())
        expected.append(sorted(counts, key=lambda token: (-counts[token], token))[:hot])

    # Compiler-free execution also covers the new stable GPU sort.
    def forbid_compile(*args, **kwargs):
        raise AssertionError("Frequency ranking must not invoke runtime compilation")

    for name in ("load_inline", "build_inline", "load", "build"):
        monkeypatch.setattr(tvm_ffi.cpp, name, forbid_compile)
    monkeypatch.setattr(subprocess, "Popen", forbid_compile)
    try:
        for call in range(2):
            batch, stats = cache.fetch_layer(fetches)
            assert isinstance(stats, list)
            assert cache.planner_calls == call + 1
            assert sum(item["planner_calls"] for item in stats) == 1
            assert pool.read_kernel_launches == call + 1
            for row in range(2):
                assert cache.tags[row, :hot].tolist() == expected[row]
                slots = cache.locs[row, :hot].long()
                physical = torch.tensor(expected[row]) + row * 1024
                assert torch.equal(
                    cache.data[slots].cpu().view(torch.int16), source[physical]
                )
                indices = fetches[row]["indices"]
                mapped = batch.indices[row * 384 : (row + 1) * 384]
                assert torch.equal(
                    batch.values[mapped].cpu().view(torch.int16),
                    source[indices.cpu() + row * 1024],
                )
    finally:
        pool.close()


@pytest.mark.gpu
def test_cxl_ragged_plan_stays_on_gpu_and_buffers_survive_workspace_growth(monkeypatch):
    pool = MemoryPool(32, 64)
    pool.backend = "cxl"
    pool.allocate(32)
    source = (torch.arange(2048 * 64) * 7).reshape(2048, 64)
    source = (source + (torch.arange(2048) // 512).reshape(-1, 1) * 31).to(torch.int16)
    pool.write_many(list(range(2048)), source.view(torch.bfloat16))
    cache = HiSparseCache(pool, rows=3, max_history=512, hot=32, topk=8, dim=64)
    cache.bind(0, list(range(512)))
    cache.bind(2, list(range(1024, 1536)))
    generator = torch.Generator().manual_seed(19)
    snapshots = []
    inspect_list = torch.Tensor.tolist

    def small_summary_only(tensor):
        assert tensor.shape == (2, 7), "Token-sized plans must not reach Python"
        return inspect_list(tensor)

    def forbid(*args, **kwargs):
        raise AssertionError("CXL read must consume the GPU plan directly")

    try:
        for sizes in ((273, 41), (601, 87), (113, 41)):
            inputs = []
            expected = []
            for row, count, source_base in zip((0, 2), sizes, (0, 1024)):
                # Old tokens overlap across users, but their physical KV differs.
                selected = torch.randint(512, (count, 8), generator=generator)
                for query in range(count):
                    old = sorted(set(selected[query, :6].tolist()))
                    tokens = old + [512 + query, -1]
                    tokens += [-1] * (8 - len(tokens))
                    selected[query] = torch.tensor(tokens)
                current_bits = torch.randint(
                    -32768, 32767, (count, 64), generator=generator
                ).to(torch.int16)
                combined = torch.cat(
                    (source[source_base : source_base + 512], current_bits)
                )
                expected.append((selected.clone(), combined))
                inputs.append(
                    layer_input(
                        row, selected.cuda(), current_bits.view(torch.bfloat16).cuda()
                    )
                )
            # Inputs are ready before alternating streams; the cache itself orders
            # its persistent metadata and reusable workspace through events.
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            before = pool.read_kernel_launches
            with torch.cuda.stream(stream), monkeypatch.context() as guard:
                guard.setattr(torch.Tensor, "tolist", small_summary_only)
                guard.setattr(torch.Tensor, "cpu", forbid)
                guard.setattr(torch, "unique", forbid)
                guard.setattr(pool, "read_many", forbid)
                guard.setattr(RegisteredCudaMapping, "read", forbid)
                guard.setattr(RegisteredCudaMapping, "validate_read", forbid)
                batch, stats = cache.fetch_layer(inputs)
                assert isinstance(stats, list)
            stream.synchronize()
            assert pool.read_kernel_launches == before + 1
            assert sum(item["planner_calls"] for item in stats) == 1
            assert batch.sequence_indptr[-1] > cache.hot
            offset = 0
            for row, (selected, combined) in zip((0, 2), expected):
                mapped = batch.indices[offset : offset + len(selected)].cpu()
                mask = selected >= 0
                assert torch.equal(mapped < 0, ~mask)
                assert torch.equal(
                    batch.values[mapped[mask].cuda()].cpu().view(torch.int16),
                    combined[selected[mask]],
                )
                frequencies = Counter(
                    selected[(selected >= 0) & (selected < 512)].tolist()
                )
                expected_hot = sorted(
                    frequencies, key=lambda token: (-frequencies[token], token)
                )[:32]
                assert cache.tags[row, :32].tolist() == expected_hot
                offset += len(selected)
            assert bool((cache.tags[1] == -1).all())
            assert bool((cache.tags[:, :32] < 512).all())
            for previous, bits, logical, indices in snapshots:
                assert torch.equal(previous.values.cpu().view(torch.int16), bits)
                assert torch.equal(previous.logical_ids.cpu(), logical)
                assert torch.equal(previous.indices.cpu(), indices)
            snapshots.append(
                (
                    batch,
                    batch.values.cpu().view(torch.int16).clone(),
                    batch.logical_ids.cpu(),
                    batch.indices.cpu(),
                )
            )
        assert cache.planner_calls == 3
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("slot", [-1, 64])
def test_invalid_backing_slot_rejected_before_copy(slot):
    pool = MemoryPool(1, 64)
    pool.backend = "cxl"
    pool.allocate(1)
    cache = HiSparseCache(pool, rows=1, max_history=8, hot=4, topk=4, dim=64)
    cache.bind(0, [slot])
    try:
        with pytest.raises(ValueError, match="bound history"):
            cache.fetch_layer(
                [
                    layer_input(
                        0,
                        torch.zeros((1, 1), dtype=torch.int64, device="cuda"),
                        torch.zeros((1, 64), dtype=torch.bfloat16, device="cuda"),
                        cached=1,
                    )
                ]
            )
        assert pool.read_kernel_launches == 0
        assert pool.cuda_mapping is None
        assert bool((cache.tags == -1).all())
    finally:
        pool.close()


@pytest.mark.gpu
def test_large_history_scan_hot_reordering_and_unchanged_slots():
    # 257 scan tiles exercises the inter-tile carry at the 64k boundary.
    # Read values across the whole extent and verify frequency ties/evictions.
    history, dim, hot = 65536, 4, 4096
    pool = MemoryPool(history // 64, dim)
    pool.allocate(history // 64)
    source = (torch.arange(history * dim) % 32000).to(torch.int16).reshape(-1, dim)
    pool.write_many(list(range(history)), source.view(torch.bfloat16))
    cache = HiSparseCache(
        pool, rows=1, max_history=history, hot=hot, topk=4096, dim=dim
    )
    cache.append_slots(0, 0, list(range(history)))
    try:
        for stride, offset in ((16, 0), (16, 0), (16, 1), (16, 1)):
            ids = torch.arange(offset, history, stride, device="cuda")[None].repeat(
                3, 1
            )
            item = layer_input(
                0,
                ids,
                torch.zeros(3, dim, device="cuda", dtype=torch.bfloat16),
                cached=history,
            )
            result, layer_counters = cache.fetch_layer([item])
            assert isinstance(layer_counters, list)
            (stats,) = layer_counters
            expected = source[ids.cpu()]
            assert torch.equal(
                result.values[result.indices].cpu().view(torch.int16), expected
            )
            assert torch.equal(cache.tags[0, :hot].long(), ids[0])
            assert torch.equal(
                cache.data[cache.locs[0, :hot].long()].cpu().view(torch.int16),
                expected[0],
            )
            assert stats["hits"] + stats["misses"] == hot
            assert stats["old_references"] == 3 * hot
        assert stats["misses"] == 0
        with pytest.raises(ValueError, match="append-only"):
            cache.append_slots(0, 0, [1])
    finally:
        pool.close()
