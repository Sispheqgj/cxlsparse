"""Region metadata, CPU file IO and real CUDA IO on anonymous mappings.

The geometry/character-device check is mocked, never the transfer kernels.
Physical devdax validation is run separately and is not implied by these tests.
"""

import mmap
import os
from types import SimpleNamespace

import pytest
import torch
import yaml
from kv_helpers import assert_write_slots

from cxl_recsys.cli import parse_args
from cxl_recsys.common.sizes import dax_regions, parse_bytes
from cxl_recsys.mempool import MemoryPool
from cxl_recsys.mempool import cxl as cxl_module
from cxl_recsys.mempool.device_plan import DeferredLayerCounters
from cxl_recsys.mempool.hisparse import HiSparseCache
from cxl_recsys.models.base import CacheContext
from cxl_recsys.models.simulation.model import TestModel

ALIGN = 2 * 1024 * 1024


@pytest.mark.gpu
@pytest.mark.parametrize("regions,pages", [(2, 17), (4, 1)])
def test_block_cache_exact_region_bytes(region_pool, regions, pages):
    pool = region_pool(regions, pages=pages, dim=256)
    pool.allocate(pages)
    history = pages * 64
    source = torch.randn(history, 256, device="cuda").bfloat16()
    pool.write_many(list(range(history)), source)
    caches = [
        HiSparseCache(
            pool,
            rows=1,
            max_history=history,
            hot=256,
            topk=256,
            dim=256,
            device_resident=True,
        )
        for _ in range(2)
    ]
    for cache in caches:
        cache.bind(0, list(range(history)))
    block_ids = sorted(set([0, pages // 2, pages - 1]))
    blocks = torch.tensor([block_ids], device="cuda")
    tokens = (blocks[..., None] * 64 + torch.arange(64, device="cuda")).flatten(1)
    try:
        for repeat in range(2):
            results = []
            for cache, ids, size in zip(caches, (tokens, blocks), (1, 64)):
                before = [region["read_bytes"] for region in pool.stats()["regions"]]
                batch, counts = cache.fetch_layer(
                    [
                        dict(
                            row=0,
                            cached_length=history,
                            history_length=history,
                            current_kv=torch.zeros(
                                1, 256, device="cuda", dtype=torch.bfloat16
                            ),
                            indices=ids,
                        )
                    ],
                    block_size=size,
                )
                assert isinstance(counts, DeferredLayerCounters)
                stats = counts.materialize(counts.statistics.tolist())
                pool.flush_read_stats()
                delta = [
                    region["read_bytes"] - old
                    for region, old in zip(pool.stats()["regions"], before)
                ]
                assert delta == [
                    sum(block % regions == r for block in block_ids) * 64 * 512
                    if repeat == 0
                    else 0
                    for r in range(regions)
                ]
                results.append((batch, stats))
            a, b = results
            assert a[1] == b[1]
            assert a[0].logical_positions == b[0].logical_positions
            assert torch.equal(
                a[0].values[: len(block_ids) * 64], b[0].values[: len(block_ids) * 64]
            )
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("regions,pages", [(2, 17), (4, 1)])
def test_device_cache_deferred_region_bytes_and_zero_miss_reads(
    region_pool, regions, pages
):
    pool = region_pool(regions, pages=pages)
    pool.allocate(pages)
    history = pages * 64
    source = (
        torch.arange(history, device="cuda", dtype=torch.float32)
        .bfloat16()[:, None]
        .expand(-1, 128)
        .contiguous()
    )
    pool.write_many(list(range(history)), source)
    cache = HiSparseCache(
        pool, rows=1, max_history=history, hot=32, topk=8, dim=128, device_resident=True
    )
    cache.bind(0, list(range(history)))
    ids = sorted(set([0, 1, history // 2, history - 1]))
    fetches = [
        dict(
            row=0,
            cached_length=history,
            history_length=history,
            indices=torch.tensor([ids], device="cuda"),
            current_kv=torch.zeros(1, 128, dtype=torch.bfloat16, device="cuda"),
        )
    ]
    for repeat in range(2):
        batch, deferred = cache.fetch_layer(fetches)
        assert isinstance(deferred, DeferredLayerCounters)
        stats = deferred.materialize(deferred.statistics.tolist())
        pool.flush_read_stats()
        assert torch.equal(batch.values[batch.indices[0]], source[ids])
        assert stats[0]["misses"] == (len(ids) if repeat == 0 else 0)
        assert stats[0]["hits"] == (0 if repeat == 0 else len(ids))
        assert pool.read_bytes == len(ids) * pool.token_bytes
        for region, actual in enumerate(pool.stats()["regions"]):
            expected = sum((slot // 64) % regions == region for slot in ids)
            assert actual["read_bytes"] == expected * pool.token_bytes


@pytest.fixture
def region_pool(tmp_path, monkeypatch, request):
    devices = [tmp_path / "dax0.0", tmp_path / "dax1.0"]
    for path in devices:
        with path.open("wb") as file:
            file.truncate(ALIGN * 4)
    monkeypatch.setattr(cxl_module, "dax_geometry", lambda _: (ALIGN, ALIGN * 4))
    monkeypatch.setattr(cxl_module.stat, "S_ISCHR", lambda _: True)
    if request.node.get_closest_marker("gpu") is not None:
        # CUDA cannot register this host's ordinary file-backed test mappings.
        # Keep the region geometry/FD tests, but use independently registered
        # anonymous mappings for GPU correctness, as in the single-region tests.
        monkeypatch.setattr(
            cxl_module,
            "mmap",
            SimpleNamespace(
                mmap=lambda fd, length, **kwargs: mmap.mmap(-1, length),
                MAP_SHARED=mmap.MAP_SHARED,
                PROT_READ=mmap.PROT_READ,
                PROT_WRITE=mmap.PROT_WRITE,
            ),
        )
    instances = []

    def create(count, pages=17, dim=128, **kwargs):
        paths = [str(devices[index // 2]) for index in range(count)]
        offsets = [(index % 2) * ALIGN * 2 for index in range(count)]
        pool = MemoryPool(
            pages,
            dim,
            "cxl",
            dax_device=paths,
            dax_offset=offsets,
            dax_bytes=[ALIGN * 2] * count,
            **kwargs,
        )
        instances.append(pool)
        return pool

    yield create
    for instance in instances:
        instance.close()


@pytest.mark.parametrize(
    "value,expected",
    [
        (128, 128),
        ("128", 128),
        ("128GiB", 128 << 30),
        ("2 MiB", ALIGN),
        ("1GB", 10**9),
        (0, 0),
    ],
)
def test_byte_units(value, expected):
    assert parse_bytes(value) == expected


@pytest.mark.parametrize("value", [True, -1, 1.5, "1.5GiB", "128G", "", [], "-1MiB"])
def test_bad_byte_units(value):
    with pytest.raises(ValueError):
        parse_bytes(value)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_yaml_and_cli_region_lists(count, tmp_path):
    devices = ["dax0.0", "dax0.0", "dax1.0", "dax1.0"][:count]
    offsets = [0, "128GiB", 0, "128GiB"][:count]
    sizes = ["128GiB"] * count
    config = tmp_path / "regions.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "worker": {
                    "backend": "cxl",
                    "dax_device": devices,
                    "dax_offset": offsets,
                    "dax_bytes": sizes,
                }
            }
        )
    )
    args = parse_args(["worker", "--config", str(config)])
    cli = parse_args(
        [
            "worker",
            "--backend",
            "cxl",
            "--dax-device",
            *devices,
            "--dax-offset",
            *map(str, offsets),
            "--dax-bytes",
            *sizes,
        ]
    )
    for name in ("dax_device", "dax_offset", "dax_bytes"):
        assert getattr(args, name) == getattr(cli, name)
    assert args.dax_device == ["/dev/" + device for device in devices]
    assert args.dax_offset == [parse_bytes(value) for value in offsets]
    # Explicit CLI triples replace YAML lists; the legacy scalar CLI is N=1.
    one = parse_args(
        [
            "worker",
            "--config",
            str(config),
            "--dax-device",
            "dax1.0",
            "--dax-offset",
            "0",
            "--dax-bytes",
            "2097152",
        ]
    )
    assert (one.dax_device, one.dax_offset, one.dax_bytes) == (
        ["/dev/dax1.0"],
        [0],
        [ALIGN],
    )


@pytest.mark.parametrize(
    "triple",
    [
        ([], [], []),
        (["dax0.0"], [0, 1], [ALIGN]),
        (["dax0.0"], [0], [0]),
        (None, [0], [ALIGN]),
    ],
)
def test_region_shapes_rejected(triple):
    with pytest.raises(ValueError):
        dax_regions(*triple)


@pytest.mark.parametrize(
    "offsets,sizes,pages,reason",
    [
        ([1, ALIGN * 2], [ALIGN, ALIGN], 16, "aligned"),
        ([0, ALIGN * 2], [ALIGN + 1, ALIGN], 16, "aligned"),
        ([0, ALIGN * 3], [ALIGN, ALIGN * 2], 16, "out of bounds"),
        ([0, ALIGN], [ALIGN * 2, ALIGN], 16, "overlap"),
        ([0, ALIGN * 2], [ALIGN, ALIGN * 2], 257, "too small"),
    ],
)
def test_invalid_region_geometry_rejected_before_open(
    offsets, sizes, pages, reason, monkeypatch
):
    monkeypatch.setattr(cxl_module, "dax_geometry", lambda _: (ALIGN, ALIGN * 4))

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid configuration must not open any device")

    with monkeypatch.context() as guard:
        guard.setattr(cxl_module.os, "open", forbidden)
        with pytest.raises(ValueError, match=reason):
            MemoryPool(
                pages,
                128,
                "cxl",
                dax_device=["/dev/dax0.0", "/dev/dax0.0"],
                dax_offset=offsets,
                dax_bytes=sizes,
            )


def test_unused_regions_have_no_mapping(region_pool):
    pool = region_pool(4, pages=1)
    assert pool.cxl is not None
    assert [region.pages for region in pool.cxl.regions] == [1, 0, 0, 0]
    assert [region.mapped_bytes for region in pool.cxl.regions] == [ALIGN, 0, 0, 0]
    assert all(region.mapping is None for region in pool.cxl.regions[1:])


def test_failed_device_open_releases_previous_fd(tmp_path, monkeypatch):
    device = tmp_path / "device"
    device.touch()
    monkeypatch.setattr(cxl_module, "dax_geometry", lambda _: (ALIGN, ALIGN))
    opened = []
    original_open = os.open

    def track(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    with monkeypatch.context() as guard:
        guard.setattr(cxl_module.os, "open", track)
        with pytest.raises(ValueError, match="character device"):
            MemoryPool(
                1, 128, "cxl", dax_device=str(device), dax_offset=0, dax_bytes=ALIGN
            )
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_striped_cpu_roundtrip_and_mapping_ownership(region_pool, count):
    pool = region_pool(count)
    pages = pool.allocate(pool.pages)
    slots = list(range(pool.pages * pool.page_size))
    expected = torch.arange(len(slots) * 128).reshape(-1, 128).bfloat16()
    assert pool.cxl is not None
    assert len(pool.cxl.fds) == (count + 1) // 2
    assert sum(r.pages for r in pool.cxl.regions) == pool.pages
    assert all(r.mapped_bytes == ALIGN for r in pool.cxl.regions)
    for page in pages:
        for offset in (0, 63):
            assert pool._location(page * 64 + offset) == (
                page % count,
                page // count * 64 + offset,
            )
    pool.write_many(slots, expected)
    actual = torch.empty_like(expected)
    pool.read_many(slots[::-1], actual)
    assert torch.equal(actual, expected.flip(0))
    assert pool.write_operations == pool.read_operations == count
    assert sum(r["write_bytes"] for r in pool.stats()["regions"]) == pool.write_bytes
    # Released/reused global page IDs preserve their physical region mapping.
    pool.release(pages)
    assert sorted(pool.allocate(pool.pages)) == pages
    fds = list(pool.cxl.fds.values())
    pool.close()
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    with pytest.raises(RuntimeError, match="closed"):
        pool.read_many([0], actual[:1])


@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.gpu
def test_multiregion_gpu_scatter_read_and_sparse_destinations(
    region_pool, count, monkeypatch
):
    pool = region_pool(count)
    pool.allocate(pool.pages)
    slots = list(range(pool.pages * 64))[::-1]
    expected = (
        (torch.arange(len(slots) * 128) * 7)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(-1, 128)
    )
    source = expected.cuda()
    stream = torch.cuda.Stream()

    def forbidden(*args, **kwargs):
        raise AssertionError("KV payload must not use CPU staging")

    with torch.cuda.stream(stream), monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "cpu", forbidden)
        guard.setattr(pool, "read_many", forbidden)
        guard.setattr(pool, "_transfer_span", forbidden)
        source.record_stream(stream)
        stream.wait_stream(torch.cuda.default_stream())
        pool.write_many(slots, source)
        assert stream.query()
        selected = [0, 63, 64, 127, 128, 191, 192, len(slots) - 1]
        rows = list(range(0, len(selected) * 2, 2))
        output = torch.full(
            (len(selected) * 2, 128), -1, dtype=torch.bfloat16, device="cuda"
        )
        pool.read_cuda(selected, output, rows)
        assert stream.query()
    expected_rows = expected[[len(slots) - 1 - slot for slot in selected]]
    assert torch.equal(
        output[rows].cpu().view(torch.int16), expected_rows.view(torch.int16)
    )
    assert bool((output[1::2] == -1).all())
    assert pool.write_kernel_launches == 1
    assert sum(r["read_bytes"] for r in pool.stats()["regions"]) == pool.read_bytes


@pytest.mark.gpu
@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.parametrize("page_size", [1, 17, 64, 128])
def test_fused_write_partial_pages_counts_and_single_launch(
    region_pool, count, page_size, monkeypatch
):
    pool = region_pool(count, page_size=page_size)
    pool.allocate(pool.pages)
    all_slots = list(range(pool.pages * page_size))
    initial = torch.full((len(all_slots), 128), -3, dtype=torch.bfloat16)
    pool.write_many(all_slots, initial)
    # Scrambled, sparse writes cross pages/regions and leave surrounding tokens
    # unchanged. The final token group is not a complete warp or CTA.
    slots = all_slots[::3][::-1]
    source = torch.arange(len(slots) * 128).reshape(-1, 128).bfloat16().cuda()
    storage = pool.cxl
    assert storage is not None
    storage._initialize_cuda(source.device)
    bases = storage.region_bases
    module = storage.module
    launches = []
    before = pool.stats()

    def launch(values, indices, addresses, size):
        assert values is source and addresses is bases
        assert indices.shape == (len(slots),) and indices.is_contiguous()
        launches.append(len(indices))
        module.write_regions(values, indices, addresses, size)

    def forbidden(*args, **kwargs):
        raise AssertionError("Fused writes must not group slots or stage KV on CPU")

    with monkeypatch.context() as guard:
        guard.setattr(storage, "location", forbidden)
        guard.setattr(storage, "module", SimpleNamespace(write_regions=launch))
        guard.setattr(torch.Tensor, "cpu", forbidden)
        pool.write_many(slots, source)
    assert launches == [len(slots)]
    assert storage.region_bases is bases
    assert pool.write_kernel_launches == 1
    after = pool.stats()
    for region in range(count):
        tokens = sum(slot // page_size % count == region for slot in slots)
        assert (
            after["regions"][region]["write_bytes"]
            - before["regions"][region]["write_bytes"]
        ) == tokens * 256
        assert (
            after["regions"][region]["write_operations"]
            - before["regions"][region]["write_operations"]
        ) == int(tokens > 0)
    actual = torch.empty_like(initial)
    pool.read_many(all_slots, actual)
    initial[slots] = source.cpu()
    assert torch.equal(actual, initial)


@pytest.mark.gpu
@pytest.mark.parametrize("dim", [64, 512, 576])
def test_fused_write_vector_widths_and_empty_input(region_pool, dim):
    pool = region_pool(3, pages=5, dim=dim)
    pool.allocate(5)
    assert pool.cxl is not None
    pool.write_many([], torch.empty((0, dim), dtype=torch.bfloat16, device="cuda"))
    assert pool.cxl.region_bases is None
    slots = [319, 0, 64, 127, 128, 191, 192, 255, 256]
    source = torch.randn(len(slots), dim, device="cuda", dtype=torch.bfloat16)
    pool.write_many(slots, source)
    output = torch.empty_like(source)
    pool.read_cuda(slots, output)
    assert torch.equal(output, source)
    assert pool.write_kernel_launches == 1


@pytest.mark.gpu
@pytest.mark.parametrize("count", [1, 4])
def test_fused_write_read_first_registration_and_no_runtime_compiler(
    region_pool, count, monkeypatch
):
    import subprocess

    import tvm_ffi.cpp

    pool = region_pool(count, pages=4)
    pool.allocate(4)
    slots = [0, 64, 128, 255]
    source = torch.ones((4, 128), dtype=torch.bfloat16)
    pool.write_many(slots, source)
    torch.cuda.init()

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime must not compile kernels or spawn a compiler")

    for name in ("load_inline", "build_inline", "load", "build"):
        monkeypatch.setattr(tvm_ffi.cpp, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    output = torch.empty((4, 128), dtype=torch.bfloat16, device="cuda")
    pool.read_cuda(slots, output)
    assert torch.equal(output.cpu(), source)
    assert pool.cxl is not None
    mappings = [region.cuda for region in pool.cxl.regions]
    pool.write_many(slots, torch.full_like(output, 11))
    assert [region.cuda for region in pool.cxl.regions] == mappings
    pool.read_many(slots, source)
    assert bool((source == 11).all())
    pool.close()
    assert pool.cxl.region_bases is None
    assert all(mapping is not None and mapping.view is None for mapping in mappings)
    pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("count", [1, 3, 4])
def test_fused_write_waits_for_prior_read_on_another_stream(region_pool, count):
    pool = region_pool(count, pages=4)
    pool.allocate(4)
    slots = list(range(256))
    first = torch.ones((256, 128), dtype=torch.bfloat16, device="cuda")
    pool.write_many(slots, first)
    read_stream = torch.cuda.Stream()
    with torch.cuda.stream(read_stream):
        sources = torch.tensor([slots], dtype=torch.int64, device="cuda")
        targets = torch.arange(256, dtype=torch.int32, device="cuda").reshape(1, -1)
        counts = torch.tensor([256], dtype=torch.int32, device="cuda")
        real = torch.ones(1, dtype=torch.int32, device="cuda")
        output = torch.empty_like(first)
        assert pool.cxl is not None
        plan = pool.route_read_plan(sources, targets, counts)
        totals = plan.counts.sum(dim=1).tolist() if plan is not None else None
        # Metadata is ready; delay only the asynchronous KV load, then immediately
        # overwrite its source from a different stream through the fused writer.
        torch.cuda._sleep(10_000_000)
        pool.read_cuda(
            sources,
            output,
            targets,
            counts=counts,
            real_requests=real,
            num_tokens=256,
            region_plan=plan,
            region_totals=totals,
        )
    write_stream = torch.cuda.Stream()
    with torch.cuda.stream(write_stream):
        replacement = torch.full_like(first, 7)
        pool.write_many(slots, replacement)
        assert write_stream.query()
    read_stream.synchronize()
    assert torch.equal(output, first)
    final = torch.empty_like(first)
    pool.read_cuda(slots, final)
    assert torch.equal(final, replacement)


@pytest.mark.gpu
@pytest.mark.parametrize("count", [2, 3, 4])
def test_multiregion_planner_ragged_batches_hits_and_workspace_reuse(
    region_pool, count
):
    pool = region_pool(count, pages=16)
    pool.allocate(16)
    source = torch.arange(1024 * 128).reshape(-1, 128).bfloat16()
    pool.write_many(list(range(1024)), source.cuda())
    cache = HiSparseCache(pool, rows=2, max_history=256, hot=32, topk=8, dim=128)
    cache.bind(0, list(range(256)))
    cache.bind(1, list(range(512, 768)))
    outputs = []
    for rows, queries in (([0, 1], 24), ([1], 8), ([0, 1], 32)):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fetches = []
            for row in rows:
                indices = torch.arange(queries * 8, device="cuda").reshape(queries, 8)
                current = torch.zeros(
                    (queries, 128), device="cuda", dtype=torch.bfloat16
                )
                fetches.append(
                    dict(
                        row=row,
                        indices=indices,
                        history_length=256,
                        cached_length=256,
                        current_kv=current,
                    )
                )
            batch, counters = cache.fetch_layer(fetches)
            stream.synchronize()
        for index, row in enumerate(rows):
            values = batch.values[
                batch.indices[index * queries : (index + 1) * queries]
            ].cpu()
            assert torch.equal(
                values,
                source[row * 512 : row * 512 + queries * 8].reshape(queries, 8, 128),
            )
        assert isinstance(counters, list)
        assert sum(c["planner_calls"] for c in counters) == 1
        assert sum(c["cxl_read_kernel_launches"] for c in counters) <= count
        outputs.append((batch.values, batch.values.clone()))
    assert all(torch.equal(value, snapshot) for value, snapshot in outputs)


@pytest.mark.gpu
def test_empty_regions_and_single_active_read(region_pool):
    pool = region_pool(4, pages=1)
    pool.allocate(1)
    source = torch.arange(64 * 128, device="cuda").reshape(64, 128).bfloat16()
    pool.write_many(list(range(64)), source)
    output = torch.empty((2, 128), device="cuda", dtype=torch.bfloat16)
    assert pool.read_cuda([3, 63], output) == 1
    assert torch.equal(output, source[[3, 63]])
    assert pool.write_kernel_launches == pool.read_kernel_launches == 1
    assert all(region["read_bytes"] == 0 for region in pool.stats()["regions"][1:])


@pytest.mark.gpu
def test_multiregion_write_requires_bfloat16(region_pool):
    pool = region_pool(2)
    pool.allocate(1)
    source = torch.zeros((1, 128), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="BF16"):
        pool.write_many([0], source)
    assert pool.write_bytes == 0


@pytest.mark.gpu
@pytest.mark.parametrize("count", [1, 3, 4])
def test_model_hidden_states_match_unstriped_reference(region_pool, count, monkeypatch):
    pool = region_pool(count, pages=16)
    model = TestModel(pool, max_users=2, max_history=256, dim=128)
    reference = TestModel(MemoryPool(16, 128), max_users=2, max_history=256, dim=128)
    write_many = pool.write_many

    def checked_write(slots, source):
        assert_write_slots(pool, slots)
        write_many(slots, source)

    monkeypatch.setattr(pool, "write_many", checked_write)
    try:
        for length in (65, 128, 256):
            ids = list(range(length)) + [900, 901]
            packed = ids + [value + 1000 for value in ids]
            positions = list(range(len(ids))) * 2
            contexts = [CacheContext(f"u{user}", length) for user in range(2)]
            boundaries = [0, len(ids), len(packed)]
            expected = reference.forward(
                packed, positions, seq_indptr=boundaries, caches=contexts
            )
            actual = model.forward(
                packed, positions, seq_indptr=boundaries, caches=contexts
            )
            assert torch.equal(actual, expected)
            for left, right in zip(
                model.batch_forward_stats(), reference.batch_forward_stats()
            ):
                for key in ("hits", "misses", "old_references", "local_tokens"):
                    assert left["cache"][key] == right["cache"][key]
    finally:
        model.close()
        reference.close()
