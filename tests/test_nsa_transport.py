"""Sparse backend adapter numerics, partial residency and exact IO accounting."""

import os

import pytest
import torch
from test_nsa import checkpoint  # noqa: F401

from cxl_recsys.mempool import MemoryPool
from cxl_recsys.mempool.hisparse import HiSparseCache
from cxl_recsys.models.nsa.backends.io_plan import selected_attention_oracle
from cxl_recsys.models.nsa.backends.legacy_transport import SelectedTransport

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("backend", ["tilelang", "tilelang_indexed", "tilelang_gpu", "tilelang_tma"])
def test_model_gate_combination_and_prefix_reuse(checkpoint, backend):  # noqa: F811
    from cxl_recsys.models.base import CacheContext
    from cxl_recsys.models.nsa import NSAModel

    path, cfg, _, _ = checkpoint
    models = []
    ids = [(i * 7 + 9) % 255 for i in range(93)]
    try:
        for selected in ("triton", backend):
            pool = MemoryPool(
                cfg.required_pages(1, 256, 64),
                128,
                kernel_dir=os.environ.get(
                    "NSA_KERNEL_DIR", "artifacts/kernels-tilelang"
                ),
            )
            models.append(
                NSAModel(
                    pool,
                    max_users=1,
                    max_history=256,
                    layers=2,
                    model_path=path,
                    selected_backend=selected,
                    kernel_backend="fla",
                    cuda_graphs=True,
                )
            )
        fresh = [
            m.forward(ids, list(range(93)), cache=CacheContext("user", 80)).cpu()
            for m in models
        ]
        reuse = [
            m.forward(ids, list(range(93)), cache=CacheContext("user", 80)).cpu()
            for m in models
        ]
        torch.testing.assert_close(fresh[1], fresh[0], rtol=0.03, atol=0.03)
        torch.testing.assert_close(reuse[1], reuse[0], rtol=0.03, atol=0.03)
        torch.testing.assert_close(reuse[1], fresh[1][-13:], rtol=0.03, atol=0.03)
        assert (
            models[1].forward_stats()["nsa_execution"]["selected_tilelang_calls"] == 4
        )
        assert models[1].forward_stats()["input_tokens"] == 13
    finally:
        for model in models:
            model.close()


@pytest.mark.parametrize("width", [16, 256])
def test_tilelang_gpu_index_map_padding_duplicates_and_graph(width):
    from cxl_recsys.models.nsa.backends.indexed import indexed_attention

    generator = torch.Generator().manual_seed(44)
    query = torch.randn(8, 16, 64, generator=generator).bfloat16().cuda()
    records = torch.randn(57, 128, generator=generator).bfloat16().cuda()
    indices = torch.randint(
        0, 55, (8, width), generator=generator, dtype=torch.int64
    ).cuda()
    indices[0] = -1
    indices[1, -3:] = -1
    indices[2] = 7
    records[-2:] = torch.nan
    k, v = records[indices.clamp_min(0)].cpu().double().split(64, dim=-1)
    logits = torch.matmul(query.cpu().double(), k.transpose(1, 2)) / 8
    logits.masked_fill_(indices.cpu()[:, None] < 0, -torch.inf)
    expected = torch.matmul(logits.softmax(-1).nan_to_num(), v)
    for repeat in range(2):
        actual = indexed_attention(query, records, indices)
        torch.testing.assert_close(
            actual.cpu().double(), expected, rtol=0.03, atol=0.03
        )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        actual = indexed_attention(query, records, indices)
    graph.replay()
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.03, atol=0.03)


def setup_case(residency, pool=None):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 required")
    pool = pool or MemoryPool(
        8,
        128,
        kernel_dir=os.environ.get("NSA_KERNEL_DIR", "artifacts/kernels-tilelang"),
    )
    cache = HiSparseCache(pool, rows=2, max_history=256, hot=256, topk=256, dim=128)
    generator = torch.Generator().manual_seed(71)
    backing = torch.randn(80, 2, 128, generator=generator).bfloat16()
    query = torch.randn(13, 2, 16, 64, generator=generator).bfloat16().cuda()
    positions = torch.arange(67, 80, device="cuda")
    # Duplicate block 0 is intentional; shared causal tail block 4 spans old/new KV.
    blocks = torch.tensor(
        [[[0, 1, 0, 4], [1, 0, 1, 4]]] * 13, dtype=torch.int32, device="cuda"
    )
    if residency == "empty":
        blocks.fill_(-1)
    fetches = []
    for head in range(2):
        pages = pool.allocate(2)
        slots = pool.token_slots(pages, 0, 67)
        payload = backing[:67, head].contiguous()
        pool.write_many(slots, payload.cuda() if pool.backend == "rdma" else payload)
        cache.bind(head, slots)
        hot = (
            list(range(67))
            if residency == "hit"
            else list(range(16)) + [17, 65]
            if residency == "partial"
            else []
        )
        if hot:
            cache.tags[head, : len(hot)] = torch.tensor(
                hot, device="cuda", dtype=torch.int32
            )
            cache.data[cache.locs[head, : len(hot)].long()] = backing[hot, head].cuda()
        fetches.append(
            dict(
                row=head,
                cached_length=67,
                history_length=67,
                current_kv=backing[67:, head].contiguous().cuda(),
                pages=pages,
            )
        )
    return pool, cache, backing, query, positions, blocks, fetches


@pytest.mark.parametrize("residency", ["miss", "partial", "hit", "empty"])
@pytest.mark.parametrize("mode", ["serial", "overlap"])
def test_mapped_reads_oracle_hot_policy_and_changed_source(
    residency, mode, monkeypatch
):
    pool, cache, backing, query, positions, blocks, fetches = setup_case(residency)
    executor = SelectedTransport(cache, mode=mode, chunk_blocks=3, io_ctas=4)
    original_prepare = executor._prepare

    def poison(*args, **kwargs):
        context = original_prepare(*args, **kwargs)
        missing = [
            d for _, destinations in context["chunk_sources"] for d in destinations
        ]
        if missing:
            context["kv"].flatten(0, 1)[torch.tensor(missing, device=query.device)] = (
                torch.nan
            )
        return context

    monkeypatch.setattr(executor, "_prepare", poison)
    try:
        for repeat in range(2):
            if repeat:
                executor.last_done.synchronize()
                query.mul_(-0.75)
                # Only ephemeral source changes; immutable historical hot data stays valid.
                backing[67:].add_(0.25)
                for h, f in enumerate(fetches):
                    f["current_kv"].copy_(backing[67:, h])
            expected = selected_attention_oracle(
                query.cpu(), backing, blocks.cpu(), positions.cpu(), 16
            )
            before = pool.read_bytes
            if repeat:
                update = torch.cuda.Event()
                update.record(torch.cuda.current_stream())
            with (
                torch.cuda.stream(torch.cuda.Stream())
                if repeat
                else torch.cuda.stream(torch.cuda.current_stream())
            ):
                if repeat:
                    torch.cuda.current_stream().wait_event(update)
                output, counters = executor.run(query, blocks, positions, fetches, 16)
            executor.last_done.synchronize()
            torch.testing.assert_close(
                output.cpu().double(), expected, rtol=0.03, atol=0.03
            )
            assert pool.read_bytes - before == sum(c["misses"] for c in counters) * 256
            assert (
                sum(c["old_references"] for c in counters) == 13 * (3 * 16 + 3) * 2
                if residency != "empty"
                else sum(c["old_references"] for c in counters) == 0
            )
            for head, frequency in enumerate(executor.keepalive["frequencies"]):
                chosen = sorted(
                    frequency, key=lambda token: (-frequency[token], token)
                )[:256]
                assert cache.tags[head, : len(chosen)].tolist() == chosen
                if chosen:
                    torch.testing.assert_close(
                        cache.data[cache.locs[head, : len(chosen)].long()].cpu(),
                        backing[chosen, head],
                        rtol=0,
                        atol=0,
                    )
    finally:
        executor.close()
        pool.close()


@pytest.mark.parametrize("mode", ["serial", "overlap"])
def test_producer_error_unblocks_all_tickets_and_preserves_hot_state(mode, monkeypatch):
    pool, cache, _, query, positions, blocks, fetches = setup_case("miss")
    # Inject a transport failure, without pretending that this is an RDMA
    # hardware measurement. Real GPUDirect is tested separately below.
    pool.backend = "rdma"
    executor = SelectedTransport(cache, mode=mode, chunk_blocks=1)

    def fail(*args, **kwargs):
        raise RuntimeError("injected transport failure")

    monkeypatch.setattr(pool, "read_rdma_cuda", fail)
    try:
        with pytest.raises(RuntimeError, match="injected transport failure"):
            executor.run(query, blocks, positions, fetches, 16)
        assert cache.tags.eq(-1).all().item()
        assert pool.read_bytes == 0
    finally:
        executor.close()
        pool.backend = "dram"
        pool.close()


@pytest.mark.rdma
@pytest.mark.parametrize("mode", ["serial", "overlap"])
def test_real_gpudirect_loopback_continuation(mode):
    from cxl_recsys.common.rpc import serve
    from cxl_recsys.mempool.rdma import PoolManager

    service = PoolManager(2**20, "mlx5_1", 1, 1)
    server = serve(
        "127.0.0.1:59189",
        {"Open": service.open, "Close": service.release, "Status": service.status},
    )
    pool = MemoryPool(
        8,
        128,
        backend="rdma",
        pool_target="127.0.0.1:59189",
        device="mlx5_1",
        port=1,
        gid_index=1,
        kernel_dir=os.environ.get("NSA_KERNEL_DIR", "artifacts/kernels-tilelang"),
    )
    executor = None
    try:
        pool, cache, backing, query, positions, blocks, fetches = setup_case(
            "partial", pool
        )
        executor = SelectedTransport(cache, mode=mode, chunk_blocks=3)
        expected = selected_attention_oracle(
            query.cpu(), backing, blocks.cpu(), positions.cpu(), 16
        )
        before = pool.rdma_gpu_read_bytes
        actual, counters = executor.run(query, blocks, positions, fetches, 16)
        executor.last_done.synchronize()
        torch.testing.assert_close(
            actual.cpu().double(), expected, rtol=0.03, atol=0.03
        )
        assert (
            pool.rdma_gpu_read_bytes - before
            == sum(c["misses"] for c in counters) * 256
            > 0
        )
        assert pool.stats()["cpu_staging_bytes"] == 0
    finally:
        if executor is not None:
            executor.close()
        pool.close()
        server.stop(1).wait()
        assert service.status({})["sessions"] == 0
        service.close()
