"""Dynamic GPU plans preserve attention, hot policy and real transport ownership."""

import os

import pytest
import torch
from test_nsa import checkpoint  # noqa: F401
from test_nsa_transport import setup_case

from cxl_recsys.models.nsa.backends.gpu_transport import GPUSelectedTransport
from cxl_recsys.models.nsa.backends.io_plan import selected_attention_oracle

pytestmark = pytest.mark.gpu


def test_gpu_plan_packed_users_do_not_alias(checkpoint):  # noqa: F811
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.base import CacheContext
    from cxl_recsys.models.nsa import NSAModel

    path, cfg, _, _ = checkpoint
    models = []
    first = [(i * 7 + 9) % 255 for i in range(93)]
    second = [(i * 11 + 5) % 255 for i in range(93)]
    try:
        for backend in ("triton", "tilelang_gpu"):
            pool = MemoryPool(
                cfg.required_pages(2, 256, 64),
                128,
                kernel_dir=os.environ.get(
                    "NSA_KERNEL_DIR", "artifacts/kernels-tilelang"
                ),
            )
            models.append(
                NSAModel(
                    pool,
                    max_users=2,
                    max_history=256,
                    layers=2,
                    model_path=path,
                    selected_backend=backend,
                    cuda_graphs=True,
                )
            )
        for repeat in range(2):
            results = [
                m.forward(
                    first + second,
                    list(range(93)) * 2,
                    seq_indptr=[0, 93, 186],
                    caches=[CacheContext("a", 80), CacheContext("b", 80)],
                ).cpu()
                for m in models
            ]
            torch.testing.assert_close(results[1], results[0], rtol=0.03, atol=0.03)
            assert not torch.equal(
                results[1][: len(results[1]) // 2], results[1][len(results[1]) // 2 :]
            )
    finally:
        for model in models:
            model.close()


@pytest.mark.parametrize("residency", ["miss", "partial", "hit", "empty"])
@pytest.mark.parametrize("mode", ["serial", "overlap"])
def test_gpu_plan_oracle_hot_policy_and_replay(residency, mode):
    pool, cache, backing, query, positions, blocks, fetches = setup_case(residency)
    executor = GPUSelectedTransport(cache, mode=mode, chunk_blocks=3, io_ctas=4)
    try:
        for repeat in range(2):
            if repeat:
                executor.last_done.synchronize()
                query.mul_(-0.75)
                backing[67:].add_(0.25)
                for head, fetch in enumerate(fetches):
                    fetch["current_kv"].copy_(backing[67:, head])
            expected = selected_attention_oracle(
                query.cpu(), backing, blocks.cpu(), positions.cpu(), 16
            )
            before = pool.read_bytes
            if repeat:
                updated = torch.cuda.Event()
                updated.record(torch.cuda.current_stream())
            with torch.cuda.stream(
                torch.cuda.Stream() if repeat else torch.cuda.current_stream()
            ):
                if repeat:
                    torch.cuda.current_stream().wait_event(updated)
                output, counters = executor.run(query, blocks, positions, fetches, 16)
            executor.last_done.synchronize()
            rows = counters.statistics.tolist()
            counters = counters.materialize(rows)
            pool.flush_read_stats()
            torch.testing.assert_close(
                output.cpu().double(), expected, rtol=0.03, atol=0.03
            )
            assert pool.read_bytes - before == sum(c["misses"] for c in counters) * 256
            for head in range(2):
                selected = blocks[:, head].cpu()[:, :, None] * 16 + torch.arange(16)
                allowed = (
                    (blocks[:, head].cpu()[:, :, None] >= 0)
                    & (selected <= positions.cpu()[:, None, None])
                    & (selected < 67)
                )
                tokens = selected[allowed]
                frequency = torch.bincount(tokens, minlength=256)
                chosen = torch.argsort(frequency, descending=True, stable=True)
                chosen = chosen[frequency[chosen] > 0]
                actual = cache.tags[head, :256].cpu()
                assert actual[actual >= 0].tolist() == chosen.tolist()
                if len(chosen):
                    torch.testing.assert_close(
                        cache.data[cache.locs[head, : len(chosen)].long()].cpu(),
                        backing[chosen, head],
                        rtol=0,
                        atol=0,
                    )
            assert executor.last_stats["cpu_selection_downloads"] == 0
        assert executor.graphs.captures == 1
        assert executor.graphs.replays == 2
    finally:
        executor.close()
        pool.close()


@pytest.mark.rdma
@pytest.mark.parametrize("mode", ["serial", "overlap"])
def test_gpu_plan_real_rdma_persistent_mr_and_failure(mode, monkeypatch):
    from cxl_recsys.common.rpc import serve
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.mempool.rdma import PoolManager

    service = PoolManager(2**20, "mlx5_1", 1, 1)
    server = serve(
        "127.0.0.1:59191",
        {"Open": service.open, "Close": service.release, "Status": service.status},
    )
    pool = MemoryPool(
        8,
        128,
        backend="rdma",
        pool_target="127.0.0.1:59191",
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
        executor = GPUSelectedTransport(cache, mode=mode, chunk_blocks=3, io_ctas=4)
        for repeat in range(2):
            expected = selected_attention_oracle(
                query.cpu(), backing, blocks.cpu(), positions.cpu(), 16
            )
            output, counters = executor.run(query, blocks, positions, fetches, 16)
            executor.last_done.synchronize()
            torch.testing.assert_close(
                output.cpu().double(), expected, rtol=0.03, atol=0.03
            )
            assert len(pool.remote.region.registrations) == 1
            w = next(iter(executor.workspaces.values()))
            w["registration"].validate(pool.remote.region, w["rdma_values"])
            with pytest.raises(ValueError, match="owned GPU allocation"):
                w["registration"].validate(pool.remote.region, w["rdma_values"].clone())
        cache.tags.fill_(-1)

        def fail(*args, **kwargs):
            raise RuntimeError("injected transport failure")

        monkeypatch.setattr(pool, "read_rdma_cuda", fail)
        with pytest.raises(RuntimeError, match="injected transport failure"):
            executor.run(query, blocks, positions, fetches, 16)
        assert cache.tags.eq(-1).all().item()
    finally:
        if executor is not None:
            executor.close()
        assert not pool.remote.region.registrations
        pool.close()
        assert service.status({})["sessions"] == 0
        server.stop(0).wait()
        service.close()
