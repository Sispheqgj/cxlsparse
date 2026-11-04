"""Real GPU/NIC IO and model equivalence; CPU only handles address metadata.

Set CXL_RECSYS_RDMA_TARGET to an already running Pool to exercise a remote host.
Otherwise a private loopback Pool is used. Device/port/GID are environment-selected
so the test never changes network configuration or touches other users' MRs.
"""

import os
import socket
from types import SimpleNamespace

import pytest
import torch

from cxl_recsys.common.request import placeholder
from cxl_recsys.common.rpc import serve
from cxl_recsys.mempool import MemoryPool
from cxl_recsys.mempool.rdma import CudaDriver, PoolManager, Region
from cxl_recsys.models.base import CacheContext
from cxl_recsys.models.simulation.model import TestModel
from cxl_recsys.worker.prompt import build_prompt


def forbid_staging(*args, **kwargs):
    raise AssertionError("KV must not use CPU staging")


@pytest.fixture
def rdma_options():
    options = {
        "device": os.environ.get("CXL_RECSYS_RDMA_DEVICE", "mlx5_0"),
        "port": int(os.environ.get("CXL_RECSYS_RDMA_PORT", "1")),
        "gid_index": int(os.environ.get("CXL_RECSYS_RDMA_GID_INDEX", "0")),
    }
    target = os.environ.get("CXL_RECSYS_RDMA_TARGET")
    manager = server = None
    if target is None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            target = f"127.0.0.1:{sock.getsockname()[1]}"
        manager = PoolManager(64 * 1024**2, **options)
        server = serve(target, {"Open": manager.open, "Close": manager.release})
    try:
        yield {"backend": "rdma", "pool_target": target, **options}
    finally:
        if server is not None:
            server.stop(0).wait()
        if manager is not None:
            manager.close()


@pytest.mark.gpu
@pytest.mark.rdma
@pytest.mark.parametrize("dim", [4, 64, 576])
def test_direct_scatter_preserves_holes_and_non_page_aligned_views(
    rdma_options, dim, monkeypatch
):
    pool = MemoryPool(4, dim, **rdma_options)
    pool.allocate(4)
    source = torch.arange(7 * dim, device="cuda").reshape(7, dim).bfloat16()[1:]
    target = torch.full((12, dim), -1, device="cuda", dtype=torch.bfloat16)
    slots, rows = [63, 64, 65, 3, 8, 9], [7, 1, 2, 5, 3, 8]
    try:
        with monkeypatch.context() as guard:
            guard.setattr(torch.Tensor, "cpu", forbid_staging)
            guard.setattr(Region, "transfer", forbid_staging)
            pool.write_many(slots, source)
            pool.read_rdma_cuda(slots, target, rows)
        assert torch.equal(target[rows], source)
        assert bool((target[[0, 4, 6, 9, 10, 11]] == -1).all())
        assert pool.write_operations == 3
        assert pool.read_operations == 5
        assert (
            pool.rdma_gpu_write_bytes == pool.rdma_gpu_read_bytes == source.numel() * 2
        )
        assert pool.stats()["cpu_staging_bytes"] == 0
        assert pool.remote is not None and pool.remote.region.endpoint.size == 0
        with pytest.raises(ValueError, match="CPU staging is disabled"):
            pool.read_many(slots, torch.empty((6, dim), dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="cuda"):
            pool.write_many(slots, torch.empty((6, dim), dtype=torch.bfloat16))
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.rdma
def test_many_noncontiguous_spans_exceed_posting_window(rdma_options):
    pool = MemoryPool(64, 64, **rdma_options)
    pool.allocate(64)
    slots = list(range(0, 4096, 2))
    source = torch.randn((len(slots), 64), device="cuda", dtype=torch.bfloat16)
    target = torch.empty_like(source)
    try:
        pool.write_many(slots, source)
        pool.read_rdma_cuda(slots, target, list(range(len(slots))))
        assert torch.equal(source, target)
        assert pool.write_operations == pool.read_operations == len(slots)
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.rdma
@pytest.mark.parametrize("dim", [64, 576])
def test_packed_forward_matches_dram_without_payload_staging(
    rdma_options, dim, monkeypatch
):
    direct = TestModel(
        MemoryPool(32, dim, **rdma_options), max_users=2, max_history=256, dim=dim
    )
    reference = TestModel(MemoryPool(32, dim), max_users=2, max_history=256, dim=dim)
    try:
        for lengths in ((3, 5), (65, 70), (129, 193), (129, 193)):
            ids, positions, boundaries, contexts = [], [], [0], []
            for user, length in enumerate(lengths):
                request = placeholder(user_id=f"u{user}", history_length=length)
                prompt = build_prompt(
                    request["history_token_ids"], request["candidates"]
                )
                ids.extend(prompt.input_ids)
                positions.extend(prompt.positions)
                boundaries.append(len(ids))
                contexts.append(CacheContext(f"u{user}", length))
            expected = reference.forward(
                ids, positions, seq_indptr=boundaries, caches=contexts
            )
            with monkeypatch.context() as guard:
                guard.setattr(torch.Tensor, "cpu", forbid_staging)
                guard.setattr(Region, "transfer", forbid_staging)
                guard.setattr(direct.pool, "read_many", forbid_staging)
                actual = direct.forward(
                    ids, positions, seq_indptr=boundaries, caches=contexts
                )
            assert torch.equal(actual, expected)
            assert all(
                s["cache"]["h2d_transfers"] == 0 for s in direct.batch_forward_stats()
            )
        assert direct.pool.rdma_gpu_read_bytes > 0
        assert direct.pool.rdma_gpu_write_bytes == direct.pool.write_bytes
        assert direct.pool.rdma_gpu_read_bytes == direct.pool.read_bytes
    finally:
        direct.close()
        reference.close()


@pytest.mark.gpu
@pytest.mark.rdma
def test_registration_error_is_fatal_without_cpu_fallback(rdma_options, monkeypatch):
    pool = MemoryPool(1, 64, **rdma_options)
    pool.allocate(1)

    def fail(*args):
        raise RuntimeError("injected DMA-BUF registration failure")

    monkeypatch.setattr(CudaDriver, "export", fail)
    monkeypatch.setattr(Region, "transfer", forbid_staging)
    try:
        with pytest.raises(RuntimeError, match="DMA-BUF"):
            pool.write_many(
                [0], torch.ones((1, 64), device="cuda", dtype=torch.bfloat16)
            )
        assert pool.write_bytes == pool.write_operations == 0
    finally:
        pool.close()


def test_pool_rejects_cpu_payload_without_transport_call(monkeypatch):
    pool = MemoryPool(1, 4)
    pool.backend = "rdma"
    monkeypatch.setattr(
        pool,
        "remote",
        SimpleNamespace(region=SimpleNamespace(transfer_cuda=forbid_staging)),
    )
    try:
        with pytest.raises(ValueError, match="cuda"):
            pool.write_many([0], torch.ones((1, 4), dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="CPU staging"):
            pool.read_many([0], torch.empty((1, 4), dtype=torch.bfloat16))
    finally:
        pool.remote = None
        pool.close()


@pytest.mark.gpu
@pytest.mark.rdma
@pytest.mark.parametrize("invalid", ["remote", "local"])
def test_native_bounds_error_discards_qp_before_releasing_gpu_mr(rdma_options, invalid):
    pool = MemoryPool(1, 64, **rdma_options)
    pool.allocate(1)
    tensor = torch.zeros((2, 64), device="cuda", dtype=torch.bfloat16)
    slots = [0, 64] if invalid == "remote" else [0, 1]
    rows = [0, 1] if invalid == "remote" else [0, 2]
    try:
        with pytest.raises(RuntimeError, match="span bounds"):
            pool.read_rdma_cuda(slots, tensor, rows)
        with pytest.raises(RuntimeError, match="session unusable"):
            pool.write_many([0, 1], tensor)
        assert pool.read_bytes == pool.write_bytes == 0
    finally:
        pool.close()
