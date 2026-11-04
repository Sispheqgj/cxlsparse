"""Use the real upstream CUDA kernel on registered anonymous host mappings.

These tests do not claim to exercise CXL hardware; scripts/smoke.py covers devdax.
"""

import pytest
import torch
from kv_helpers import LengthOnlySlots, assert_history_equal, assert_write_slots

from cxl_recsys.common.request import placeholder
from cxl_recsys.mempool import MemoryPool
from cxl_recsys.mempool.cuda_transfer import RegisteredCudaMapping, load_transfer_kernel
from cxl_recsys.models.simulation.model import TestModel
from cxl_recsys.worker.inference import execute_request


@pytest.fixture
def mapped_pool():
    pool = MemoryPool(16, 64)
    # Exercise the identical registration/write path without writing /dev/dax.
    pool.backend = "cxl"
    yield pool
    pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("dim", [64, 576])
def test_upstream_kernel_scatter_cross_page_without_source_d2h(dim, monkeypatch):
    pool = MemoryPool(4, dim)
    pool.backend = "cxl"
    pool.allocate(4)
    slots = list(range(62, 194))[::-1] + [0, 255]
    expected = torch.arange(len(slots) * dim).reshape(len(slots), dim).bfloat16()
    source = expected.cuda()

    def forbid_cpu(*args, **kwargs):
        raise AssertionError("Source KV must not call Tensor.cpu()")

    try:
        with monkeypatch.context() as guard:
            guard.setattr(torch.Tensor, "cpu", forbid_cpu)
            pool.write_many(slots, source)
        target = torch.empty_like(expected)
        pool.read_many(slots, target)
        assert torch.equal(target, expected)
        gap = torch.empty((1, dim), dtype=torch.bfloat16)
        pool.read_many([61], gap)
        assert bool((gap == 0).all())
        assert pool.write_operations == pool.write_kernel_launches == 1
        assert pool.write_bytes == source.numel() * 2
        assert "write_verify_readback_bytes" not in pool.stats()
        writer = pool.cuda_mapping
        assert writer is not None
        pool.close()
        assert writer.view is None
        pool.close()  # Idempotent; no second cudaHostUnregister.
    finally:
        pool.close()


@pytest.mark.gpu
def test_non_default_stream_write_observes_producer_and_finishes_before_return(
    mapped_pool,
):
    pool = mapped_pool
    pool.allocate(1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.full((64, 64), 13.0, device="cuda", dtype=torch.bfloat16)
        pool.write_many(list(range(64)), source)
        assert stream.query()
    output = torch.empty((64, 64), dtype=torch.bfloat16)
    pool.read_many(list(range(64)), output)
    assert bool((output == 13).all())


@pytest.mark.gpu
@pytest.mark.parametrize(
    "invalid",
    ["shape", "dtype", "noncontiguous"],
)
def test_bad_cuda_batch_rejected_before_registration_or_launch(mapped_pool, invalid):
    pool = mapped_pool
    pool.allocate(1)
    slots = [0, 1]
    source = torch.zeros((2, 64), dtype=torch.bfloat16, device="cuda")
    if invalid == "shape":
        source = source.reshape(1, 128)
    elif invalid == "dtype":
        source = source.float()
    else:
        source = torch.zeros((2, 128), dtype=torch.bfloat16, device="cuda")[:, ::2]
    with pytest.raises(ValueError, match="Batch IO"):
        pool.write_many(slots, source)
    assert pool.cuda_mapping is None
    assert pool.write_bytes == pool.write_kernel_launches == 0


@pytest.mark.gpu
@pytest.mark.parametrize("count", [0, 2, 4096])
def test_cuda_write_validation_never_scans_slots(mapped_pool, count):
    source = torch.empty((count, 64), dtype=torch.bfloat16, device="cuda")
    slots = LengthOnlySlots(count)
    mapped_pool._validate_batch(slots, source, device_type="cuda")
    writer = mapped_pool._cuda_mapping(source.device)
    writer._validate(slots, source)
    # This is a metadata-only test; never send fabricated slots to a CUDA kernel.
    assert mapped_pool.write_bytes == mapped_pool.write_kernel_launches == 0


@pytest.mark.gpu
def test_empty_cuda_batch_does_not_register(mapped_pool):
    pool = mapped_pool
    pool.write_many([], torch.empty((0, 64), dtype=torch.bfloat16, device="cuda"))
    assert pool.cuda_mapping is None


def test_unsupported_kernel_width_rejected():
    with pytest.raises(ValueError, match="128"):
        load_transfer_kernel(8)


@pytest.mark.gpu
def test_cxl_model_no_cpu_kv_write_and_identical_backing_kv(mapped_pool, monkeypatch):
    model = TestModel(mapped_pool, max_users=1, max_history=256)
    reference = TestModel(MemoryPool(16, 64), max_users=1, max_history=256)
    write_many = mapped_pool.write_many
    sources = []

    def record(slots, tensor):
        assert tensor.is_cuda
        assert_write_slots(mapped_pool, slots)
        sources.append(len(tensor))
        write_many(slots, tensor)

    monkeypatch.setattr(mapped_pool, "write_many", record)
    try:
        for length in (3, 128, 128):
            request = placeholder(history_length=length)
            execute_request(model, request)
            execute_request(reference, request)
            assert_history_equal(model, reference)
        assert sources == [3, 3, 125, 125]
        assert mapped_pool.write_kernel_launches == 4
    finally:
        reference.close()


@pytest.mark.gpu
def test_zero_prefix_extend_has_no_verification_readback_or_metadata(
    mapped_pool, monkeypatch
):
    def forbid(*args, **kwargs):
        raise AssertionError("Zero-prefix Extend must not read KV back")

    monkeypatch.setattr(mapped_pool, "read_many", forbid)
    monkeypatch.setattr(mapped_pool, "read_cuda", forbid)
    model = TestModel(mapped_pool, max_users=1, max_history=64)
    result = execute_request(model, placeholder(history_length=64))
    assert mapped_pool.write_kernel_launches == 2
    assert mapped_pool.read_bytes == 0
    assert "verified" not in result
    assert "digests" not in model.users["user-0"]
    assert not hasattr(mapped_pool, "verify_cuda_write")
    assert not hasattr(mapped_pool.cuda_mapping, "verify_and_digest")


@pytest.mark.gpu
def test_direct_write_is_not_read_back_for_integrity_checks(mapped_pool, monkeypatch):
    model = TestModel(mapped_pool, max_users=1, max_history=64)
    execute_request(model, placeholder(history_length=3))
    write = RegisteredCudaMapping.write

    def corrupt(self, slots, source, *, synchronize=True):
        write(self, slots, source, synchronize=synchronize)
        assert self.view is not None
        self.view[slots[-1], -1] = -123

    monkeypatch.setattr(RegisteredCudaMapping, "write", corrupt)
    result = execute_request(model, placeholder(history_length=6))
    assert result["kv_length"] == 6
    assert len(mapped_pool.allocated) == 2
    writer = mapped_pool.cuda_mapping
    assert writer is not None and writer.view is not None
    assert writer.view[model.users["user-0"]["slots"][1][-1], -1] == -123


@pytest.mark.gpu
def test_registration_failure_propagates_without_cpu_fallback_or_rollback(
    mapped_pool, monkeypatch
):
    import cxl_recsys.mempool.pool as pool_module

    def fail(*args, **kwargs):
        raise RuntimeError("injected registration failure")

    monkeypatch.setattr(pool_module, "RegisteredCudaMapping", fail)
    model = TestModel(mapped_pool, max_users=1, max_history=64)
    with pytest.raises(RuntimeError, match="registration"):
        execute_request(model, placeholder(history_length=3))
    assert model.snapshot()["user-0"]["kv_length"] == 3
    assert len(mapped_pool.allocated) == 2
    assert mapped_pool.write_bytes == mapped_pool.write_kernel_launches == 0


@pytest.mark.gpu
def test_second_layer_failure_does_not_release_or_restore_state(
    mapped_pool, monkeypatch
):
    model = TestModel(mapped_pool, max_users=1, max_history=128)
    execute_request(model, placeholder(history_length=3))
    write = mapped_pool.write_many
    calls = 0

    def fail(slots, tensor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected layer failure")
        write(slots, tensor)

    with monkeypatch.context() as guard:
        guard.setattr(mapped_pool, "write_many", fail)
        with pytest.raises(OSError):
            execute_request(model, placeholder(history_length=128))
    assert model.snapshot()["user-0"]["kv_length"] == 128
    assert all(len(slots) == 128 for slots in model.users["user-0"]["slots"])
    assert len(mapped_pool.allocated) == 4
    assert model.forward_stats() == {}
