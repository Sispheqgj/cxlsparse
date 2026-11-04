import threading

import pytest
import torch
from kv_helpers import history_kv

from cxl_recsys.common.request import placeholder, validate
from cxl_recsys.mempool.pool import MemoryPool
from cxl_recsys.worker import Worker
from cxl_recsys.worker.inference import execute_request


def test_request():
    r = placeholder()
    assert validate(r) == r
    for field, value in [
        ("history_token_ids", []),
        ("history_token_ids", [True]),
        ("user_id", ""),
    ]:
        with pytest.raises(ValueError):
            validate({**r, field: value})
    with pytest.raises(ValueError):
        validate({**r, "history_version": 1})


def test_pool():
    p = MemoryPool(3, 64)
    try:
        slots = p.allocate(3)
        with pytest.raises(ValueError):
            p.allocate(1)
        v = torch.arange(64).to(torch.bfloat16)
        out = torch.empty_like(v)
        token_slot = p.token_slot(slots[1], 63)
        p.write(token_slot, v)
        p.read_into(token_slot, out)
        assert torch.equal(v, out)
        p.release(slots)
        assert len(p.free) == 3
        with pytest.raises(ValueError):
            p.release(slots)
    finally:
        p.close()


def test_closed_pool_rejects_local_io():
    pool = MemoryPool(1, 64)
    slot = pool.allocate(1)[0]
    vector = torch.zeros(64, dtype=torch.bfloat16)
    pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        pool.write(slot, vector)
    with pytest.raises(RuntimeError, match="closed"):
        pool.read_into(slot, vector)


class FakeModel:
    max_history = 4096
    layers = 2

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.pool = self
        self.calls = 0

    def forward(self, input_ids, positions, *, seq_indptr, caches):
        self.entered.set()
        self.release.wait(5)
        self.calls += 1
        self.stats = {"input_tokens": len(input_ids), "cache": {}}
        return torch.zeros(len(input_ids), 64)

    def batch_forward_stats(self):
        return [self.stats.copy()]

    def snapshot(self):
        return {}

    def close(self):
        pass


def test_worker_backpressure_dedup():
    e = FakeModel()
    w = Worker(e, queue_capacity=1)
    try:
        r = placeholder()
        assert w.submit(r)["accepted"]
        assert e.entered.wait(2)
        assert w.submit(r)["duplicate"]
        assert w.submit(placeholder(request_id="r2"))["accepted"]
        assert not w.submit(placeholder(request_id="r3"))["accepted"]
        with pytest.raises(ValueError):
            w.submit({**r, "history_token_ids": [99]})
        e.release.set()
        w.jobs.join()
        assert e.calls == 2
        assert w.get({"request_id": r["request_id"]})["status"] == "SUCCEEDED"
    finally:
        e.release.set()
        w.close()


@pytest.mark.gpu
def test_hisparse_append_eviction_and_short_history():
    from cxl_recsys.models.simulation.model import TestModel

    p = MemoryPool(4096, 64)
    try:
        e = TestModel(p, max_users=2, max_history=256)
        first = execute_request(e, placeholder(history_length=3))
        assert first["cache"]["misses"] == 0  # Zero-prefix Extend uses current GPU KV.
        assert first["cache"]["hits"] == 0
        assert first["cache"]["local_tokens"] > 0
        assert first["kv_length"] == 3
        assert first["mode"] == "extend"
        assert p.stats()["allocated_pages"] == 2
        second = execute_request(
            e, placeholder(request_id="append", history_length=128)
        )
        assert second["mode"] == "extend"
        assert second["cache"]["new_pages"] == 2
        assert second["cache"]["new_tokens"] == 250
        stable = execute_request(
            e, placeholder(request_id="repeat", history_length=128)
        )
        assert stable["cache"]["new_pages"] == 0
        warm = execute_request(
            e, placeholder(request_id="repeat-hot", history_length=128)
        )
        # The full prompt has more query positions than the old branch model;
        # its working set may exceed hot=32 even on a repeated request.
        assert warm["prompt"] == stable["prompt"]
        assert warm["cache"]["new_tokens"] == 0
        assert warm["postprocessing"] == "not_implemented"
        expected = history_kv(e)
        # Force hot-set replacement, including the previously cold newest token.
        for indices in (
            [0, 1, 2, 3, 4, 5, 6, 7],
            [120, 121, 122, 123, 124, 125, 126, 127],
            *[list(range(i, i + 8)) for i in range(8, 112, 8)],
            [0, 1, 2, 3, 4, 5, 6, 127],
        ):
            pages, _ = e.cache.fetch(0, indices, 128)
            assert torch.equal(pages.cpu(), expected[indices])
        with pytest.raises(ValueError):
            execute_request(e, placeholder(history_length=127))
        bad = placeholder(history_length=129)
        bad["history_token_ids"][0] = 777
        with pytest.raises(ValueError):
            execute_request(e, bad)
        assert e.snapshot()["user-0"]["kv_length"] == 128
    finally:
        p.close()


@pytest.mark.gpu
def test_failed_write_leaves_partial_state_without_rollback():
    from cxl_recsys.models.simulation.model import TestModel

    p = MemoryPool(1024, 64)
    try:
        e = TestModel(p, max_users=1, max_history=256)
        execute_request(e, placeholder(history_length=3))
        real_write = p.write_many
        calls = 0

        def fail(slots, tensor, *, asynchronous=False):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second-layer write failure")
            real_write(slots, tensor, asynchronous=asynchronous)

        p.write_many = fail
        with pytest.raises(OSError):
            execute_request(e, placeholder(history_length=6))
        assert e.snapshot()["user-0"]["kv_length"] == 6
        assert all(len(slots) == 6 for slots in e.users["user-0"]["slots"])
        assert len(p.allocated) == 2
        assert e.forward_stats() == {}
        # Partial state is invalid. Close the model; never retry this run.
    finally:
        p.close()


@pytest.mark.gpu
def test_failed_fetch_propagates_before_hot_update():
    from cxl_recsys.models.simulation.model import TestModel

    p = MemoryPool(1024, 64)
    try:
        e = TestModel(p, max_users=1, max_history=256)
        execute_request(e, placeholder(history_length=64))
        e.cache.invalidate()

        def fail(slots, tensor, *, asynchronous=False):
            raise OSError("injected read failure")

        p.read_many = fail
        with pytest.raises(OSError):
            e.cache.fetch(0, [0, 1, 2, 3, 4, 5, 6, 7], 64)
        # The hot update is after IO; no rollback/protection is involved.
        assert bool((e.cache.tags == -1).all())
    finally:
        p.close()


@pytest.mark.gpu
def test_mla_576_and_single_token():
    from cxl_recsys.models.simulation.model import TestModel

    p = MemoryPool(1024, 576)
    try:
        e = TestModel(p, max_users=1, max_history=256, dim=576)
        first = execute_request(e, placeholder(history_length=1))
        assert first["cache"]["misses"] == 0
        assert first["cache"]["hits"] == 0
        assert first["kv_layout"]["latent_dim"] == 512
        assert first["kv_layout"]["rope_dim"] == 64
        second = execute_request(e, placeholder(history_length=64))
        assert second["kv_length"] == 64
        assert p.token_bytes == 1152
        assert p.page_bytes == 64 * 1152
    finally:
        p.close()


@pytest.mark.gpu
@pytest.mark.parametrize("page_size", [1, 64])
def test_paged_append_and_sparse_token_reads(page_size):
    from cxl_recsys.models.simulation.model import TestModel

    pool = MemoryPool(512, 64, page_size=page_size)
    try:
        model = TestModel(pool, max_users=2, max_history=256)
        previous_pages = 0
        for length in (1, 63, 64, 65, 127, 128, 129):
            result = execute_request(model, placeholder(history_length=length))
            expected_pages = 2 * ((length + page_size - 1) // page_size)
            assert result["cache"]["new_pages"] == expected_pages - previous_pages
            assert len(pool.allocated) == expected_pages
            previous_pages = expected_pages
        # Interleaved layer page allocations make flattened locations nontrivial.
        assert model.users["user-0"]["slots"][0][64] != 64
        model.cache.invalidate()
        expected = history_kv(model)
        before = pool.read_bytes
        indices = [0, 63, 64, 65, 127, 128]
        values, counters = model.cache.fetch(0, indices, 129)
        assert torch.equal(values.cpu(), expected[indices])
        assert counters == {"hits": 0, "misses": 6}
        assert pool.read_bytes - before == 6 * pool.token_bytes
        _, counters = model.cache.fetch(0, indices, 129)
        assert counters == {"hits": 6, "misses": 0}
        assert pool.read_bytes - before == 6 * pool.token_bytes
        assert (
            execute_request(model, placeholder(user_id="user-1", history_length=65))[
                "kv_length"
            ]
            == 65
        )
        assert (
            execute_request(model, placeholder(history_length=129))["kv_length"] == 129
        )
    finally:
        pool.close()


@pytest.mark.gpu
def test_cross_page_failure_keeps_allocations_and_partial_metadata():
    from cxl_recsys.models.simulation.model import TestModel

    pool = MemoryPool(4, 64)
    try:
        model = TestModel(pool, max_users=1, max_history=256)
        execute_request(model, placeholder(history_length=63))
        real_transfer = pool._transfer_span
        calls = 0

        def fail(slot, tensor, write):
            nonlocal calls
            calls += int(write)
            if write and calls == 4:
                raise OSError("cross-page failure")
            real_transfer(slot, tensor, write)

        pool._transfer_span = fail
        with pytest.raises(OSError):
            execute_request(model, placeholder(history_length=65))
        assert len(pool.allocated) == 4
        assert all(len(slots) == 65 for slots in model.users["user-0"]["slots"])
        assert model.snapshot()["user-0"]["kv_length"] == 65
        assert model.forward_stats() == {}
    finally:
        pool.close()


def test_page_token_bounds():
    pool = MemoryPool(2, 64)
    try:
        page = pool.allocate(1)[0]
        value = torch.zeros(64, dtype=torch.bfloat16)
        for slot in (-1, 64, 128):
            with pytest.raises(ValueError):
                pool.read_into(slot, value)
        with pytest.raises(ValueError):
            pool.token_slot(page, 64)
        pool.release([page])
        with pytest.raises(ValueError):
            pool.write(0, value)
    finally:
        pool.close()


@pytest.mark.parametrize("failure", ["forward", "snapshot"])
def test_worker_failure_aborts_queue_and_rejects_more_work(failure):
    class FailingModel(FakeModel):
        def forward(self, input_ids, positions, *, seq_indptr, caches):
            result = super().forward(
                input_ids, positions, seq_indptr=seq_indptr, caches=caches
            )
            if failure == "forward":
                raise OSError("injected forward failure")
            return result

        def snapshot(self):
            raise OSError("injected snapshot failure")

    model = FailingModel()
    worker = Worker(model, queue_capacity=2)
    try:
        assert worker.submit(placeholder(request_id="fail"))["accepted"]
        assert model.entered.wait(2)
        assert worker.submit(placeholder(request_id="queued-1"))["accepted"]
        assert worker.submit(placeholder(request_id="queued-2"))["accepted"]
        model.release.set()
        worker.thread.join(5)
        assert not worker.thread.is_alive()
        assert model.calls == 1
        assert worker.get({"request_id": "fail"})["status"] == "FAILED"
        for rid in ("queued-1", "queued-2"):
            result = worker.get({"request_id": rid})
            assert result["status"] == "ABORTED"
            assert result["run_failed"]
        assert worker.status({})["failed"]
        assert worker.status({})["users"] == {}
        assert worker.status({})["outstanding"] == 0
        assert worker.jobs.unfinished_tasks == 0
        assert worker.submit(placeholder(request_id="later"))["reason"] == "run_failed"
        assert worker.submit(placeholder(request_id="fail"))["reason"] == "run_failed"
    finally:
        model.release.set()
        worker.close()
