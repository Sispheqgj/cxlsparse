"""Batch transport, user FIFO, packed GPU execution and cache isolation."""

import copy
import json
import socket
import threading
import weakref

import pytest

from cxl_recsys.common.batch import validate_batch
from cxl_recsys.common.request import placeholder
from cxl_recsys.common.rpc import serve
from cxl_recsys.models.base import CacheContext
from cxl_recsys.scheduler import Scheduler
from cxl_recsys.worker import Worker
from cxl_recsys.worker.inference import execute_batch
from cxl_recsys.worker.prompt import build_prompt


def requests(lengths=(3, 5, 7), suffix="a"):
    values = []
    for i, length in enumerate(lengths):
        request = placeholder(
            user_id=f"u{i}", request_id=f"u{i}-{suffix}", history_length=length
        )
        request["history_token_ids"] = [1000 * i + j for j in range(length)]
        request["candidates"] = [{"item_id": "item", "token_ids": [200 + i] * (i + 1)}]
        values.append(request)
    return values


class HiddenStates:
    def __getattribute__(self, name):
        raise AssertionError("Worker must not inspect hidden states")


class BatchModel:
    max_history = 4096
    layers = 2

    def __init__(self):
        self.calls, self.users = [], {}
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.fail = False

    def forward(self, ids, positions, *, seq_indptr, caches):
        self.calls.append((ids, positions, seq_indptr, caches))
        self.entered.set()
        assert self.release.wait(5)
        if self.fail:
            raise OSError("injected batch failure")
        self.stats = []
        for i, cache in enumerate(caches):
            old = self.users.get(cache.cache_key, {}).get("kv_length", 0)
            self.stats.append(
                {"input_tokens": seq_indptr[i + 1] - seq_indptr[i] - old, "cache": {}}
            )
            self.users[cache.cache_key] = {
                "kv_length": cache.cacheable_prefix_length,
                "backend": "dram",
            }
        hidden = HiddenStates()
        self.hidden_ref = weakref.ref(hidden)
        return hidden

    def batch_forward_stats(self):
        return self.stats

    def forward_stats(self):
        return self.stats[0]

    def snapshot(self):
        return dict(self.users)

    def close(self):
        pass


def address():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{sock.getsockname()[1]}"


def test_empty_worker_batch_rejected_without_calling_model():
    model = BatchModel()
    with pytest.raises(ValueError, match="empty"):
        execute_batch(model, [])
    assert model.calls == []


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 3])
def test_all_batch_sizes_use_one_layer_path(monkeypatch, batch_size):
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel

    model = TestModel(MemoryPool(12, 64), max_users=3, max_history=128)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("The model must not use single-sequence operators")

    assert not hasattr(model.ops, "select")
    assert not hasattr(model.ops, "index_project")
    monkeypatch.setattr(model.cache, "fetch", forbidden)
    for owner, name in (
        (model.ops, "project"),
        (model.ops, "select_batch"),
        (model.cache, "fetch_layer"),
        (model.ops, "consume"),
    ):
        original = getattr(owner, name)

        def traced(*args, name=name, original=original, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, traced)
    try:
        assert not hasattr(model, "forward_batch")
        assert not hasattr(model.cache, "fetch_layers")
        assert not hasattr(model.cache, "_fetch_old")
        assert not hasattr(model.cache, "_plan_chunk")
        for lengths in ((3, 5, 7), (4, 7, 8), (4, 7, 8)):
            calls.clear()
            results = execute_batch(model, requests(lengths)[:batch_size])
            assert (
                calls
                == ["project", "select_batch", "fetch_layer", "consume"] * model.layers
            )
            assert [result["batch"]["size"] for result in results] == [
                batch_size
            ] * batch_size
            stats = model.batch_forward_stats()
            assert model.forward_stats() == (stats[0] if batch_size == 1 else {})
    finally:
        model.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "invalid", ["missing", "boundary_only", "caches_only", "mixed"]
)
def test_forward_requires_unambiguous_batch_metadata(invalid):
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel

    model = TestModel(MemoryPool(4, 64), max_users=1, max_history=128)
    context = CacheContext("u0", 1)
    kwargs = {
        "missing": {},
        "boundary_only": {"seq_indptr": [0, 2]},
        "caches_only": {"caches": [context]},
        "mixed": {"cache": context, "seq_indptr": [0, 2], "caches": [context]},
    }[invalid]
    try:
        with pytest.raises(ValueError):
            model.forward([1, 2], [0, 1], **kwargs)
        assert model.snapshot() == {}
        assert model.batch_forward_stats() == []
        assert model.forward_stats() == {}
        assert not model.pool.allocated
    finally:
        model.close()


@pytest.mark.parametrize(
    "bad", ["empty", "size", "user", "request", "id", "extra", "payload"]
)
def test_batch_validation_is_atomic(bad):
    value = {"batch_id": "batch", "requests": requests((2, 3))}
    if bad == "empty":
        value["requests"] = []
    elif bad == "size":
        value["requests"] = requests((2, 3, 4))
    elif bad in ("user", "request"):
        value["requests"][1][bad + "_id"] = value["requests"][0][bad + "_id"]
    elif bad == "id":
        value["batch_id"] = ""
    elif bad == "extra":
        value["unexpected"] = 1
    else:
        value["requests"][1]["history_token_ids"] = [True]
    worker = Worker(BatchModel(), max_batch_size=2)
    try:
        with pytest.raises(ValueError):
            worker.submit_batch(value)
        assert not worker.records and not worker.batches
        assert worker.outstanding == 0
    finally:
        worker.close()


def test_worker_packs_one_call_and_preserves_request_order_without_processing_output():
    model = BatchModel()
    worker = Worker(model)
    batch = {"batch_id": "packed", "requests": requests()}
    original = copy.deepcopy(batch)
    try:
        assert worker.submit_batch(batch)["accepted"]
        worker.jobs.join()
        assert len(model.calls) == 1
        ids, positions, indptr, caches = model.calls[0]
        assert indptr == [0, 4, 11, 21]
        for i, request in enumerate(batch["requests"]):
            prompt = build_prompt(request["history_token_ids"], request["candidates"])
            assert ids[indptr[i] : indptr[i + 1]] == prompt.input_ids
            assert positions[indptr[i] : indptr[i + 1]] == prompt.positions
            assert caches[i] == CacheContext(
                request["user_id"], len(request["history_token_ids"])
            )
        assert model.hidden_ref() is None
        reply = worker.get_batch({"batch_id": "packed"})
        assert reply["status"] == "SUCCEEDED"
        assert [r["request_id"] for r in reply["requests"]] == [
            r["request_id"] for r in batch["requests"]
        ]
        assert [r["batch_position"] for r in reply["requests"]] == [0, 1, 2]
        assert all(
            r["result"]["postprocessing"] == "not_implemented"
            for r in reply["requests"]
        )
        assert worker.submit_batch(batch)["duplicate"]
        assert len(model.calls) == 1
        assert batch == original == validate_batch(batch)
        json.dumps(reply)
        with pytest.raises(ValueError, match="batch_id"):
            worker.submit_batch({**batch, "requests": requests((4, 5, 7))})
        with pytest.raises(ValueError, match="request_id"):
            worker.submit_batch({**batch, "batch_id": "different"})
    finally:
        worker.close()


@pytest.mark.parametrize("fail", [False, True])
def test_batch_queue_capacity_and_failure(fail):
    model = BatchModel()
    model.release.clear()
    model.fail = fail
    worker = Worker(model, queue_capacity=1)
    first = {"batch_id": "first", "requests": requests()}
    second = {"batch_id": "second", "requests": requests((4, 6, 8), "b")}
    try:
        assert worker.submit_batch(first)["accepted"]
        assert model.entered.wait(2)
        assert worker.submit_batch(second)["accepted"]
        assert worker.outstanding == 2 and len(worker.records) == 6
        assert (
            worker.submit_batch(
                {"batch_id": "third", "requests": requests((5, 7, 9), "c")}
            )["reason"]
            == "busy"
        )
        model.release.set()
        worker.jobs.join()
        assert worker.get_batch({"batch_id": "first"})["status"] == (
            "FAILED" if fail else "SUCCEEDED"
        )
        assert worker.get_batch({"batch_id": "second"})["status"] == (
            "ABORTED" if fail else "SUCCEEDED"
        )
        assert len(model.calls) == (1 if fail else 2)
        assert worker.outstanding == 0
        if fail:
            assert all(
                r["status"] == "FAILED"
                for r in worker.get_batch({"batch_id": "first"})["requests"]
            )
            assert worker.status({})["users"] == {}
            assert worker.submit_batch(first)["reason"] == "run_failed"
    finally:
        model.release.set()
        worker.close()


@pytest.mark.parametrize(
    "worker_count,batch_size,max_pending", [(1, 4, 100), (2, 2, 100), (1, 4, 1)]
)
def test_scheduler_batch_grouping_fifo_tail_and_sticky_owner(
    worker_count, batch_size, max_pending
):
    workers, servers, targets, submissions = [], [], [], []
    scheduler = None
    try:
        for i in range(worker_count):
            worker = Worker(BatchModel(), max_batch_size=4)
            workers.append(worker)
            target = address()
            targets.append(target)

            def submit(value, worker=worker, index=i):
                submissions.append((index, value))
                return worker.submit_batch(value)

            servers.append(
                serve(
                    target,
                    {
                        "SubmitBatch": submit,
                        "GetBatch": worker.get_batch,
                        "Status": worker.status,
                    },
                )
            )
        scheduler = Scheduler(targets, batch_size=batch_size, max_pending=max_pending)
        # Interleaved same-user dependencies must not share a batch or overtake.
        first, second = requests((2, 3, 4, 5, 6)), requests((3, 4, 5, 6, 7), "b")
        workload = [request for pair in zip(first, second) for request in pair]
        report = scheduler.run_sequence(workload, timeout=10)
        assert report["passed"] and report["succeeded"] == 10
        assert [r["request_id"] for r in report["requests"]] == [
            r["request_id"] for r in workload
        ]
        seen = {f"u{i}": [] for i in range(5)}
        for index, batch in submissions:
            validate_batch(batch, max_batch_size=batch_size)
            for request in batch["requests"]:
                uid = request["user_id"]
                assert index == int(uid[1:]) % worker_count
                seen[uid].append(request["request_id"])
        assert all(ids == [f"{uid}-a", f"{uid}-b"] for uid, ids in seen.items())
        assert len(submissions) < 10 if max_pending > 1 else len(submissions) == 10
        assert max(report["scheduler"]["batch_sizes"]) == (
            batch_size if max_pending > 1 else 1
        )
        assert min(report["scheduler"]["batch_sizes"]) == 1
    finally:
        if scheduler:
            scheduler.close()
        for server in servers:
            server.stop(0).wait()
        for worker in workers:
            worker.close()


@pytest.mark.gpu
def test_metadata_only_entries_preserve_history_and_page_boundaries(monkeypatch):
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel

    model = TestModel(MemoryPool(16, 64), max_users=2, max_history=128)
    expected = []
    expected_histories = {}
    history_lists = {}
    visited_layers = []
    forward_layer = model._forward_layer

    def inspect_entries(entries, *args, **kwargs):
        visited_layers.append(args[-1])
        assert len(entries) == len(expected)
        for entry, (history_length, total_length, new_pages_num) in zip(
            entries, expected
        ):
            assert set(entry) == {
                "cache",
                "state",
                "history_length",
                "total_length",
                "new_pages_num",
                "stats",
            }
            assert entry["history_length"] == history_length
            assert entry["total_length"] == total_length
            assert entry["new_pages_num"] == new_pages_num
            assert entry["stats"]["new_pages"] == new_pages_num * model.layers
            assert (
                entry["stats"]["new_tokens"]
                == (total_length - history_length) * model.layers
            )
            cache_key = entry["cache"].cache_key
            assert entry["state"] is model.users[cache_key]
            # Live state already has the new history; the entry keeps the old bound.
            assert len(entry["state"]["tokens"]) == total_length
            assert entry["state"]["tokens"] == expected_histories[cache_key]
            # Appends and reuse preserve the canonical list rather than copying it.
            if cache_key in history_lists:
                assert entry["state"]["tokens"] is history_lists[cache_key]
            else:
                history_lists[cache_key] = entry["state"]["tokens"]
        return forward_layer(entries, *args, **kwargs)

    monkeypatch.setattr(model, "_forward_layer", inspect_entries)
    try:
        for step, (lengths, boundaries) in enumerate(
            (
                ((3, 65), [(0, 3, 1), (0, 65, 2)]),
                ((66, 67), [(3, 66, 1), (65, 67, 0)]),
                ((66, 67), [(66, 66, 0), (67, 67, 0)]),
            )
        ):
            workload = requests(lengths)
            original = copy.deepcopy(workload)
            expected_histories = {
                request["user_id"]: request["history_token_ids"][:]
                for request in workload
            }
            if step == 2:
                workload.reverse()
                original.reverse()
                boundaries.reverse()
            expected = boundaries
            visited_layers.clear()
            results = execute_batch(model, workload)
            assert workload == original
            assert visited_layers == list(range(model.layers))
            assert [result["kv_length"] for result in results] == [
                len(request["history_token_ids"]) for request in workload
            ]
    finally:
        model.close()


@pytest.mark.gpu
def test_batch_partition_invariance_for_ragged_extend_reuse_and_reordered_users(
    monkeypatch,
):
    import torch
    from kv_helpers import history_kv

    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel

    packed = TestModel(
        MemoryPool(32, 64), max_users=3, max_history=256, layers=4, hot=4, topk=2
    )
    serial = TestModel(
        MemoryPool(32, 64), max_users=3, max_history=256, layers=4, hot=4, topk=2
    )
    calls = []
    for name in ("project", "select_batch", "consume"):
        original = getattr(packed.ops, name)

        def traced(*args, name=name, original=original, **kwargs):
            calls.append((name, len(args[0])))
            return original(*args, **kwargs)

        monkeypatch.setattr(packed.ops, name, traced)
    outputs = []
    forward = packed.forward

    def capture(*args, **kwargs):
        value = forward(*args, **kwargs)
        outputs.append(value.clone())
        return value

    monkeypatch.setattr(packed, "forward", capture)
    try:
        for step, lengths in enumerate(((3, 65, 7), (66, 67, 128), (66, 67, 128))):
            values = requests(lengths, str(step))
            if step == 2:
                values.reverse()
            calls.clear()
            results = execute_batch(packed, values)
            expected = []
            for request in values:
                prompt = build_prompt(
                    request["history_token_ids"], request["candidates"]
                )
                expected.append(
                    serial.forward(
                        prompt.input_ids,
                        prompt.positions,
                        cache=CacheContext(request["user_id"], prompt.history_length),
                    )
                )
            assert torch.equal(outputs[-1], torch.cat(expected))
            tokens = sum(len(value) for value in expected)
            assert calls == [
                (name, tokens)
                for _ in range(4)
                for name in ("project", "select_batch", "consume")
            ]
            for layer in range(4):
                for request in values:
                    uid = request["user_id"]
                    assert torch.equal(
                        history_kv(packed, layer, uid), history_kv(serial, layer, uid)
                    )
            assert sum(result["cache"]["queries"] for result in results) == tokens * 4
            assert [result["batch"]["position"] for result in results] == [0, 1, 2]
            assert [packed.users[f"u{i}"]["row"] for i in range(3)] == [0, 4, 8]
        assert packed.snapshot() == serial.snapshot()
    finally:
        packed.close()
        serial.close()


@pytest.mark.gpu
def test_multirow_hisparse_plans_in_parallel_and_reads_all_misses_once(monkeypatch):
    import torch

    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.mempool.hisparse import HiSparseCache

    pool = MemoryPool(4, 4)
    pool.allocate(4)
    source = torch.arange(256 * 4).reshape(256, 4).bfloat16()
    pool.write_many(list(range(256)), source)
    cache = HiSparseCache(pool, rows=4, max_history=128, hot=4, topk=2, dim=4)
    cache.bind(1, list(range(128)))
    cache.bind(3, list(range(128, 256)))
    cache.fetch(3, [10, 11], 128)
    launches, reads = [], []
    original_plan = cache._plan_layer

    def plan(indices, descriptors, current, work):
        launches.append((indices.shape, descriptors[:, 0].tolist()))
        return original_plan(indices, descriptors, current, work)

    monkeypatch.setattr(cache, "_plan_layer", plan)
    read = pool.read_many

    def transfer(slots, destination):
        reads.append(list(slots))
        return read(slots, destination)

    monkeypatch.setattr(pool, "read_many", transfer)
    selected = torch.arange(12, device="cuda").reshape(6, 2)
    fetches = [
        {
            "row": row,
            "indices": selected,
            "history_length": 128,
            "cached_length": 128,
            "current_kv": torch.zeros((6, 4), device="cuda", dtype=torch.bfloat16),
        }
        for row in (1, 3)
    ]
    try:
        batch, counts = cache.fetch_layer(fetches)
        assert isinstance(counts, list)
        assert batch.sequence_indptr == [0, 12, 24]
        assert launches == [(torch.Size([12, 2]), [1, 3])]
        assert len(reads) == 1 and len(reads[0]) == 22
        assert [count["hits"] for count in counts] == [0, 2]
        assert sum(count["h2d_transfers"] for count in counts) == 1
        expected = torch.cat((source[selected.cpu()], source[selected.cpu() + 128]))
        assert torch.equal(batch.values[batch.indices].cpu(), expected)
        assert bool((cache.tags[[0, 2]] == -1).all())
        # Later hot-set replacement cannot corrupt the packed attention input.
        cache.fetch(1, [100, 101], 128)
        assert torch.equal(batch.values[batch.indices].cpu(), expected)
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dram", "cxl"])
def test_hisparse_ragged_rows_mix_old_current_and_padding(backend):
    import torch

    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.mempool.hisparse import HiSparseCache

    pool = MemoryPool(2, 64)
    pool.allocate(2)
    source = torch.arange(128).reshape(2, 64).bfloat16()
    pool.write_many([0, 1], source)
    pool.backend = backend  # CXL mode uses a registered anonymous test mapping.
    cache = HiSparseCache(pool, rows=4, max_history=128, hot=4, topk=2, dim=64)
    cache.bind(1, [0, 1, 2])
    cache.bind(3, [64, 65])
    current = [
        torch.full((3, 64), 77, device="cuda", dtype=torch.bfloat16),
        torch.full((2, 64), 99, device="cuda", dtype=torch.bfloat16),
    ]
    selected = [
        torch.tensor([[0, 2], [2, 3], [1, 4]], device="cuda"),
        torch.tensor([[0, -1], [0, 1]], device="cuda"),
    ]
    fetches = [
        {
            "row": 1,
            "indices": selected[0],
            "history_length": 3,
            "cached_length": 2,
            "current_kv": current[0],
        },
        {
            "row": 3,
            "indices": selected[1],
            "history_length": 2,
            "cached_length": 0,
            "current_kv": current[1],
        },
    ]
    try:
        for warm in (False, True):
            before = pool.read_bytes
            batch, counters = cache.fetch_layer(fetches)
            assert isinstance(counters, list)
            assert batch.sequence_indptr == [0, 5, 7]
            assert batch.logical_positions == [0, 1, 2, 3, 4, 0, 1]
            assert batch.cached_tokens == 2
            assert torch.equal(
                batch.values, torch.cat((source.cuda(), current[0], current[1]))
            )
            assert torch.equal(batch.indices[:3], selected[0])
            assert torch.equal(
                batch.indices[3:],
                torch.tensor([[5, -1], [5, 6]], device="cuda"),
            )
            assert [item["hits"] for item in counters] == [2 if warm else 0, 0]
            assert [item["misses"] for item in counters] == [0 if warm else 2, 0]
            assert [item["local_tokens"] for item in counters] == [4, 3]
            assert pool.read_bytes - before == (0 if warm else 2 * pool.token_bytes)
            assert sum(item["h2d_transfers"] for item in counters) == int(
                not warm and backend == "dram"
            )
            assert sum(item["cxl_read_kernel_launches"] for item in counters) == int(
                not warm and backend == "cxl"
            )
            assert bool((cache.tags[[0, 2]] == -1).all())
            assert cache.tags[1, :4].tolist() == [2, 0, 1, -1]
            assert cache.tags[3, :4].tolist() == [0, 1, -1, -1]
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("invalid", ["empty", "duplicate", "width", "row", "future"])
def test_hisparse_rejects_invalid_batch_before_io_or_hot_update(invalid, monkeypatch):
    import torch

    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.mempool.hisparse import HiSparseCache

    pool = MemoryPool(2, 64)
    cache = HiSparseCache(pool, rows=2, max_history=128, hot=4, topk=2, dim=64)
    fetches = [
        {
            "row": row,
            "indices": torch.tensor([[0, -1]], device="cuda"),
            "history_length": 1,
            "cached_length": 0,
            "current_kv": torch.zeros((1, 64), device="cuda", dtype=torch.bfloat16),
        }
        for row in range(2)
    ]
    if invalid == "empty":
        fetches = []
    elif invalid == "duplicate":
        fetches[1]["row"] = 0
    elif invalid == "width":
        fetches[1]["indices"] = torch.tensor([[0]], device="cuda")
    elif invalid == "row":
        fetches[1]["row"] = 2
    else:
        fetches[1]["indices"] = torch.tensor([[0, 1]], device="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid batch must not read KV")

    monkeypatch.setattr(pool, "read_many", forbidden)
    try:
        with pytest.raises(ValueError):
            cache.fetch_layer(fetches)
        assert pool.read_bytes == 0
        assert bool((cache.tags == -1).all())
    finally:
        pool.close()


@pytest.mark.gpu
def test_single_query_wrapper_uses_layer_batch_and_preserves_order(monkeypatch):
    import torch

    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.mempool.hisparse import HiSparseCache

    pool = MemoryPool(1, 64)
    pool.allocate(1)
    source = torch.arange(8 * 64).reshape(8, 64).bfloat16()
    pool.write_many(list(range(8)), source)
    cache = HiSparseCache(pool, rows=1, max_history=64, hot=4, topk=4, dim=64)
    cache.bind(0, list(range(8)))
    calls = []
    fetch_layer = cache.fetch_layer

    def traced(fetches):
        calls.append(len(fetches))
        return fetch_layer(fetches)

    monkeypatch.setattr(cache, "fetch_layer", traced)
    try:
        selected = [7, 0, 5, 2]
        for warm in (False, True):
            values, counts = cache.fetch(0, selected, 8)
            assert torch.equal(values.cpu(), source[selected])
            assert counts == {"hits": 4 if warm else 0, "misses": 0 if warm else 4}
        assert calls == [1, 1]
        empty, counts = cache.fetch(0, [], 8)
        assert empty.shape == (0, 64) and counts == {"hits": 0, "misses": 0}
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "invalid", ["duplicate_user", "boundary", "append", "capacity"]
)
def test_invalid_packed_inputs_do_not_modify_any_user(invalid):
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel

    model = TestModel(MemoryPool(8, 64), max_users=2, max_history=128)
    try:
        model.forward([1, 2], [0, 1], cache=CacheContext("u0", 1))
        before = model.snapshot()
        allocated, writes = set(model.pool.allocated), model.pool.write_bytes
        ids, pos, ptr = [1, 2, 9, 3, 4], [0, 1, 2, 0, 1], [0, 3, 5]
        caches = [CacheContext("u0", 2), CacheContext("u1", 1)]
        if invalid == "duplicate_user":
            caches[1] = CacheContext("u0", 1)
        elif invalid == "boundary":
            ptr = [0, 0, 5]
        elif invalid == "append":
            caches.reverse()
        else:
            caches = [CacheContext("u1", 2), CacheContext("u2", 1)]
        with pytest.raises(ValueError):
            model.forward(ids, pos, seq_indptr=ptr, caches=caches)
        assert model.snapshot() == before
        assert model.pool.allocated == allocated
        assert model.pool.write_bytes == writes
        assert model.batch_forward_stats() == []
    finally:
        model.close()


def test_worker_record_limit_rejects_whole_batch():
    worker = Worker(BatchModel(), max_records=2)
    try:
        assert (
            worker.submit_batch({"batch_id": "too-many", "requests": requests()})[
                "reason"
            ]
            == "record_capacity"
        )
        assert worker.records == {} and worker.batches == {}
        assert worker.outstanding == 0
    finally:
        worker.close()


@pytest.mark.parametrize(
    "cls,kwargs", [(Scheduler, {"batch_size": 0}), (Worker, {"max_batch_size": 0})]
)
def test_nonpositive_batch_limit_rejected(cls, kwargs):
    with pytest.raises(ValueError):
        cls(["127.0.0.1:1"] if cls is Scheduler else BatchModel(), **kwargs)


@pytest.mark.gpu
def test_packed_mixed_warm_and_new_users_and_failure(monkeypatch):
    import torch

    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel
    from cxl_recsys.worker.inference import execute_request

    packed = TestModel(MemoryPool(16, 64), max_users=3, max_history=128, hot=4, topk=2)
    serial = TestModel(MemoryPool(16, 64), max_users=3, max_history=128, hot=4, topk=2)
    try:
        initial = requests((3, 4))[1]
        execute_request(packed, initial)
        execute_request(serial, initial)
        values = requests((5, 7, 8), "b")
        ids, positions, ptr, contexts = [], [], [0], []
        reference = []
        for request in values:
            prompt = build_prompt(request["history_token_ids"], request["candidates"])
            context = CacheContext(request["user_id"], prompt.history_length)
            ids.extend(prompt.input_ids)
            positions.extend(prompt.positions)
            ptr.append(len(ids))
            contexts.append(context)
            reference.append(
                serial.forward(prompt.input_ids, prompt.positions, cache=context)
            )
        output = packed.forward(ids, positions, seq_indptr=ptr, caches=contexts)
        assert torch.equal(output, torch.cat(reference))
        assert [packed.users[f"u{i}"]["row"] for i in range(3)] == [2, 0, 4]

        def fail(*args):
            raise OSError("batched attention failed")

        monkeypatch.setattr(packed.ops, "consume", fail)
        with pytest.raises(OSError, match="batched attention"):
            execute_batch(packed, requests((6, 8, 9), "c"))
        assert packed.batch_forward_stats() == []
        # State is deliberately live, not restored after a partial layer.
        assert packed.snapshot()["u0"]["kv_length"] == 6
    finally:
        packed.close()
        serial.close()
