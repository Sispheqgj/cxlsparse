"""Task timing boundaries, queueing, GPU completion and optional breakdown."""

import json
import logging
import threading
import time

import pytest
from test_batching import BatchModel, requests

from cxl_recsys.common.timing import StageTimer
from cxl_recsys.worker import Worker

LAYER_PHASES = (
    "before_kv_write_ms",
    "kv_write_ms",
    "query_absorb_ms",
    "indexer_ms",
    "kv_read_ms",
    "attention_ms",
    "ffn_ms",
)


def assert_breakdown(reply, layers):
    breakdown = reply["timing_breakdown"]
    assert breakdown["total_ms"] == reply["execution_ms"]
    assert sum(
        breakdown[name] for name in ("preprocess_ms", "model_ms", "postprocess_ms")
    ) == pytest.approx(reply["execution_ms"])
    model = breakdown["model"]
    assert len(model["layers"]) == layers
    assert [item["layer"] for item in model["layers"]] == list(range(layers))
    for item in model["layers"]:
        assert all(item[name] >= 0 for name in LAYER_PHASES)
        assert sum(item[name] for name in LAYER_PHASES) == pytest.approx(
            item["total_ms"]
        )
    assert model["layers_ms"] == pytest.approx(
        sum(item["total_ms"] for item in model["layers"]) + model["layer_overhead_ms"]
    )
    assert model["total_ms"] == pytest.approx(
        model["prepare_ms"] + model["layers_ms"] + model["finalize_ms"]
    )
    assert breakdown["model_ms"] == pytest.approx(
        model["total_ms"] + model["call_overhead_ms"]
    )
    assert all(
        model[name] >= 0
        for name in (
            "prepare_ms",
            "layers_ms",
            "finalize_ms",
            "layer_overhead_ms",
            "call_overhead_ms",
        )
    )


def test_timer_counts_completion_wait_and_cpu_phases(monkeypatch):
    clock = [1_000_000]
    calls = []
    monkeypatch.setattr(time, "perf_counter_ns", lambda: clock[0])

    def synchronize():
        calls.append(True)
        clock[0] += 4_000_000

    timer = StageTimer(synchronize)
    clock[0] += 2_000_000
    timer.mark("gpu_ms")
    clock[0] += 3_000_000
    timer.mark("cpu_ms", synchronize=False)
    assert timer.report() == {"gpu_ms": 6.0, "cpu_ms": 3.0, "total_ms": 9.0}
    assert len(calls) == 1


def test_queue_time_is_separate_and_duplicate_does_not_reset_it(caplog):
    caplog.set_level(logging.DEBUG)
    model = BatchModel()
    model.release.clear()
    worker = Worker(model, queue_capacity=1)
    first = {"batch_id": "first", "requests": requests((3,), "a")}
    second = {"batch_id": "second", "requests": requests((5,), "b")}
    try:
        assert worker.submit_batch(first)["accepted"]
        assert model.entered.wait(5)
        assert worker.submit_batch(second)["accepted"]
        queued_ns = worker.batches["second"]["queued_ns"]
        assert worker.get_batch({"batch_id": "second"})["status"] == "QUEUED"
        # The first task holds the executor while the second task waits in queue.
        assert not threading.Event().wait(0.03)
        assert worker.submit_batch(second)["duplicate"]
        assert worker.batches["second"]["queued_ns"] == queued_ns
        model.release.set()
        worker.jobs.join()
        a = worker.get_batch({"batch_id": "first"})
        b = worker.get_batch({"batch_id": "second"})
        assert a["execution_ms"] >= 30
        assert b["queue_wait_ms"] >= 30
        assert b["execution_ms"] < b["queue_wait_ms"]
        for reply in (a, b):
            assert reply["status"] == "SUCCEEDED"
            assert reply["execution_ms"] == reply["elapsed_ms"] >= 0
            assert reply["timing_scope"] == "batch"
            assert "timing_breakdown" not in reply
            assert "queued_ns" not in reply
            assert reply["requests"][0]["queue_wait_ms"] == reply["queue_wait_ms"]
            assert reply["requests"][0]["execution_ms"] == reply["execution_ms"]
            json.dumps(reply)
        assert "queue_wait_ms=" in caplog.text
        assert "execution_ms=" in caplog.text
        assert "batch_timing" not in caplog.text
    finally:
        model.release.set()
        worker.close()


class AsyncModel(BatchModel):
    """CPU stand-in: forward returns before device completion is signaled."""

    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()
        self.device_done = threading.Event()
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1
        self.waiting.set()
        assert self.device_done.wait(5)


def test_normal_timing_waits_for_async_model_without_inspecting_hidden_states():
    model = AsyncModel()
    worker = Worker(model)
    batch = {"batch_id": "async", "requests": requests((3, 5))}
    try:
        worker.submit_batch(batch)
        assert model.waiting.wait(5)
        assert worker.get_batch({"batch_id": "async"})["status"] == "RUNNING"
        assert not threading.Event().wait(0.03)
        model.device_done.set()
        worker.jobs.join()
        reply = worker.get_batch({"batch_id": "async"})
        assert reply["status"] == "SUCCEEDED"
        assert reply["execution_ms"] >= 30
        assert model.sync_calls == 1
        assert model.hidden_ref() is None
        assert "timing_breakdown" not in reply
    finally:
        model.device_done.set()
        worker.close()


def test_failed_and_aborted_tasks_do_not_get_success_breakdown():
    model = BatchModel()
    model.fail = True
    model.release.clear()
    worker = Worker(model)
    try:
        worker.submit_batch({"batch_id": "failed", "requests": requests((3,), "a")})
        assert model.entered.wait(5)
        worker.submit_batch({"batch_id": "aborted", "requests": requests((4,), "b")})
        model.release.set()
        worker.jobs.join()
        failed = worker.get_batch({"batch_id": "failed"})
        aborted = worker.get_batch({"batch_id": "aborted"})
        assert failed["status"] == "FAILED"
        assert failed["execution_ms"] >= 0
        assert failed["queue_wait_ms"] >= 0
        assert "timing_breakdown" not in failed
        assert aborted["status"] == "ABORTED"
        assert "execution_ms" not in aborted
        assert "queue_wait_ms" not in aborted
    finally:
        model.release.set()
        worker.close()


@pytest.mark.gpu
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_gpu_breakdown_covers_all_layers_and_resets_each_task(
    enabled, batch_size, caplog, monkeypatch
):
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel

    caplog.set_level(logging.DEBUG)
    model = TestModel(
        MemoryPool(16, 128), max_users=2, max_history=128, layers=4, dim=128
    )
    sync = model.synchronize
    sync_calls = []

    def traced_sync():
        sync_calls.append(True)
        sync()

    monkeypatch.setattr(model, "synchronize", traced_sync)
    worker = Worker(model, timing_breakdown=enabled)
    try:
        for step, lengths in enumerate(((3, 5), (7, 9), (7, 9))):
            sync_calls.clear()
            batch = {
                "batch_id": f"timed-{step}",
                "requests": requests(lengths, str(step))[:batch_size],
            }
            assert worker.submit_batch(batch)["accepted"]
            worker.jobs.join()
            reply = worker.get_batch({"batch_id": batch["batch_id"]})
            assert reply["status"] == "SUCCEEDED", reply
            assert reply["execution_ms"] == reply["elapsed_ms"] > 0
            assert reply["queue_wait_ms"] >= 0
            if enabled:
                assert_breakdown(reply, model.layers)
                assert len(sync_calls) == 7 * model.layers + 3
                for record in reply["requests"]:
                    assert record["timing_breakdown"] == reply["timing_breakdown"]
                    assert record["execution_ms"] == reply["execution_ms"]
            else:
                assert "timing_breakdown" not in reply
                assert len(sync_calls) == 1
            assert all(
                record["result"]["kv_length"] == lengths[i]
                for i, record in enumerate(reply["requests"])
            )
            # Responses remain RPC-safe and never include raw clock timestamps.
            assert "queued_ns" not in json.dumps(reply)
        assert ("batch_timing" in caplog.text) is enabled
    finally:
        worker.close()
