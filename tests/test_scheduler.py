import math
import threading
import time

import pytest
from batch_helpers import batch_methods

from cxl_recsys.common.request import placeholder
from cxl_recsys.common.rpc import serve
from cxl_recsys.scheduler import Scheduler
from cxl_recsys.scheduler import service as scheduler_service


class ImmediateWorker:
    def __init__(self):
        self.epoch = "incarnation-1"
        self.requests = []
        self.results = {}
        self.reject_once = True

    def status(self, _):
        return {"epoch": self.epoch}

    def submit(self, req):
        if self.reject_once:
            self.reject_once = False
            return {"accepted": False, "reason": "busy", "epoch": self.epoch}
        self.requests.append(req)
        self.results[req["request_id"]] = {
            "status": "SUCCEEDED",
            "epoch": self.epoch,
            "user_state": {
                "kv_length": len(req["history_token_ids"]),
                "backend": "dram",
            },
        }
        return {"accepted": True, "epoch": self.epoch}

    def get(self, req):
        return self.results[req["request_id"]]


def await_result(scheduler, rid):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        r = scheduler.get({"request_id": rid})
        if r["status"] in ("FAILED", "SUCCEEDED"):
            return r
        time.sleep(0.01)
    raise AssertionError("scheduler stalled")


def test_grpc_fifo_busy_dedup_and_epoch_guard():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        target = f"127.0.0.1:{sock.getsockname()[1]}"
    worker = ImmediateWorker()
    server = serve(target, batch_methods(worker.submit, worker.get, worker.status))
    scheduler = Scheduler([target])
    try:
        r1 = placeholder(request_id="a", history_length=3)
        r2 = placeholder(request_id="b", history_length=5)
        assert scheduler.enqueue(r1)["accepted"]
        assert scheduler.enqueue(r2)["accepted"]
        assert scheduler.enqueue(r1)["duplicate"]
        with pytest.raises(ValueError):
            scheduler.enqueue({**r1, "history_token_ids": [999]})
        assert await_result(scheduler, "b")["status"] == "SUCCEEDED"
        assert [r["request_id"] for r in worker.requests] == ["a", "b"]
        assert scheduler.status({})["users"]["user-0"]["kv_length"] == 5
        worker.epoch = "incarnation-2"
        scheduler.enqueue(placeholder(request_id="c", history_length=6))
        assert await_result(scheduler, "c")["status"] == "FAILED"
        assert len(worker.requests) == 2  # No dispatch against stale ownership.
        assert scheduler.status({})["failed"]
        assert scheduler.status({})["users"] == {}
        assert (
            scheduler.enqueue(placeholder(request_id="later"))["reason"] == "run_failed"
        )
    finally:
        scheduler.close()
        server.stop(0).wait()


@pytest.mark.parametrize("failure", ["worker", "rpc", "lost_reply", "worker_status"])
def test_run_failure_stops_dispatch_without_retry_and_invalidates_results(failure):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        target = f"127.0.0.1:{sock.getsockname()[1]}"
    worker = ImmediateWorker()
    worker.reject_once = False
    submit_calls = []
    get_calls = []

    def status(value):
        if failure == "worker_status" and worker.requests:
            return {"epoch": worker.epoch, "failed": True, "error": "injected failure"}
        return worker.status(value)

    def submit(request):
        submit_calls.append(request["request_id"])
        if request["request_id"] == "fail" and failure == "rpc":
            raise OSError("injected transport failure")
        reply = worker.submit(request)
        if request["request_id"] == "fail":
            if failure == "lost_reply":
                # Submit took effect, but its response was lost. Do not resubmit.
                raise OSError("injected response loss")
            worker.results["fail"] = {
                "epoch": worker.epoch,
                "status": "FAILED",
                "error": "injected inference failure",
            }
        return reply

    def get(request):
        get_calls.append(request["request_id"])
        return worker.get(request)

    server = serve(target, batch_methods(submit, get, status))
    scheduler = Scheduler([target])
    try:
        requests = [
            placeholder(request_id=rid, history_length=length)
            for rid, length in (("ok", 3), ("fail", 4), ("later", 5))
        ]
        report = scheduler.run_sequence(requests, timeout=5)
        assert report["state"] == "FAILED"
        assert not report["passed"] and not report["results_valid"]
        assert report["error"]
        assert [r["status"] for r in report["requests"]] == [
            "SUCCEEDED",
            "FAILED",
            "ABORTED",
        ]
        scheduler.thread.join(2)
        assert not scheduler.thread.is_alive()
        assert submit_calls == (
            ["ok"] if failure == "worker_status" else ["ok", "fail"]
        )
        assert get_calls.count("fail") == int(failure == "worker")
        assert scheduler.status({})["users"] == {}
        assert (
            scheduler.enqueue(placeholder(request_id="new"))["reason"] == "run_failed"
        )
    finally:
        scheduler.close()
        server.stop(0).wait()


class ReplayClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 0.02
        self.sleeps.append(seconds)
        # A real clock progresses even when a sub-ULP delay rounds to zero.
        self.now = math.nextafter(self.now + seconds, math.inf)


class ReplayScheduler(Scheduler):
    """Complete admissions synchronously to isolate the arrival-time policy."""

    def __init__(self, clock, blocked_until: float = 0):
        self.clock = clock
        self.started = clock.monotonic()
        self.blocked_until = blocked_until
        self.failure: str | None = None
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.records = {}
        self.admissions = []

    def enqueue(self, value):
        elapsed = self.clock.monotonic() - self.started
        if elapsed < self.blocked_until:
            return {"accepted": False, "reason": "busy"}
        request_id = value["request_id"]
        self.admissions.append((request_id, elapsed))
        self.records[request_id] = {
            "request_id": request_id,
            "status": "SUCCEEDED",
        }
        return {"accepted": True}

    def status(self, _):
        return {}


def replay_request(request_id, timestamp, visit_index):
    return {
        **placeholder(request_id=request_id),
        "timestamp": timestamp,
        "visit_index": visit_index,
    }


def test_sequence_replays_relative_arrivals_and_preserves_trace_metadata(monkeypatch):
    clock = ReplayClock()
    monkeypatch.setattr(scheduler_service, "time", clock)
    scheduler = ReplayScheduler(clock)
    timestamps = [0.0, 0.005, 0.005, 0.031]
    requests = [
        replay_request(f"visit-{index}", timestamp, index)
        for index, timestamp in enumerate(timestamps)
    ]

    report = scheduler.run_sequence(requests, timeout=1)

    assert report["passed"] and report["results_valid"]
    assert [rid for rid, _ in scheduler.admissions] == [
        request["request_id"] for request in requests
    ]
    assert [elapsed for _, elapsed in scheduler.admissions] == pytest.approx(timestamps)
    assert sum(clock.sleeps) == pytest.approx(timestamps[-1])
    assert [request["timestamp"] for request in report["requests"]] == timestamps
    assert [request["visit_index"] for request in report["requests"]] == [0, 1, 2, 3]


def test_sequence_without_timestamps_admits_immediately(monkeypatch):
    clock = ReplayClock()
    monkeypatch.setattr(scheduler_service, "time", clock)
    scheduler = ReplayScheduler(clock)
    requests = [placeholder(request_id=f"request-{index}") for index in range(3)]

    report = scheduler.run_sequence(requests, timeout=1)

    assert report["passed"]
    assert [elapsed for _, elapsed in scheduler.admissions] == [0, 0, 0]
    assert not clock.sleeps
    assert all("timestamp" not in request for request in report["requests"])


def test_sequence_replays_late_arrivals_in_order_after_backpressure(monkeypatch):
    clock = ReplayClock()
    monkeypatch.setattr(scheduler_service, "time", clock)
    scheduler = ReplayScheduler(clock, blocked_until=0.06)
    requests = [
        replay_request(f"visit-{index}", index * 0.01, index) for index in range(3)
    ]

    report = scheduler.run_sequence(requests, timeout=1)

    assert report["passed"]
    assert [rid for rid, _ in scheduler.admissions] == ["visit-0", "visit-1", "visit-2"]
    assert [elapsed for _, elapsed in scheduler.admissions] == pytest.approx([0.06] * 3)
    assert [request["timestamp"] for request in report["requests"]] == [0, 0.01, 0.02]


def test_sequence_timeout_includes_waiting_for_arrivals(monkeypatch):
    clock = ReplayClock()
    monkeypatch.setattr(scheduler_service, "time", clock)
    scheduler = ReplayScheduler(clock)
    requests = [replay_request("now", 0, 0), replay_request("later", 0.5, 1)]

    report = scheduler.run_sequence(requests, timeout=0.025)

    assert report["state"] == "TIMED_OUT"
    assert not report["passed"] and not report["results_valid"]
    assert report["submitted"] == 1
    assert report["elapsed_seconds"] == pytest.approx(0.025)
    assert report["requests"][1] == {
        "request_id": "later",
        "status": "NOT_SUBMITTED",
        "timestamp": 0.5,
        "visit_index": 1,
    }


@pytest.mark.parametrize("stop_reason", ["cancel", "failure"])
def test_sequence_stops_while_waiting_for_arrivals(monkeypatch, stop_reason):
    clock = ReplayClock()
    monkeypatch.setattr(scheduler_service, "time", clock)
    scheduler = ReplayScheduler(clock)
    stop_event = threading.Event()
    sleep = clock.sleep

    def stop_during_sleep(seconds):
        sleep(seconds)
        if stop_reason == "cancel":
            stop_event.set()
        else:
            scheduler.failure = "injected Worker failure"

    monkeypatch.setattr(clock, "sleep", stop_during_sleep)
    request = replay_request("later", 0.5, 0)

    report = scheduler.run_sequence([request], timeout=1, stop_event=stop_event)

    assert report["state"] == ("INTERRUPTED" if stop_reason == "cancel" else "FAILED")
    assert not report["passed"] and not report["results_valid"]
    assert report["submitted"] == 0
    assert report["elapsed_seconds"] == pytest.approx(0.02)
    assert report["requests"][0]["timestamp"] == 0.5
    assert scheduler.stop.is_set()


@pytest.mark.gpu
def test_real_worker_grpc_failure_discards_entire_sequence(monkeypatch):
    import socket

    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel
    from cxl_recsys.worker import Worker

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        target = f"127.0.0.1:{sock.getsockname()[1]}"
    pool = MemoryPool(8, 64)
    model = TestModel(pool, max_users=1, max_history=128)
    write = pool.write_many
    calls = []

    def fail_second_layer(slots, tensor):
        calls.append(len(slots))
        if len(calls) == 4:
            raise OSError("injected second-layer write failure")
        write(slots, tensor)

    monkeypatch.setattr(pool, "write_many", fail_second_layer)
    worker = Worker(model)
    server = serve(
        target,
        {
            "SubmitBatch": worker.submit_batch,
            "GetBatch": worker.get_batch,
            "Status": worker.status,
        },
    )
    scheduler = Scheduler([target])
    try:
        requests = [
            placeholder(request_id=rid, history_length=length)
            for rid, length in (("ok", 3), ("fail", 65), ("later", 66))
        ]
        report = scheduler.run_sequence(requests, timeout=10)
        assert report["state"] == "FAILED"
        assert not report["passed"] and not report["results_valid"]
        assert [r["status"] for r in report["requests"]] == [
            "SUCCEEDED",
            "FAILED",
            "ABORTED",
        ]
        assert calls == [3, 3, 62, 62]
        assert len(pool.allocated) == 4
        assert model.snapshot()["user-0"]["kv_length"] == 65
        assert worker.status({})["users"] == {}
        assert worker.get({"request_id": "ok"})["run_failed"]
        assert worker.submit(placeholder(request_id="new"))["reason"] == "run_failed"
    finally:
        scheduler.close()
        server.stop(0).wait()
        worker.close()
