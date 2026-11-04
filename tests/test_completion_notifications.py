import threading

import pytest
from test_batching import BatchModel, address

from cxl_recsys.common.request import placeholder
from cxl_recsys.common.rpc import serve
from cxl_recsys.scheduler import Scheduler
from cxl_recsys.worker import Worker


def start(model):
    worker = Worker(model)
    target = address()
    calls = []

    def status(value):
        calls.append(value)
        return worker.status(value)

    server = serve(
        target,
        {
            "Status": status,
            "SubmitBatch": worker.submit_batch,
            "GetBatch": worker.get_batch,
        },
    )
    return worker, server, target, calls


def test_stale_epoch_rejected_before_admission():
    model = BatchModel()
    worker = Worker(model)
    try:
        reply = worker.submit_batch(
            dict(batch_id="batch", requests=[placeholder()], expected_epoch="stale")
        )
        assert not reply["accepted"] and reply["reason"] == "epoch_mismatch"
        assert not worker.records and not worker.batches and not model.calls
    finally:
        worker.close()


def test_completed_notification_contains_full_results():
    model = BatchModel()
    model.release.clear()
    worker = Worker(model)
    result = []
    try:
        worker.submit_batch(dict(batch_id="batch", requests=[placeholder()]))
        assert model.entered.wait(2)
        t = threading.Thread(
            target=lambda: result.append(
                worker.get_batch(dict(batch_id="batch", wait_ms=1000))
            )
        )
        t.start()
        model.release.set()
        t.join(2)
        assert not t.is_alive() and result[0]["status"] == "SUCCEEDED"
        assert "result" in result[0]["requests"][0]
    finally:
        model.release.set()
        worker.close()


def test_slow_worker_does_not_block_other_worker_completion():
    slow = BatchModel()
    slow.release.clear()
    fast = BatchModel()
    entries = [start(slow), start(fast)]
    s = Scheduler([x[2] for x in entries], batch_size=1)
    try:
        s.enqueue(placeholder(request_id="slow", user_id="u0"))
        assert slow.entered.wait(2)
        s.enqueue(placeholder(request_id="fast", user_id="u1"))
        assert s.get(dict(request_id="fast", wait_ms=1000))["status"] == "SUCCEEDED"
        assert (
            not slow.release.is_set()
            and s.get(dict(request_id="slow"))["status"] == "RUNNING"
        )
        slow.release.set()
        assert s.get(dict(request_id="slow", wait_ms=1000))["status"] == "SUCCEEDED"
    finally:
        slow.release.set()
        s.close()
        for worker, server, _, _ in entries:
            server.stop(0).wait()
            worker.close()


def test_epoch_guard_avoids_repeated_health_rpc_and_stale_dispatch():
    model = BatchModel()
    w, server, target, calls = start(model)
    s = Scheduler([target])
    try:
        s.enqueue(placeholder(request_id="first"))
        assert s.get(dict(request_id="first", wait_ms=1000))["status"] == "SUCCEEDED"
        s.enqueue(placeholder(request_id="second"))
        assert s.get(dict(request_id="second", wait_ms=1000))["status"] == "SUCCEEDED"
        assert len(calls) == 1 and len(model.calls) == 2
        w.epoch = "new-incarnation"
        s.enqueue(placeholder(request_id="third"))
        assert s.get(dict(request_id="third", wait_ms=1000))["status"] == "FAILED"
        assert len(calls) == 1 and len(model.calls) == 2 and "third" not in w.records
    finally:
        s.close()
        server.stop(0).wait()
        w.close()


def test_failure_wakes_waiting_client_and_stops_run():
    model = BatchModel()
    model.fail = True
    w, server, target, _ = start(model)
    s = Scheduler([target])
    try:
        s.enqueue(placeholder())
        assert s.get(dict(request_id="request-0", wait_ms=1000))["status"] == "FAILED"
        assert s.status({})["failed"]
        assert not s.enqueue(placeholder(request_id="later"))["accepted"]
    finally:
        s.close()
        server.stop(0).wait()
        w.close()


@pytest.mark.parametrize("wait", [True, -1, 1001, float("nan"), None])
def test_invalid_wait_rejected(wait):
    w = Worker(BatchModel())
    s = Scheduler(["127.0.0.1:1"])
    try:
        with pytest.raises(ValueError):
            w.get_batch(dict(batch_id="missing", wait_ms=wait))
        with pytest.raises(ValueError):
            s.get(dict(request_id="missing", wait_ms=wait))
    finally:
        s.close()
        w.close()


def test_bounded_wait_can_rearm_until_completion():
    model = BatchModel()
    model.release.clear()
    worker = Worker(model)
    try:
        worker.submit_batch(dict(batch_id="batch", requests=[placeholder()]))
        assert model.entered.wait(2)
        assert (
            worker.get_batch(dict(batch_id="batch", wait_ms=1))["status"] == "RUNNING"
        )
        model.release.set()
        assert (
            worker.get_batch(dict(batch_id="batch", wait_ms=1000))["status"]
            == "SUCCEEDED"
        )
    finally:
        model.release.set()
        worker.close()


def test_scheduler_close_releases_waiter_and_cancels_rpc():
    model = BatchModel()
    model.release.clear()
    worker, server, target, _ = start(model)
    s = Scheduler([target])
    result = []
    try:
        s.enqueue(placeholder())
        assert model.entered.wait(2)
        waiter = threading.Thread(
            target=lambda: result.append(
                s.get(dict(request_id="request-0", wait_ms=1000))
            )
        )
        waiter.start()
        s.close()
        waiter.join(2)
        assert not waiter.is_alive() and not s.thread.is_alive()
    finally:
        model.release.set()
        server.stop(0).wait()
        worker.close()
