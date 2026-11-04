"""Bounded batch-task queue with one GPU executor and per-request diagnostics."""

import json
import logging
import queue
import threading
import time
import uuid
from typing import Any

from ..common.batch import validate_batch
from ..common.logging import request_context
from ..common.request import digest, validate
from ..common.timing import StageTimer, synchronize_model
from .inference import execute_batch

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        model,
        queue_capacity=1,
        max_records=10000,
        max_batch_size=16,
        timing_breakdown=False,
    ):
        if queue_capacity < 0 or max_records <= 0 or max_batch_size <= 0:
            raise ValueError("Invalid Worker capacities")
        self.model = model
        self.capacity = queue_capacity + 1
        self.max_records = max_records
        self.max_batch_size = max_batch_size
        self.timing_breakdown = timing_breakdown
        self.epoch = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.jobs = queue.Queue()
        self.records = {}
        self.batches = {}
        self.outstanding = 0
        self.users = {}
        self.closing = False
        self.failure = None
        self.thread = threading.Thread(target=self._execute, name="gpu-executor")
        self.thread.start()
        logger.info(
            "worker_initialized capacity=%d max_batch_size=%d epoch=%s timing_breakdown=%s",
            self.capacity,
            max_batch_size,
            self.epoch,
            self.timing_breakdown,
        )

    def submit(self, value):
        """Compatibility endpoint: a single request becomes a singleton task."""
        request = validate(value, self.model.max_history)
        rid = request["request_id"]
        with self.lock:
            if self.failure is not None:
                return self._rejected("run_failed")
            if rid in self.records:
                if self.records[rid]["digest"] != digest(request):
                    raise ValueError("request_id reused with a different payload")
                return {
                    "accepted": True,
                    "duplicate": True,
                    "epoch": self.epoch,
                    "request_id": rid,
                    "batch_id": self.records[rid]["batch_id"],
                }
            reply = self._accept(
                {
                    "batch_id": "single-" + uuid.uuid4().hex,
                    "requests": [request],
                }
            )
            if reply["accepted"]:
                reply["request_id"] = rid
            return reply

    def submit_batch(self, value):
        if isinstance(value, dict) and "expected_epoch" in value:
            expected = value["expected_epoch"]
            if not isinstance(expected, str) or not 0 < len(expected) <= 128:
                raise ValueError("expected_epoch must be a nonempty string")
            # Reject stale ownership before validating/admitting or touching KV.
            if expected != self.epoch:
                return self._rejected("epoch_mismatch")
            value = {k: v for k, v in value.items() if k != "expected_epoch"}
        batch = validate_batch(value, self.model.max_history, self.max_batch_size)
        with self.lock:
            return self._accept(batch)

    def _rejected(self, reason):
        return {"accepted": False, "reason": reason, "epoch": self.epoch}

    def _accept(self, batch):
        """Accept all members together; capacity is measured in batch tasks."""
        bid = batch["batch_id"]
        requests = batch["requests"]
        if self.failure is not None:
            return self._rejected("run_failed")
        if bid in self.batches:
            if self.batches[bid]["payload"] != batch:
                raise ValueError("batch_id reused with a different payload")
            return {
                "accepted": True,
                "duplicate": True,
                "batch_id": bid,
                "epoch": self.epoch,
            }
        if any(request["request_id"] in self.records for request in requests):
            raise ValueError("request_id already belongs to an accepted batch")
        if self.closing or self.outstanding >= self.capacity:
            return self._rejected("busy")
        if len(self.records) + len(requests) > self.max_records:
            return self._rejected("record_capacity")
        self.batches[bid] = {
            "payload": batch,
            "status": "QUEUED",
            "batch_id": bid,
            "request_ids": [request["request_id"] for request in requests],
        }
        for position, request in enumerate(requests):
            self.records[request["request_id"]] = {
                "digest": digest(request),
                "status": "QUEUED",
                "request_id": request["request_id"],
                "batch_id": bid,
                "batch_position": position,
            }
        self.outstanding += 1
        # Record admission immediately before enqueue; duplicate submissions never
        # replace this timestamp. It is process-local and must not go over RPC.
        self.batches[bid]["queued_ns"] = time.perf_counter_ns()
        self.jobs.put(batch)
        logger.debug(
            "batch_queued batch_id=%r batch_size=%d outstanding=%d",
            bid,
            len(requests),
            self.outstanding,
        )
        return {
            "accepted": True,
            "duplicate": False,
            "batch_id": bid,
            "epoch": self.epoch,
        }

    def _execute(self):
        while True:
            batch = self.jobs.get()
            if batch is None:
                self.jobs.task_done()
                return
            bid, requests = batch["batch_id"], batch["requests"]
            with self.lock:
                started_ns = time.perf_counter_ns()
                queue_wait_ms = (
                    started_ns - self.batches[bid]["queued_ns"]
                ) / 1_000_000
                self._mark_batch(
                    bid, "RUNNING", queue_wait_ms=queue_wait_ms, timing_scope="batch"
                )
            timing = (
                StageTimer(lambda: synchronize_model(self.model), started_ns=started_ns)
                if self.timing_breakdown
                else None
            )
            logger.debug(
                "batch_running batch_id=%r batch_size=%d queue_wait_ms=%.3f",
                bid,
                len(requests),
                queue_wait_ms,
            )
            try:
                if len(requests) == 1:
                    with request_context(
                        requests[0]["request_id"], requests[0]["user_id"]
                    ):
                        results = execute_batch(self.model, requests, timing=timing)
                else:
                    results = execute_batch(self.model, requests, timing=timing)
                snapshot = self.model.snapshot()
                if timing is not None:
                    # Candidate selection returns CPU results only after its GPU
                    # scoring/argmax and transfer finish; include that work here.
                    timing.mark("postprocess_ms", synchronize=False)
                    finished_ns = timing.last_ns
                else:
                    finished_ns = time.perf_counter_ns()
                elapsed_ms = (finished_ns - started_ns) / 1_000_000
            except Exception as exc:
                logger.exception("batch_failed batch_id=%r", bid)
                error = f"{type(exc).__name__}: {exc}"
                with self.lock:
                    self.failure = error
                    self.closing = True
                    self.users = {}
                    failed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                    self._mark_batch(
                        bid,
                        "FAILED",
                        error=error,
                        elapsed_ms=failed_ms,
                        execution_ms=failed_ms,
                    )
                    # Partial KV state is invalid. Never execute or retry another
                    # batch after a failed member; there is no rollback.
                    while True:
                        try:
                            queued = self.jobs.get_nowait()
                        except queue.Empty:
                            break
                        if queued is not None:
                            self._mark_batch(
                                queued["batch_id"],
                                "ABORTED",
                                error="Worker run failed: " + error,
                            )
                        self.jobs.task_done()
                    self.outstanding = 0
                self.jobs.task_done()
                logger.error("worker_run_failed error=%r results_valid=False", error)
                return
            with self.lock:
                fields: dict[str, Any] = {
                    "elapsed_ms": elapsed_ms,
                    "execution_ms": elapsed_ms,
                }
                if timing is not None:
                    fields["timing_breakdown"] = timing.report()
                self._mark_batch(bid, "SUCCEEDED", **fields)
                for request, result in zip(requests, results):
                    rid, uid = request["request_id"], request["user_id"]
                    self.records[rid].update(
                        result=result,
                        user_state=snapshot.get(uid),
                    )
                    logger.info(
                        "request_finished request_id=%r user_id=%r batch_id=%r status=SUCCEEDED queue_wait_ms=%.3f elapsed_ms=%.3f",
                        rid,
                        uid,
                        bid,
                        queue_wait_ms,
                        elapsed_ms,
                    )
                self.users = snapshot
                self.outstanding -= 1
            logger.info(
                "batch_finished batch_id=%r batch_size=%d queue_wait_ms=%.3f execution_ms=%.3f elapsed_ms=%.3f",
                bid,
                len(requests),
                queue_wait_ms,
                elapsed_ms,
                elapsed_ms,
            )
            if timing is not None:
                logger.debug(
                    "batch_timing batch_id=%r scope=batch breakdown=%s",
                    bid,
                    json.dumps(fields["timing_breakdown"]),
                )
            self.jobs.task_done()

    def _mark_batch(self, bid, status, **fields):
        self.batches[bid].update(status=status, **fields)
        for rid in self.batches[bid]["request_ids"]:
            self.records[rid].update(status=status, **fields)

        self.changed.notify_all()

    def _request_reply(self, rid):
        result = self.records.get(rid)
        if result is None:
            return {"epoch": self.epoch, "status": "UNKNOWN", "error": self.failure}
        return {
            "epoch": self.epoch,
            "run_failed": self.failure is not None,
            **{key: value for key, value in result.items() if key != "digest"},
        }

    def get(self, value) -> dict[str, Any]:
        with self.lock:
            return self._request_reply(value["request_id"])

    def get_batch(self, value):
        wait_ms = value.get("wait_ms", 0)
        if type(wait_ms) not in (int, float) or not 0 <= wait_ms <= 1000:
            raise ValueError("wait_ms must be between 0 and 1000")
        with self.changed:
            bid = value["batch_id"]
            if wait_ms and bid in self.batches:
                self.changed.wait_for(
                    lambda: self.batches[bid]["status"] in ("SUCCEEDED", "FAILED", "ABORTED")
                    or self.failure is not None or self.closing,
                    timeout=wait_ms / 1000,
                )
            batch = self.batches.get(bid)
            if batch is None:
                return {"epoch": self.epoch, "status": "UNKNOWN", "error": self.failure}
            return {
                "epoch": self.epoch,
                "run_failed": self.failure is not None,
                **{
                    key: item
                    for key, item in batch.items()
                    if key not in ("payload", "queued_ns")
                },
                "requests": [self._request_reply(rid) for rid in batch["request_ids"]],
            }

    def status(self, _):
        with self.lock:
            return {
                "control_features": {"completion_wait": True, "epoch_guarded_submit": True},
                "epoch": self.epoch,
                "outstanding": self.outstanding,
                "capacity": self.capacity,
                "max_batch_size": self.max_batch_size,
                "timing_breakdown": self.timing_breakdown,
                "records": len(self.records),
                "batches": len(self.batches),
                "users": dict(self.users),
                "closing": self.closing,
                "failed": self.failure is not None,
                "error": self.failure,
            }

    def close(self):
        logger.info("worker_draining")
        with self.lock:
            self.closing = True
            self.changed.notify_all()
            if self.failure is None and self.thread.is_alive():
                self.jobs.put(None)
        self.thread.join()
        self.model.close()
