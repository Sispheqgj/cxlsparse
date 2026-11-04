"""Round-robin first-user placement, sticky ownership, per-user FIFO.

Any Worker/RPC failure aborts the entire run without retry or failover.
State is in-memory and updated in place. Restart all participants for a new run.
"""

import logging
import threading
import time
import uuid
from collections import deque

from ..common.request import digest, validate
from ..common.rpc import RPC, ValidatedEnvelope

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        targets,
        max_pending=1024,
        max_records=10000,
        max_history=4096,
        batch_size=4,
    ):
        if not targets or max_pending <= 0 or max_records <= 0 or batch_size <= 0:
            raise ValueError("Scheduler needs Workers and positive limits")
        self.targets = list(targets)
        self.clients = [RPC(target) for target in targets]
        self.max_pending = max_pending
        self.max_records = max_records
        self.max_history = max_history
        self.batch_size = batch_size
        self.batches = {}
        self.records = {}
        self.owners = {}
        self.epochs = {}
        self.completion_features = {}
        self.completions = {}
        self.locations = {}
        self.pending = deque()
        self.active = {}
        self.round_robin = 0
        self.failure = None
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.wakeup = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="scheduler")
        self.thread.start()
        logger.info(
            "scheduler_initialized workers=%d max_pending=%d",
            len(self.targets),
            max_pending,
        )

    def enqueue(self, value):
        req = validate(value, self.max_history)
        rid, fp = req["request_id"], digest(req)
        with self.lock:
            if self.failure is not None:
                return {"accepted": False, "reason": "run_failed"}
            if rid in self.records:
                if self.records[rid]["digest"] != fp:
                    raise ValueError("request_id reused with a different payload")
                return {"accepted": True, "duplicate": True, "request_id": rid}
            if (
                self.stop.is_set()
                or len(self.pending) + len(self.active) >= self.max_pending
            ):
                return {"accepted": False, "reason": "busy"}
            if len(self.records) >= self.max_records:
                return {"accepted": False, "reason": "record_capacity"}
            uid = req["user_id"]
            if uid not in self.owners:
                self.owners[uid] = self.round_robin % len(self.clients)
                self.round_robin += 1
                logger.debug(
                    "user_placed user_id=%r worker=%s",
                    uid,
                    self.targets[self.owners[uid]],
                )
            self.records[rid] = {
                "status": "QUEUED",
                "request_id": rid,
                "request": req,
                "digest": fp,
                "worker": self.targets[self.owners[uid]],
            }
            self.pending.append(rid)
            self.wakeup.set()
            logger.debug(
                "request_queued request_id=%r user_id=%r pending=%d",
                rid,
                uid,
                len(self.pending),
            )
        return {"accepted": True, "duplicate": False, "request_id": rid}

    def get(self, value):
        wait_ms = value.get("wait_ms", 0)
        if type(wait_ms) not in (int, float) or not 0 <= wait_ms <= 1000:
            raise ValueError("wait_ms must be between 0 and 1000")
        with self.lock:
            rid = value["request_id"]
            if wait_ms and rid in self.records:
                self.changed.wait_for(
                    lambda: self.records[rid]["status"] in ("SUCCEEDED", "FAILED", "ABORTED")
                    or self.stop.is_set(), timeout=wait_ms / 1000,
                )
            r = self.records.get(rid)
            return (
                {k: v for k, v in r.items() if k not in ("request", "digest")}
                if r
                else {"status": "UNKNOWN"}
            )

    def status(self, _):
        with self.lock:
            return {
                "pending": len(self.pending),
                "active_users": len(self.active),
                "batch_size": self.batch_size,
                "batches": len(self.batches),
                "batch_sizes": [
                    len(task["request_ids"]) for task in self.batches.values()
                ],
                "workers": self.targets,
                "users": dict(self.locations),
                "records": len(self.records),
                "failed": self.failure is not None,
                "error": self.failure,
            }

    def _epoch(self, index, epoch):
        previous = self.epochs.setdefault(index, epoch)
        if previous != epoch:
            raise RuntimeError(
                "Worker restarted; KV ownership invalid. Coordinated restart/replay required"
            )

    def _fail_run(self, rid, error):
        """Stop admission/dispatch; retain completed records only for diagnosis."""
        with self.lock:
            if self.failure is not None:
                return
            self.failure = error
            self.records[rid].update(status="FAILED", error=error)
            failed_batch = self.records[rid].get("batch_id")
            for record in self.records.values():
                if record["status"] in ("QUEUED", "RUNNING"):
                    status = (
                        "FAILED"
                        if record.get("batch_id") == failed_batch
                        else "ABORTED"
                    )
                    record.update(status=status, error="Run failed: " + error)
            for task in self.batches.values():
                if task["status"] in ("QUEUED", "RUNNING"):
                    task["status"] = (
                        "FAILED" if task["batch_id"] == failed_batch else "ABORTED"
                    )
            self.pending.clear()
            self.active.clear()
            self.locations.clear()
            self.stop.set()
            self.changed.notify_all()
        logger.error(
            "scheduler_run_failed request_id=%r error=%r results_valid=False",
            rid,
            error,
        )

    def _loop(self):
        while not self.stop.is_set():
            self.wakeup.clear()
            with self.lock:
                self._form_batches()
                tasks = [
                    task
                    for task in self.batches.values()
                    if task["status"] in ("QUEUED", "RUNNING")
                ]
            for task in tasks:
                if self.stop.is_set():
                    break
                index = task["worker_index"]
                rid = task["request_ids"][0]
                try:
                    if task["status"] == "QUEUED":
                        if not self.completion_features.get(index, False):
                            health = self.clients[index].call("Status", timeout=1)
                            self._epoch(index, health["epoch"])
                            if health.get("failed"):
                                raise RuntimeError("Worker run failed: " + str(health["error"]))
                            features = health.get("control_features", {})
                            self.completion_features[index] = bool(
                                features.get("completion_wait") and features.get("epoch_guarded_submit")
                            )
                        reply = self.clients[index].call(
                            "SubmitBatch",
                            ValidatedEnvelope({
                                **({"expected_epoch": self.epochs[index]}
                                   if self.completion_features.get(index) else {}),
                                "batch_id": task["batch_id"],
                                "requests": [
                                    self.records[r]["request"]
                                    for r in task["request_ids"]
                                ],
                            }),
                            timeout=1,
                        )
                        self._epoch(index, reply["epoch"])
                        if not reply["accepted"]:
                            if reply["reason"] != "busy":
                                raise RuntimeError(
                                    "Worker rejected: " + reply["reason"]
                                )
                            continue
                        with self.lock:
                            task["status"] = "RUNNING"
                            for request_id in task["request_ids"]:
                                record = self.records[request_id]
                                record["status"] = "RUNNING"
                                logger.debug(
                                    "request_dispatched request_id=%r user_id=%r batch_id=%r worker=%s",
                                    request_id,
                                    record["request"]["user_id"],
                                    task["batch_id"],
                                    self.targets[index],
                                )
                        logger.debug(
                            "batch_dispatched batch_id=%r batch_size=%d worker=%s",
                            task["batch_id"],
                            len(task["request_ids"]),
                            self.targets[index],
                        )
                    if self.completion_features.get(index):
                        bid = task["batch_id"]
                        future = self.completions.get(bid)
                        if future is None:
                            future = self.clients[index].call_async(
                                "GetBatch", {"batch_id": bid, "wait_ms": 50}, timeout=1
                            )
                            self.completions[bid] = future
                            future.add_done_callback(lambda _: self.wakeup.set())
                        if not future.done():
                            continue
                        del self.completions[bid]
                        reply = future.result()
                        # A bounded long poll can expire during prefill. Re-arm
                        # on the next iteration; never block other Workers.
                        if reply["status"] in ("QUEUED", "RUNNING"):
                            self.wakeup.set()
                    else:
                        reply = self.clients[index].call(
                            "GetBatch", {"batch_id": task["batch_id"]}, timeout=1
                        )
                    self._epoch(index, reply["epoch"])
                    if reply["status"] == "UNKNOWN":
                        raise RuntimeError("Worker lost accepted batch")
                    if reply.get("run_failed") or reply["status"] in (
                        "FAILED",
                        "ABORTED",
                    ):
                        raise RuntimeError(reply.get("error", "Worker run failed"))
                    if reply["status"] == "SUCCEEDED":
                        results = reply["requests"]
                        if [r["request_id"] for r in results] != task[
                            "request_ids"
                        ] or any(r["status"] != "SUCCEEDED" for r in results):
                            raise RuntimeError(
                                "Worker returned inconsistent batch results"
                            )
                        with self.lock:
                            task["status"] = "SUCCEEDED"
                            for result in results:
                                record = self.records[result["request_id"]]
                                uid = record["request"]["user_id"]
                                record.update(result, epoch=reply["epoch"])
                                if result.get("user_state"):
                                    self.locations[uid] = {
                                        **result["user_state"],
                                        "worker": self.targets[index],
                                        "epoch": reply["epoch"],
                                    }
                                del self.active[uid]
                                logger.info(
                                    "request_finished request_id=%r user_id=%r batch_id=%r status=SUCCEEDED",
                                    result["request_id"],
                                    uid,
                                    task["batch_id"],
                                )
                            self.changed.notify_all()
                            self.wakeup.set()
                        logger.info(
                            "batch_finished batch_id=%r batch_size=%d",
                            task["batch_id"],
                            len(results),
                        )
                except Exception as exc:
                    logger.exception(
                        "batch_failed batch_id=%r worker=%s",
                        task["batch_id"],
                        self.targets[index],
                    )
                    self._fail_run(rid, f"{type(exc).__name__}: {exc}")
            # Admissions and completed RPC futures wake dispatch immediately.
            # Legacy Workers and busy admission retain bounded polling.
            with self.lock:
                busy = bool(self.pending) or any(
                    task["status"] == "QUEUED" or (
                        task["status"] == "RUNNING"
                        and not self.completion_features.get(task["worker_index"])
                    ) for task in self.batches.values()
                )
            if self.stop.is_set():
                break
            self.wakeup.wait(0.002 if busy else None)

    def _form_batches(self):
        """Reserve each user's oldest request and group by sticky Worker owner.

        Called under the scheduler lock. A rejected-busy batch keeps its identity
        and user reservations. Partial batches are dispatched without waiting for
        more users, including tails and workloads smaller than batch_size.
        """
        groups = {}
        waiting_workers = {
            task["worker_index"]
            for task in self.batches.values()
            if task["status"] == "QUEUED"
        }
        for _ in range(len(self.pending)):
            rid = self.pending.popleft()
            uid = self.records[rid]["request"]["user_id"]
            index = self.owners[uid]
            group = groups.setdefault(index, [])
            if (
                uid in self.active
                or index in waiting_workers
                or len(group) >= self.batch_size
            ):
                self.pending.append(rid)
                continue
            self.active[uid] = rid
            group.append(rid)
        for index, request_ids in groups.items():
            if not request_ids:
                continue
            batch_id = "batch-" + uuid.uuid4().hex
            self.batches[batch_id] = {
                "batch_id": batch_id,
                "request_ids": request_ids,
                "worker_index": index,
                "status": "QUEUED",
            }
            for position, rid in enumerate(request_ids):
                self.records[rid].update(batch_id=batch_id, batch_position=position)
            logger.debug(
                "batch_formed batch_id=%r batch_size=%d request_ids=%s worker=%s",
                batch_id,
                len(request_ids),
                request_ids,
                self.targets[index],
            )

    def close(self):
        logger.info("scheduler_stopping")
        with self.changed:
            self.stop.set()
            self.changed.notify_all()
        self.wakeup.set()
        self.thread.join()
        for future in self.completions.values():
            future.cancel()
        for c in self.clients:
            c.close()

    def run_sequence(self, requests, timeout: float = 300, stop_event=None):
        """Own admission and completion of a finite, prevalidated workload.

        Failure/timeout never reroutes work. Already running remote work may finish,
        but its results are invalid. Restart Workers before another run.
        Optional timestamps are simulated seconds relative to this run's start;
        admission waits until each arrival and preserves order under backpressure.
        No Scheduler RPC Submit endpoint exists.
        """
        if timeout <= 0:
            raise ValueError("run timeout must be positive")
        started, cursor = time.monotonic(), 0
        logger.info("sequence_started requests=%d timeout=%s", len(requests), timeout)
        while True:
            if self.failure is not None:
                state = "FAILED"
                break
            if stop_event is not None and stop_event.is_set():
                state = "INTERRUPTED"
                break
            if time.monotonic() - started >= timeout:
                state = "TIMED_OUT"
                break
            # Admit a window atomically so the dispatcher sees available peers,
            # rather than racing the first enqueue and forming tiny batches.
            with self.lock:
                while cursor < len(requests):
                    elapsed = time.monotonic() - started
                    if (
                        elapsed >= timeout
                        or (stop_event is not None and stop_event.is_set())
                        or requests[cursor].get("timestamp", 0) > elapsed
                    ):
                        break
                    reply = self.enqueue(requests[cursor])
                    if not reply["accepted"]:
                        if reply["reason"] not in ("busy", "run_failed"):
                            raise RuntimeError("Admission rejected: " + reply["reason"])
                        break
                    cursor += 1
            with self.lock:
                completed = sum(
                    r["status"] in ("SUCCEEDED", "FAILED")
                    for r in self.records.values()
                )
            if (
                self.failure is None
                and cursor == len(requests)
                and completed == len(requests)
            ):
                state = "COMPLETED"
                break
            # Wake at the next arrival or timeout; a busy Worker is polled at
            # most every 20 ms, with the same bound for cancellation checks.
            elapsed = time.monotonic() - started
            delay = min(0.02, max(0, timeout - elapsed))
            if cursor < len(requests):
                until_arrival = requests[cursor].get("timestamp", 0) - elapsed
                if until_arrival > 0:
                    delay = min(delay, until_arrival)
            if delay > 0:
                time.sleep(delay)
        self.stop.set()
        results = []
        for index, request in enumerate(requests):
            request_id = request["request_id"]
            if index < cursor:
                result = self.get({"request_id": request_id})
            else:
                result = {"request_id": request_id, "status": "NOT_SUBMITTED"}
            for field in ("timestamp", "visit_index"):
                if field in request:
                    result[field] = request[field]
            results.append(result)
        success = sum(r["status"] == "SUCCEEDED" for r in results)
        logger.info(
            "sequence_finished state=%s submitted=%d succeeded=%d total=%d",
            state,
            cursor,
            success,
            len(requests),
        )
        return {
            "state": state,
            "passed": state == "COMPLETED" and success == len(requests),
            "results_valid": state == "COMPLETED" and success == len(requests),
            "error": self.failure,
            "total": len(requests),
            "submitted": cursor,
            "succeeded": success,
            "elapsed_seconds": time.monotonic() - started,
            "requests": results,
            "scheduler": self.status({}),
        }
