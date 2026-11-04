"""Process lifecycle and fixed GR inputs for the final comparison."""

import json
import socket
import subprocess
import time
from pathlib import Path

import grpc
from tokenizers import Tokenizer

from cxl_recsys.common.request import validate
from cxl_recsys.common.rpc import RPC


def free_address():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{sock.getsockname()[1]}"


def ready(process, address, timeout=180):
    client = RPC(address)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Process exited before ready: {address}")
            try:
                return client.call("Status", timeout=1)
            except grpc.RpcError:
                time.sleep(0.2)
        raise TimeoutError(address)
    finally:
        client.close()


def stop(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # Preserve CXL restoration: do not SIGKILL a worker holding devdax.
        raise RuntimeError(
            f"Process {process.pid} did not shut down; inspect before terminating"
        )


def run_request(scheduler, request, timeout=300):
    start = time.perf_counter_ns()
    if not scheduler.enqueue(request)["accepted"]:
        raise RuntimeError("Scheduler rejected request")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reply = scheduler.get({
            "request_id": request["request_id"],
            "wait_ms": min(50, max(0, (deadline - time.monotonic()) * 1000)),
        })
        if reply["status"] == "SUCCEEDED":
            return (time.perf_counter_ns() - start) / 1e6, reply
        if reply["status"] in ("FAILED", "ABORTED"):
            raise RuntimeError(reply)
        time.sleep(0.001)
    raise TimeoutError(request["request_id"])


def inputs(args, config, history_length=None):
    source = json.loads(args.requests.read_text())[-1]
    history = source["history_token_ids"]
    if history_length is not None:
        history = (history * ((history_length + len(history) - 1) // len(history)))[
            :history_length
        ]
    tokenizer = Tokenizer.from_file(
        str(Path(config["worker"]["model_path"]) / "tokenizer.json")
    )
    # Same template and candidate IDs at every size. Deterministic filler varies
    # only the number of tokens. This is a latency workload, not a quality eval.
    template = tokenizer.encode(
        "Candidates:\n(A) lamp\n(B) bottle\n(C) bag\nAnswer:\n",
        add_special_tokens=False,
    ).ids
    filler = tokenizer.encode(
        " product quality durable everyday", add_special_tokens=False
    ).ids
    requests = {}
    for length in args.lengths:
        if length < len(template):
            raise ValueError("Extend too short for candidate template")
        padding = length - len(template)
        suffix = (filler * ((padding + len(filler) - 1) // len(filler)))[
            :padding
        ] + template
        assert len(suffix) == length
        assert tokenizer.decode(suffix, skip_special_tokens=False).endswith(
            "\nAnswer:\n"
        )
        request = dict(
            request_id=f"length-{length}",
            user_id="sweep-user",
            history_token_ids=history,
            candidate_token_ids=suffix,
            candidate_item_ids=source["candidate_item_ids"][:3],
        )
        requests[length] = validate(request, config["worker"]["max_history"])
    return requests
