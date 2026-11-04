"""Unprofiled matched GR benchmark, including cold buckets and mixed suffixes.

Uses the actual Scheduler/gRPC/Worker and owned physical CXL or GPUDirect RDMA
loopback. The frozen working tree, not Git HEAD, supplies both original controls.
Results are never inferred from a DRAM proxy or a silently falling-back GR path.
"""

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from cases import CASES

from cxl_recsys.common.rpc import RPC
from cxl_recsys.scheduler.service import Scheduler
from scripts.benchmark_support import free_address, inputs, ready, run_request, stop

VARIANTS = {}
TMA_READ_ORDERS = {}
READY_WINDOWS = {}


def require_idle_gpu():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if text:
        raise RuntimeError(
            f"GPU has active compute processes; leave them undisturbed: {text}"
        )


def captures(execution):
    result = {
        name: values["captures"]
        for name, values in execution.items()
        if name.endswith("graphs") and isinstance(values, dict)
    }
    transport = execution.get("selected_transport") or {}
    attention = transport.get("attention_graphs")
    if attention:
        result["selected_attention_graphs"] = attention["captures"]
    return result


def stop_owned(process):
    """Reap only a Popen created here; recovery runs after the GPU process exits."""
    try:
        stop(process)
        return False
    except RuntimeError:
        if process is None or process.poll() is not None:
            raise
        process.kill()
        process.wait(timeout=30)
        return True


def case(args, base, history, round_index, backend, variant):
    require_idle_gpu()
    out = args.output / f"round-{round_index}" / backend / variant / f"h-{history}"
    out.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(base)
    w = config["worker"]
    selected, group = VARIANTS[variant]
    w.update(
        backend=backend,
        model="nsa_1b",
        max_users=1,
        max_batch_size=1,
        max_history=history,
        pages=32 * ((history + w["page_size"] - 1) // w["page_size"]),
        listen=free_address(),
        nsa_kernel_backend="fla",
        nsa_cuda_graphs=True,
        nsa_graph_cache_size=5,
        nsa_cache_backend="auto",
        nsa_selected_backend=selected,
        nsa_selected_max_queries=1024
        if selected in ("triton_gr", "tilelang_gr", "tilelang_tma")
        else 128,
        timing_breakdown=False,
        log_level="INFO",
        log_dir=str(out / "worker-logs"),
    )
    if variant in TMA_READ_ORDERS:
        w["nsa_selected_read_order"] = TMA_READ_ORDERS[variant]
    if READY_WINDOWS.get(variant, 1) != 1:
        w["nsa_selected_ready_window"] = READY_WINDOWS[variant]
    else:
        # Frozen controls predate this setting; the unchanged path defaults to 1.
        w.pop("nsa_selected_ready_window", None)
    env = dict(os.environ)
    if env.get("NSA_TMA_ORDER_PROBE"):
        env["NSA_TMA_ORDER_PROBE"] = str(out / "order-probe")
    if env.get("NSA_TMA_WINDOW_PROBE"):
        env["NSA_TMA_WINDOW_PROBE"] = str(out / "window-probe")
    if getattr(args, "save_outputs", False):
        env["NSA_GR_CAPTURE_OUTPUTS"] = str(out / "outputs")
    else:
        env.pop("NSA_GR_CAPTURE_OUTPUTS", None)
    if variant in ("origin", "gpu_baseline"):
        env["NSA_RUNTIME_SNAPSHOT"] = str(args.baseline_snapshot)
        w.pop("nsa_selected_query_group", None)
        w.pop("nsa_selected_workspace_mib", None)
    else:
        runtime = getattr(args, "runtime_snapshot", None)
        if runtime:
            env["NSA_RUNTIME_SNAPSHOT"] = str(runtime)
        else:
            env.pop("NSA_RUNTIME_SNAPSHOT", None)
        w.update(nsa_selected_query_group=group, nsa_selected_workspace_mib=512)
    requests = inputs(args, config, history)
    (out / "inputs.json").write_text(json.dumps(requests))
    rows, pool, worker, scheduler, pool_state = [], None, None, None, None
    completed = False
    with (
        (out / "worker.log").open("x") as worker_log,
        (out / "pool.log").open("x") as pool_log,
    ):
        try:
            if backend == "rdma":
                w["pool_target"] = free_address()
                pool = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "cxl_recsys.cli",
                        "pool",
                        "--listen",
                        w["pool_target"],
                        "--max-bytes",
                        str(w["pages"] * w["page_size"] * 256),
                        "--rdma-device",
                        "mlx5_1",
                        "--rdma-port",
                        "1",
                        "--gid-index",
                        "1",
                        "--log-dir",
                        str(out / "pool-logs"),
                    ],
                    stdout=pool_log,
                    stderr=subprocess.STDOUT,
                )
                ready(pool, w["pool_target"])
            config_path = out / "config.json"
            config_path.write_text(json.dumps(config, indent=2))
            worker = subprocess.Popen(
                [
                    sys.executable,
                    str(
                        getattr(
                            args,
                            "worker_script",
                            Path(__file__).with_name("nsa_gr_worker.py"),
                        )
                    ),
                    str(out),
                    "worker",
                    "--config",
                    str(config_path),
                ],
                env=env,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
            )
            ready(worker, w["listen"], timeout=300)
            scheduler = Scheduler([w["listen"]], max_history=history, batch_size=1)
            history_tokens = next(iter(requests.values()))["history_token_ids"]
            sequence = 0
            previous_execution = None

            def submit(request, stage):
                nonlocal sequence, previous_execution
                request = copy.deepcopy(request)
                request["request_id"] = f"gr-{sequence}"
                sequence += 1
                elapsed, reply = run_request(scheduler, request, timeout=1200)
                result, execution = reply["result"], reply["result"]["nsa_execution"]
                assert result["pool"]["backend"] == backend
                assert result["recommended_item_id"] in request["candidate_item_ids"]
                q = len(request["candidate_token_ids"])
                if stage != "prefill":
                    assert result["prompt"]["input_tokens"] == q
                    assert result["pool"]["write_bytes"] == history * 8192
                    treatment = CASES[variant]
                    transport = execution["selected_transport"]
                    assert (
                        execution["selected_cross_query_dedup"] is treatment["shared"]
                    )
                    assert transport["cross_query_dedup"] is treatment["shared"]
                    if not treatment["shared"]:
                        assert transport["planner"] == "query_private_gpu"
                    if selected == "triton_gr":
                        assert transport["consumer"] == "native_indexed_bounded"
                    if backend == "rdma":
                        assert (
                            transport["rdma_registration"] == treatment["registration"]
                        )
                    if selected in ("triton_gr", "tilelang_gr", "tilelang_tma"):
                        prior = previous_execution or {}
                        assert execution["selected_native_fallbacks"] == prior.get(
                            "selected_native_fallbacks", 0
                        )
                        assert (
                            execution["selected_gr_calls"]
                            - prior.get("selected_gr_calls", 0)
                            == 16
                        )
                        assert execution["selected_transport"]["effective_q"] == q
                        if selected == "tilelang_tma":
                            assert (
                                execution["selected_tma_calls"]
                                - prior.get("selected_tma_calls", 0)
                                == 16
                            )
                            assert (
                                execution["selected_transport"]["consumer"]
                                == "tma_bounded_rows"
                            )
                            assert (
                                execution["selected_transport"]["read_order"]
                                == TMA_READ_ORDERS[variant]
                            )
                            assert (
                                execution["selected_transport"].get("ready_window", 1)
                                == READY_WINDOWS[variant]
                            )
                if backend == "rdma":
                    assert result["pool"]["transport"] == "gpudirect_dmabuf"
                    assert result["pool"]["cpu_staging_bytes"] == 0
                row = dict(
                    round=round_index,
                    backend=backend,
                    variant=variant,
                    stage=stage,
                    history=history,
                    extend=q,
                    e2e_ms=elapsed,
                    worker_ms=reply["execution_ms"],
                    captures=captures(execution),
                    response=reply,
                )
                rows.append(row)
                previous_execution = execution
                (out / "results.json").write_text(json.dumps(rows, indent=2))
                return row

            for end in range(
                args.prefill_chunk, history + args.prefill_chunk, args.prefill_chunk
            ):
                request = copy.deepcopy(requests[min(requests)])
                request["history_token_ids"] = history_tokens[: min(end, history)]
                submit(request, "prefill")
            ordered = sorted(requests, reverse=bool(round_index % 2))
            for q in ordered:
                submit(requests[q], "new_length")
                for _ in range(args.warmups):
                    warm = submit(requests[q], "warmup")
                for _ in range(args.repeats):
                    measured = submit(requests[q], "steady")
                    assert measured["captures"] == warm["captures"], (
                        "Unexpected steady capture"
                    )
            mixed = (
                []
                if getattr(args, "skip_mixed", False)
                else list(requests) * args.repeats
            )
            random.Random(args.seed + round_index).shuffle(mixed)
            warm = rows[-1]["captures"]
            for q in mixed:
                row = submit(requests[q], "mixed")
                if selected in ("triton_gr", "tilelang_gr", "tilelang_tma"):
                    assert row["captures"] == warm, (
                        "Bucket warmed, but mixed Q recaptured"
                    )
            completed = True
        finally:
            cleanup_errors, forced = [], []

            def cleanup(name, action):
                try:
                    return action()
                except Exception as error:
                    cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")

            if scheduler:
                cleanup("scheduler", scheduler.close)
            if cleanup("worker", lambda: stop_owned(worker)):
                forced.append("worker")
            if backend == "cxl" and worker is not None:

                def restore():
                    if worker.poll() is None:
                        raise RuntimeError("Cannot restore while GPU Worker is alive")
                    manifest = out / "dax-backup.json"
                    if (
                        manifest.exists()
                        and not json.loads(manifest.read_text())["restored"]
                    ):
                        subprocess.run(
                            [
                                sys.executable,
                                str(
                                    Path(args.worker_script).with_name(
                                        "restore_nsa_dax.py"
                                    )
                                ),
                                str(manifest),
                            ],
                            check=True,
                        )
                    if (
                        not manifest.exists()
                        or not json.loads(manifest.read_text())["restored"]
                    ):
                        raise RuntimeError(f"CXL restoration not verified: {out}")

                cleanup("CXL restore", restore)
            try:
                if pool is not None:
                    if pool.poll() is not None:
                        raise RuntimeError(
                            "Owned RDMA Pool exited before cleanup verification"
                        )
                    rpc = RPC(w["pool_target"])
                    try:
                        pool_state = rpc.call("Status")
                        assert (
                            pool_state["sessions"] == pool_state["allocated_bytes"] == 0
                        )
                    finally:
                        rpc.close()
            except Exception as error:
                cleanup_errors.append(f"RDMA release: {type(error).__name__}: {error}")
            finally:
                if cleanup("pool", lambda: stop_owned(pool)):
                    forced.append("pool")
                (out / "pool-final-state.json").write_text(
                    json.dumps(pool_state, indent=2)
                )
                (out / "run-state.json").write_text(
                    json.dumps(
                        dict(
                            completed=completed and not cleanup_errors and not forced,
                            requests_completed=completed,
                            cleanup_errors=cleanup_errors,
                            forced_process_stops=forced,
                        ),
                        indent=2,
                    )
                )
            if cleanup_errors or forced:
                raise RuntimeError(
                    f"Run cleanup failed: {cleanup_errors}; forced stops: {forced}"
                )
    return rows
