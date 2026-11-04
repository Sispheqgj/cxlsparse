import argparse
import json
import logging
import signal
import threading

from .common.rpc import serve

logger = logging.getLogger("cxl_recsys.cli")


def parse_args(argv=None):
    from .common.config import configure
    from .common.sizes import dax_regions, dax_values, parse_bytes
    from .models import MODEL_NAMES

    # Only the config locator is bootstrapped outside YAML (it locates YAML).
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    selected, _ = probe.parse_known_args(argv)
    parser = argparse.ArgumentParser(
        description="NSA memory-semantic KV pool with CXL and RDMA backends"
    )
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Optional YAML override of project config.yml",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    scheduler_parser = sub.add_parser("scheduler")
    scheduler_parser.add_argument("--workers", nargs="+")
    scheduler_parser.add_argument("--max-pending", type=int)
    scheduler_parser.add_argument("--batch-size", type=int)
    scheduler_parser.add_argument("--max-history", type=int)
    scheduler_parser.add_argument(
        "--source", choices=["gr", "generated", "requests", "dataset"]
    )
    scheduler_parser.add_argument(
        "--input", help="JSON array or JSONL; gr without --input generates requests"
    )
    scheduler_parser.add_argument("--output")
    scheduler_parser.add_argument(
        "--save-requests", help="Save the processed sequence as JSON for replay"
    )
    scheduler_parser.add_argument("--num-requests", type=int)
    scheduler_parser.add_argument("--users", type=int)
    scheduler_parser.add_argument("--history-length", type=int)
    scheduler_parser.add_argument("--append-tokens", type=int)
    scheduler_parser.add_argument("--candidates", type=int)
    scheduler_parser.add_argument(
        "--sampling", choices=["sequential", "random", "zipf"]
    )
    scheduler_parser.add_argument("--zipf-alpha", type=float)
    scheduler_parser.add_argument("--seed", type=int)
    scheduler_parser.add_argument(
        "--gr-heat-source",
        choices=["curve", "beauty", "games", "books", "clothing", "industrial"],
    )
    scheduler_parser.add_argument("--gr-heat-path")
    scheduler_parser.add_argument("--gr-curve-dataset")
    scheduler_parser.add_argument("--gr-curve-field")
    scheduler_parser.add_argument(
        "--gr-industrial-heat-field", choices=["pv_share", "pv_int", "pv_scaled_1_100"]
    )
    scheduler_parser.add_argument("--gr-data-root")
    scheduler_parser.add_argument("--gr-tokenizer")
    scheduler_parser.add_argument(
        "--gr-prompt-format", choices=["deepseek", "nosa", "nsa"]
    )
    scheduler_parser.add_argument(
        "--gr-text-material", choices=["catalog", "synthetic"]
    )
    scheduler_parser.add_argument(
        "--gr-text-dataset", choices=["beauty", "games", "books", "clothing"]
    )
    scheduler_parser.add_argument("--gr-text-catalog-path")
    scheduler_parser.add_argument("--gr-user-lengths", type=int, nargs="+")
    scheduler_parser.add_argument("--gr-user-probabilities", type=float, nargs="+")
    scheduler_parser.add_argument("--gr-item-lengths", type=int, nargs="+")
    scheduler_parser.add_argument("--gr-item-probabilities", type=float, nargs="+")
    scheduler_parser.add_argument("--gr-candidate-count", type=int)
    scheduler_parser.add_argument("--gr-history-cache-users", type=int)
    scheduler_parser.add_argument("--gr-qps", type=int)
    scheduler_parser.add_argument(
        "--gr-arrival", choices=["poisson", "constant", "timeslot"]
    )
    scheduler_parser.add_argument(
        "--gr-sampling", choices=["weighted", "uniform", "sequential"]
    )
    scheduler_parser.add_argument("--run-timeout", type=float)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--listen")
    worker_parser.add_argument("--backend", choices=["dram", "cxl", "rdma"])
    worker_parser.add_argument("--model", choices=MODEL_NAMES)
    worker_parser.add_argument(
        "--model-path", help="Local sparse model checkpoint directory"
    )
    worker_parser.add_argument("--nsa-query-tile", type=int)
    worker_parser.add_argument("--nsa-kernel-backend", choices=["fla", "torch"])
    worker_parser.add_argument(
        "--nsa-selected-backend",
        choices=[
            "triton",
            "triton_gr",
            "tilelang",
            "tilelang_indexed",
            "tilelang_gpu",
            "tilelang_gr",
            "tilelang_tma",
        ],
    )
    worker_parser.add_argument(
        "--nsa-cross-query-dedup", action=argparse.BooleanOptionalAction
    )
    worker_parser.add_argument(
        "--nsa-rdma-registration", choices=["persistent", "per_call"]
    )
    worker_parser.add_argument("--nsa-selected-mode", choices=["serial", "overlap"])
    worker_parser.add_argument(
        "--nsa-selected-order", choices=["strict", "resident_first", "block_ready"]
    )
    worker_parser.add_argument(
        "--nsa-selected-ready-window", type=int, choices=[1, 2, 4]
    )
    worker_parser.add_argument(
        "--nsa-selected-read-order",
        choices=[
            "physical",
            "head_interleaved",
            "query_first_use",
            "query_first_use_region",
            "consumption_depth",
            "consumption_depth_shared",
        ],
    )
    worker_parser.add_argument("--nsa-selected-read-group-size", type=int)
    worker_parser.add_argument("--nsa-selected-max-queries", type=int)
    worker_parser.add_argument(
        "--nsa-selected-query-group", type=int, choices=[1, 2, 4, 8]
    )
    worker_parser.add_argument("--nsa-selected-workspace-mib", type=int)
    worker_parser.add_argument(
        "--nsa-cache-backend", choices=["auto", "reference", "device"]
    )
    worker_parser.add_argument(
        "--nsa-cuda-graphs", action=argparse.BooleanOptionalAction
    )
    worker_parser.add_argument("--nsa-graph-cache-size", type=int)
    worker_parser.add_argument("--pages", type=int)
    worker_parser.add_argument(
        "--page-size", type=int, help="Tokens per backing allocation page"
    )
    worker_parser.add_argument("--dim", type=int)
    worker_parser.add_argument("--layers", type=int)
    worker_parser.add_argument("--max-users", type=int)
    worker_parser.add_argument("--max-history", type=int)
    worker_parser.add_argument("--hot", type=int)
    worker_parser.add_argument("--topk", type=int)
    for name in (
        "n-heads",
        "head-dim",
        "q-lora-rank",
        "rope-dim",
        "original-seq-len",
        "seed",
        "query-tile",
    ):
        worker_parser.add_argument(f"--indexer-{name}", type=int)
    for name in ("rope-theta", "rope-factor", "beta-fast", "beta-slow"):
        worker_parser.add_argument(f"--indexer-{name}", type=float)
    worker_parser.add_argument(
        "--kernel-dir", help="Directory containing prebuilt CUDA libraries"
    )
    worker_parser.add_argument("--queue-capacity", type=int)
    worker_parser.add_argument("--max-batch-size", type=int)
    worker_parser.add_argument(
        "--timing-breakdown",
        action=argparse.BooleanOptionalAction,
        help="Profile Worker/model phases with synchronized GPU boundaries",
    )
    worker_parser.add_argument(
        "--dax-device", nargs="+", help="Ordered daxN.M names or /dev paths"
    )
    worker_parser.add_argument(
        "--dax-offset",
        nargs="+",
        type=parse_bytes,
        help="Per-region offsets: bytes or e.g. 128GiB",
    )
    worker_parser.add_argument(
        "--dax-bytes",
        nargs="+",
        type=parse_bytes,
        help="Per-region reserved sizes: bytes or e.g. 128GiB",
    )
    worker_parser.add_argument("--pool-target")
    pool_parser = sub.add_parser("pool")
    pool_parser.add_argument("--listen")
    pool_parser.add_argument("--max-bytes", type=int)
    build_parser = sub.add_parser(
        "build-kernels",
        help="Offline build of all kernels required by the Worker profile",
    )
    build_parser.add_argument("--kernel-dir")
    build_parser.add_argument("--hot", type=int)
    build_parser.add_argument("--dim", type=int)
    build_parser.add_argument("--backend", choices=["dram", "cxl", "rdma"])
    build_parser.add_argument("--model", choices=MODEL_NAMES)
    build_parser.add_argument("--model-path")
    build_parser.add_argument(
        "--cuda-arch", help="Target compute capability (auto detects the current GPU)"
    )
    for cmd in (worker_parser, pool_parser):
        cmd.add_argument("--rdma-device")
        cmd.add_argument("--rdma-port", type=int)
        cmd.add_argument("--gid-index", type=int)
    parsers = {
        "scheduler": scheduler_parser,
        "worker": worker_parser,
        "pool": pool_parser,
        "build-kernels": build_parser,
    }
    for command_parser in parsers.values():
        command_parser.add_argument("--log-dir")
        command_parser.add_argument(
            "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        command_parser.add_argument("--log-max-bytes", type=int)
        command_parser.add_argument("--log-backup-count", type=int)
        command_parser.add_argument(
            "--config",
            default=argparse.SUPPRESS,
            help="Optional YAML override of project config.yml",
        )
    try:
        configure(parsers, selected.config)
    except ValueError as exc:
        parser.error(str(exc))
    args = parser.parse_args(argv)
    if args.command == "worker":
        try:
            for key in ("dax_device", "dax_offset", "dax_bytes"):
                setattr(args, key, dax_values(key, getattr(args, key)))
            if args.backend == "cxl":
                dax_regions(args.dax_device, args.dax_offset, args.dax_bytes)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main():
    from .common.logging import close_logging, configure_logging

    args = parse_args()
    configure_logging(
        "mempool" if args.command in ("pool", "build-kernels") else args.command,
        args.log_dir,
        args.log_level,
        args.log_max_bytes,
        args.log_backup_count,
    )
    try:
        logger.info("service_starting role=%s", args.command)
        run(args)
    except SystemExit as exc:
        logger.error("service_exit code=%s", exc.code)
        raise
    except Exception:
        logger.exception("service_failed role=%s", args.command)
        raise
    finally:
        logger.info("service_stopped role=%s", args.command)
        close_logging()


def run(args):
    model_config = None
    if args.command in ("worker", "build-kernels") and args.model == "nsa_1b":
        from cxl_recsys.models.nsa.config import prepare_nsa

        model_config = prepare_nsa(args)
    if args.command == "build-kernels":
        from .mempool.build_kernels import build_configured_kernels

        libraries = build_configured_kernels(args)
        print(json.dumps({"built": [str(path) for path in libraries]}), flush=True)
        return
    if args.command == "scheduler":
        from pathlib import Path

        from .scheduler import Scheduler
        from .scheduler.workload import load_workload

        # Load and validate the ENTIRE sequence before dispatching any request.
        requests = load_workload(args)
        logger.debug(
            "workload_loaded source=%s requests=%d", args.source, len(requests)
        )
        output = Path(args.output)
        saved = Path(args.save_requests) if args.save_requests else None
        targets = [output] + ([saved] if saved else [])
        if len({path.resolve() for path in targets}) != len(targets):
            raise ValueError("Output and saved request paths must differ")
        for path in targets:
            if path.exists() or (
                args.input and path.resolve() == Path(args.input).resolve()
            ):
                raise ValueError(f"Refusing to overwrite existing file: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
        if saved:
            with saved.open("x") as f:
                json.dump(requests, f, indent=2)
        stopped = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stopped.set())
        service = Scheduler(
            args.workers,
            args.max_pending,
            max_history=args.max_history,
            batch_size=args.batch_size,
        )
        try:
            report = service.run_sequence(requests, args.run_timeout, stopped)
            report["source"] = args.source
            with output.open("x") as f:
                json.dump(report, f, indent=2)
            logger.info(
                "report_saved state=%s succeeded=%d total=%d output=%s",
                report["state"],
                report["succeeded"],
                report["total"],
                output,
            )
            print(
                json.dumps(
                    {
                        "state": report["state"],
                        "succeeded": report["succeeded"],
                        "total": report["total"],
                        "output": str(output),
                    }
                ),
                flush=True,
            )
        finally:
            service.close()
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.command == "pool":
        from .mempool.rdma import PoolManager, Region

        # Fail at startup, not only on the first request, when link/MR unavailable.
        probe = Region(4096, args.rdma_device, args.rdma_port, args.gid_index)
        probe.close()
        service = PoolManager(
            args.max_bytes, args.rdma_device, args.rdma_port, args.gid_index
        )
        methods = {
            "Open": service.open,
            "Close": service.release,
            "Status": service.status,
        }
    else:
        from .models import resolve_model

        # Reject unavailable models before CUDA initialization or backing allocation.
        model_type = resolve_model(args.model)
        from cxl_recsys.models.simulation.config import IndexerConfig

        indexer = IndexerConfig(
            **{
                name: getattr(args, "indexer_" + name)
                for name in IndexerConfig.__dataclass_fields__
            }
        )
        logger.info("model_selected model=%s", args.model)
        from .mempool import MemoryPool
        from .mempool.kernels import prepare_worker_kernels
        from .worker import Worker

        # Resolve every supported chunk length before mapping CXL, contacting an
        # RDMA pool, starting the execution thread or announcing service_ready.
        prepare_worker_kernels(args)
        pool = MemoryPool(
            args.pages,
            args.dim,
            args.backend,
            page_size=args.page_size,
            dax_device=args.dax_device,
            dax_offset=args.dax_offset,
            dax_bytes=args.dax_bytes,
            pool_target=args.pool_target,
            device=args.rdma_device,
            port=args.rdma_port,
            gid_index=args.gid_index,
            kernel_dir=args.kernel_dir,
        )
        try:
            model_options = {}
            if model_config is not None:
                model_options = {
                    "model_path": args.model_path,
                    "config": model_config,
                    "query_tile": args.nsa_query_tile,
                }
                if args.model == "nsa_1b":
                    model_options.update(
                        kernel_backend=args.nsa_kernel_backend,
                        cuda_graphs=args.nsa_cuda_graphs,
                        graph_cache_size=args.nsa_graph_cache_size,
                        cache_backend=args.nsa_cache_backend,
                        selected_backend=args.nsa_selected_backend,
                        cross_query_dedup=args.nsa_cross_query_dedup,
                        rdma_registration=args.nsa_rdma_registration,
                        selected_mode=args.nsa_selected_mode,
                        selected_order=args.nsa_selected_order,
                        selected_ready_window=args.nsa_selected_ready_window,
                        selected_read_order=args.nsa_selected_read_order,
                        selected_read_group_size=args.nsa_selected_read_group_size,
                        selected_max_queries=args.nsa_selected_max_queries,
                        selected_query_group=args.nsa_selected_query_group,
                        selected_workspace_mib=args.nsa_selected_workspace_mib,
                    )
            model = model_type(
                pool,
                args.max_users,
                args.max_history,
                args.layers,
                args.hot,
                args.topk,
                args.dim,
                indexer=indexer,
                **model_options,
            )
            service = Worker(
                model,
                args.queue_capacity,
                max_batch_size=args.max_batch_size,
                timing_breakdown=args.timing_breakdown,
            )
        except Exception:
            pool.close()
            raise
        methods = {
            "SubmitBatch": service.submit_batch,
            "GetBatch": service.get_batch,
            "Submit": service.submit,
            "Get": service.get,
            "Status": service.status,
        }
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    server = None
    try:
        server = serve(args.listen, methods)
        logger.info("service_ready role=%s listen=%s", args.command, args.listen)
        print(
            json.dumps({"ready": True, "service": args.command, "listen": args.listen}),
            flush=True,
        )
        stopped.wait()
    finally:
        if server:
            server.stop(3).wait()
        service.close()


if __name__ == "__main__":
    main()
