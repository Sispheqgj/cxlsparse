"""Matched CXL/RDMA system baselines; every case runs one frozen working tree."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

from cases import CASES, configure, folder
from evidence import check_snapshot, freeze, provenance, verify_external, write

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/nsa-tma.yml")
    parser.add_argument(
        "--snapshot", type=Path, help="Reuse a snapshot produced by an earlier run"
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=REPO / "results/final/request-source.json",
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--histories", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--skip-mixed", action="store_true")
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--frozen", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repository", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(
            *args.histories,
            *args.lengths,
            args.rounds,
            args.repeats,
            args.warmups,
            args.prefill_chunk,
        )
        < 1
        or max(args.lengths) > 1024
    ):
        parser.error("Positive counts and Extend <= 1024 are required")
    for name in ("histories", "lengths", "cases"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"Duplicate {name}")
    for name in ("output", "requests", "snapshot", "frozen", "repository", "config"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    return args


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.snapshot or args.output / "source"
    manifest = (
        check_snapshot(source)
        if args.snapshot
        else freeze(args.repository, TASK, source)
    )
    write(args.output / "source-sha256.json", manifest)
    write(
        args.output / "source.json",
        dict(
            path=source,
            manifest_sha256=hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
        ),
    )
    shutil.copy2(args.requests, args.output / "request-source.json")
    shutil.copy2(args.config, args.output / "config-source.yml")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(
        map(
            str,
            (
                source,
                source / "third_party",
                args.repository / "third_party",
            ),
        )
    )
    env["NSA_RUNTIME_SNAPSHOT"] = str(source)
    # Restart before importing Scheduler or the runner shared with Worker.
    # Pool's `python -m cxl_recsys.cli` inherits this same PYTHONPATH and cwd.
    os.chdir(source)
    os.execve(
        sys.executable,
        [
            sys.executable,
            str(source / "experiment/runner.py"),
            *sys.argv[1:],
            "--frozen",
            str(source),
            "--repository",
            str(args.repository),
            "--output",
            str(args.output),
            "--snapshot",
            str(source),
            "--requests",
            str(args.output / "request-source.json"),
            "--config",
            str(args.output / "config-source.yml"),
        ],
        env,
    )


def execute(args):
    import benchmark as bench
    import yaml

    source = args.frozen
    manifest = check_snapshot(source)
    if any(
        os.environ.get(name)
        for name in (
            "NSA_TMA_ORDER_PROBE",
            "NSA_TMA_WINDOW_PROBE",
            "NSA_TMA_PREPARE_PROBE",
        )
    ):
        raise ValueError("Formal runs cannot enable diagnostic probes")
    locks = []
    prefix = os.environ.get("NSA_LOCK_PREFIX", f"/tmp/nsa-kvpool-{os.getuid()}")
    for name in ("queue", "gpu0"):
        path = Path(f"{prefix}-{name}.lock")
        handle = path.open("r+" if path.exists() else "x+")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    bench.require_idle_gpu()
    args.baseline_snapshot = args.runtime_snapshot = source
    args.worker_script = source / "scripts/nsa_gr_worker.py"
    args.variants = list(args.cases)
    args.order_probe = args.window_probe = args.prepare_probe = False
    base = yaml.safe_load(args.config.read_text())
    for name in ("model_path", "kernel_dir"):
        base["worker"][name] = str((args.repository / base["worker"][name]).resolve())
    write(args.output / "parameters.json", vars(args))
    write(args.output / "base-config.json", base)
    provenance(
        args.repository,
        Path(base["worker"]["model_path"]),
        Path(base["worker"]["kernel_dir"]),
        args.requests,
        args.output,
    )
    write(
        args.output / "runtime-imports.json",
        dict(
            benchmark=bench.__file__,
            scheduler=bench.Scheduler.__module__,
            snapshot=source,
            pythonpath=os.environ["PYTHONPATH"],
            cwd=Path.cwd(),
            all_variants_share_source=True,
        ),
    )
    source_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    rows, order = [], []
    for round_index in range(args.rounds):
        histories = list(args.histories)
        random.Random(args.seed + round_index).shuffle(histories)
        cases = list(args.cases)
        if round_index % 2:
            cases.reverse()
        for history in histories:
            for case in cases:
                backend, variant = CASES[case]["backend"], case
                config = copy.deepcopy(base)
                configure(bench, config["worker"], case)
                config["worker"]["dax_bytes"] = [
                    ((history * 8192 + 2**21 - 1) // 2**21) * 2**21
                ]
                order.append(
                    dict(
                        round=round_index,
                        history=history,
                        backend=backend,
                        variant=variant,
                    )
                )
                write(args.output / "run-order.json", order)
                print(f"START round={round_index} H={history} {case}", flush=True)
                result = bench.case(
                    args, config, history, round_index, backend, variant
                )
                case_folder = folder(args.output, round_index, case, history)
                write(
                    case_folder / "runtime-source.json",
                    dict(
                        runtime_snapshot=source,
                        source_manifest_sha256=source_digest,
                        backend=backend,
                        variant=variant,
                    ),
                )
                rows.extend(result)
                write(args.output / "results.json", rows)
                print(f"DONE round={round_index} H={history} {case}", flush=True)
    check_snapshot(source)
    if not verify_external(args.output)["passed"]:
        raise RuntimeError(
            "External model, kernels or request file changed during the run"
        )
    write(
        args.output / "complete.json",
        dict(passed=True, samples=len(rows), cases=len(order)),
    )


def main():
    args = arguments()
    if args.frozen is None:
        prepare(args)
    uid = int(os.environ.get("SUDO_UID", os.getuid()))
    gid = int(os.environ.get("SUDO_GID", os.getgid()))
    os.umask(0o007)
    try:
        execute(args)
    finally:
        for folder in (
            args.output,
            Path(os.environ.get("NSA_CACHE_DIR", f"/dev/shm/nsa-kvpool-{uid}")),
        ):
            if folder.exists():
                for item in (folder, *folder.rglob("*")):
                    os.chown(item, uid, gid)


if __name__ == "__main__":
    main()
