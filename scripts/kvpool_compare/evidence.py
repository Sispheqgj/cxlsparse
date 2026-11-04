"""Small, CPU-only provenance helpers for the system comparison."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashes(folder):
    return {
        str(p.relative_to(folder)): digest(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name != "source-manifest.json"
    }


def freeze(repo, task, destination):
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache", ".venv", "cache"
    )
    for name in ("cxl_recsys", "scripts", "third_party/nsa_runtime", "third_party/GR"):
        shutil.copytree(repo / name, destination / name, ignore=ignore)
    for name in (
        "config.yml",
        "configs/nsa-tma.yml",
        "pyproject.toml",
        "uv.lock",
        "tools/nsa-tilelang/pyproject.toml",
        "tools/nsa-tilelang/uv.lock",
    ):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    control = destination / "experiment"
    control.mkdir()
    for path in sorted(task.glob("*.py")) + sorted(task.glob("*.sh")):
        shutil.copy2(path, control / path.name)
    manifest = hashes(destination)
    write(destination / "source-manifest.json", manifest)
    return manifest


def check_snapshot(source):
    expected = json.loads((source / "source-manifest.json").read_text())
    if hashes(source) != expected:
        raise ValueError(f"Frozen source changed: {source}")
    return expected


def verify_external(output):
    """Compare post-run files with the exact fingerprints taken before setup."""
    original = json.loads((output / "input-model-fingerprints.json").read_text())
    checks = {}
    for name in ("model", "kernels"):
        entry = original[name]
        actual, expected = hashes(Path(entry["path"])), entry["files"]
        checks[name] = dict(
            path=entry["path"],
            passed=actual == expected,
            actual_files=actual,
            missing=sorted(expected.keys() - actual.keys()),
            extra=sorted(actual.keys() - expected.keys()),
            changed=[
                p
                for p in sorted(actual.keys() & expected.keys())
                if actual[p] != expected[p]
            ],
        )
    request = original["requests"]
    actual = digest(request["path"])
    checks["requests"] = dict(
        path=request["path"],
        actual_sha256=actual,
        expected_sha256=request["sha256"],
        passed=actual == request["sha256"],
    )
    result = dict(
        passed=all(c["passed"] for c in checks.values()),
        checks=checks,
        verified_utc=datetime.now(timezone.utc).isoformat(),
        original_fingerprints_sha256=digest(output / "input-model-fingerprints.json"),
    )
    write(output / "external-fingerprints-verified.json", result)
    return result


def provenance(repo, model, kernels, requests, output):
    def command(*args):
        return subprocess.check_output(args, cwd=repo, text=True).strip()

    write(
        output / "environment.json",
        dict(
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            packages={
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            cuda_home=os.environ.get("CUDA_HOME"),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            gpu=command(
                "nvidia-smi",
                "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total",
                "--format=csv",
            ),
            topology=command("nvidia-smi", "topo", "-m"),
        ),
    )
    write(
        output / "input-model-fingerprints.json",
        dict(
            requests=dict(path=str(requests), sha256=digest(requests)),
            model=dict(path=str(model), files=hashes(model)),
            kernels=dict(path=str(kernels), files=hashes(kernels)),
        ),
    )
    write(
        output / "git-state.json",
        dict(
            repository=str(repo),
            head=command("git", "rev-parse", "HEAD"),
            status=command("git", "status", "--short"),
        ),
    )
    (output / "working-tree.patch").write_text(
        command("git", "diff", "--binary") + "\n"
    )
