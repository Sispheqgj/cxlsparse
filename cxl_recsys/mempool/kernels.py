"""Load prebuilt HiSparse libraries. This module never invokes a compiler."""

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ..common.paths import PROJECT_ROOT

UPSTREAM_COMMIT = "8ae962021dbb5eb8ee408c578e772c13e2c934fe"
# Bump this when changing project-owned exports, template flags or native glue.
KERNEL_ABI = 8
DEFAULT_KERNEL_DIR = PROJECT_ROOT / "artifacts/kernels-profile-v8"
logger = logging.getLogger(__name__)


def cuda_architecture(architecture: str = "auto") -> str:
    if architecture == "auto":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Worker requires CUDA; no CPU kernel fallback exists")
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}"
    if not re.fullmatch(r"[1-9][0-9]*\.[0-9]", architecture):
        raise ValueError("cuda_arch must be auto or a compute capability such as 8.0")
    return architecture


def validate_hot(hot: int) -> None:
    if type(hot) is not int or not 0 < hot <= 16384 or hot & (hot - 1):
        raise ValueError("hot must be a power of two, 0 < hot <= 16384")


def validate_token_bytes(token_bytes: int) -> None:
    if type(token_bytes) is not int or token_bytes <= 0 or token_bytes % 128:
        raise ValueError("Upstream CXL transfer kernels require token_bytes % 128 == 0")


@dataclass(frozen=True)
class KernelSpec:
    family: str
    size: int
    architecture: str

    def __post_init__(self):
        if self.architecture == "auto":
            raise ValueError("KernelSpec requires an explicit CUDA architecture")
        cuda_architecture(self.architecture)
        if self.family == "planner":
            validate_hot(self.size)
        elif self.family == "transfer":
            validate_token_bytes(self.size)
        else:
            raise ValueError(f"Unknown kernel family: {self.family}")

    @property
    def name(self) -> str:
        arch = self.architecture.replace(".", "")
        return f"cxl_{self.family}_sm{arch}_{self.size}"

    @property
    def exports(self) -> list[str]:
        if self.family == "planner":
            return ["plan_layer", "gather_layer", "update_hot"]
        return [
            "transfer",
            "write_regions",
            "read",
            "register_mapping",
            "unregister_mapping",
            "route_regions",
            "offset_region_destinations",
        ]

    def metadata(self) -> dict:
        return {
            "kernel_abi": KERNEL_ABI,
            "upstream_commit": UPSTREAM_COMMIT,
            "tvm_ffi_version": version("apache-tvm-ffi"),
            "architecture": self.architecture,
            "family": self.family,
            "size": self.size,
            "exports": self.exports,
        }

    def directory(self, kernel_dir: str | Path) -> Path:
        release = (
            f"abi{KERNEL_ABI}-{UPSTREAM_COMMIT[:8]}-ffi{version('apache-tvm-ffi')}"
        )
        return Path(kernel_dir).expanduser().resolve() / release / self.name


def load_library(spec: KernelSpec, kernel_dir: str | Path = DEFAULT_KERNEL_DIR) -> Any:
    # Resolve before caching: identical relative paths in different working
    # directories must not alias the same library.
    return _load_library(spec, str(spec.directory(kernel_dir)))


@lru_cache(maxsize=None)
def _load_library(spec: KernelSpec, directory: str) -> Any:
    import tvm_ffi

    folder = Path(directory)
    library = folder / f"{spec.name}.so"
    hint = (
        "Run 'uv run cxl-recsys build-kernels' with the matching --config, "
        "--hot, --dim, --backend and --kernel-dir before starting Worker. "
        "Runtime compilation is disabled."
    )
    try:
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata != spec.metadata() or not library.is_file():
            raise ValueError("missing library or incompatible build metadata")
        module = tvm_ffi.load_module(str(library))
        # Resolve all entry points before the Worker accepts any requests.
        for name in spec.exports:
            if not callable(getattr(module, name)):
                raise ValueError(f"Missing callable export: {name}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load prebuilt kernel {library}: {exc}. {hint}"
        ) from exc
    logger.info(
        "kernel_loaded family=%s size=%d arch=%s exports=%d path=%s",
        spec.family,
        spec.size,
        spec.architecture,
        len(spec.exports),
        library,
    )
    return module


def load_planner(hot: int, kernel_dir: str | Path = DEFAULT_KERNEL_DIR) -> Any:
    validate_hot(hot)
    return load_library(KernelSpec("planner", hot, cuda_architecture()), kernel_dir)


def load_transfer(token_bytes: int, kernel_dir: str | Path = DEFAULT_KERNEL_DIR) -> Any:
    validate_token_bytes(token_bytes)
    return load_library(
        KernelSpec("transfer", token_bytes, cuda_architecture()), kernel_dir
    )


def prepare_worker_kernels(args) -> None:
    validate_hot(args.hot)
    if not 0 < args.topk <= args.hot:
        raise ValueError("topk must satisfy 0 < topk <= hot")
    load_planner(args.hot, args.kernel_dir)
    if args.backend == "cxl":
        load_transfer(args.dim * 2, args.kernel_dir)
    logger.info(
        "worker_kernels_ready hot=%d dim=%d backend=%s runtime_compile=disabled",
        args.hot,
        args.dim,
        args.backend,
    )
