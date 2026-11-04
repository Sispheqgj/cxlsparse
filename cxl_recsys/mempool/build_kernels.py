"""Offline CUDA build entry point; never imported by the Worker runtime."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from ..common.paths import PROJECT_ROOT
from .kernels import UPSTREAM_COMMIT, KernelSpec, cuda_architecture

logger = logging.getLogger(__name__)


def kernel_source(spec: KernelSpec) -> str:
    includes = [
        "#include <tvm/ffi/function.h>",
        "#include <kvcacheio/hisparse.cuh>",
    ]
    if spec.family == "planner":
        includes = ["#include <tvm/ffi/function.h>", "#include <layer_plan.cuh>"]
        # Whole-layer planning is independent of the hot-cache capacity. Query
        # counts and logical extents are runtime values, not JIT specializations.
        exports = [
            f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({name}, (cxl_recsys::{name}<{spec.size}>));"
            for name in spec.exports
        ]
    else:
        includes.extend(
            [
                "#include <kvcacheio/hicache.cuh>",
                "#include <cuda_registration.cuh>",
                "#include <region_plan.cuh>",
                "#include <region_write.cuh>",
            ]
        )
        unroll = 4 if spec.size <= 512 else 2 if spec.size <= 1024 else 1
        exports = [
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(transfer, "
            f"(sglang::HiCacheKernel<{spec.size}, {unroll}, 2, 1024>::run_one_mla));",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(write_regions, "
            f"(cxl_recsys::write_regions<{spec.size}, {unroll}>));",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(read, "
            "(sglang::copy_cache_planned<1024, true, false, false>));",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_mapping, cxl_recsys::register_mapping);",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(unregister_mapping, cxl_recsys::unregister_mapping);",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(route_regions, cxl_recsys::route_regions);",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(offset_region_destinations, cxl_recsys::offset_region_destinations);",
        ]
    return "\n".join(includes + exports) + "\n"


def build_library(spec: KernelSpec, kernel_dir: str | Path) -> Path:
    import tvm_ffi.cpp

    # Isolated worktrees can reuse the pinned source checkout read-only.
    upstream = Path(
        os.environ.get("CXL_RECSYS_SGLANG_SOURCE", PROJECT_ROOT / "third_party/sglang")
    ).resolve()
    jit = upstream / "python/sglang/kernels/jit"
    for header in ("hisparse.cuh", "hicache.cuh"):
        if not (jit / "csrc/kvcacheio" / header).is_file():
            raise RuntimeError(
                "Missing HiSparse source; run git submodule update --init third_party/sglang"
            )
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError("HiSparse source commit mismatch; refusing to build")

    folder = spec.directory(kernel_dir)
    folder.mkdir(parents=True, exist_ok=True)
    major, minor = (int(part) for part in spec.architecture.split("."))
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    previous_arch = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = spec.architecture
    started = time.monotonic()
    logger.info("kernel_build_started name=%s directory=%s", spec.name, folder)
    try:
        library = tvm_ffi.cpp.build_inline(
            name=spec.name,
            cuda_sources=kernel_source(spec),
            build_directory=str(folder),
            extra_include_paths=[
                str(jit / "include"),
                str(jit / "csrc"),
                str(PROJECT_ROOT / "cxl_recsys/mempool/native"),
            ],
            extra_cuda_cflags=[
                "-std=c++20",
                "-O3",
                f"-DSGL_CUDA_ARCH={major * 100 + minor * 10}",
            ],
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous_arch
    (folder / "metadata.json").write_text(json.dumps(spec.metadata(), indent=2) + "\n")
    logger.info(
        "kernel_build_finished name=%s elapsed_ms=%.3f path=%s",
        spec.name,
        (time.monotonic() - started) * 1000,
        library,
    )
    return Path(library)


def build_configured_kernels(args) -> list[Path]:
    architecture = cuda_architecture(args.cuda_arch)
    specs = [KernelSpec("planner", args.hot, architecture)]
    if args.dim <= 0:
        raise ValueError("dim must be positive")
    if args.backend == "cxl":
        specs.append(KernelSpec("transfer", args.dim * 2, architecture))
    return [build_library(spec, args.kernel_dir) for spec in specs]
