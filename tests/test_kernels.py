"""Prebuilt-library coverage, startup rejection and compiler-free CUDA execution."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import tvm_ffi
import tvm_ffi.cpp
import yaml

from cxl_recsys.cli import parse_args, run
from cxl_recsys.mempool import MemoryPool, build_kernels, kernels
from cxl_recsys.mempool.hisparse import HiSparseCache


def forbid_compile(*args, **kwargs):
    raise AssertionError("Runtime must not invoke a compiler or spawn a child process")


def forbid_compilers(monkeypatch):
    for name in ("load_inline", "build_inline", "load", "build"):
        monkeypatch.setattr(tvm_ffi.cpp, name, forbid_compile)
    monkeypatch.setattr(subprocess, "Popen", forbid_compile)


def fake_library(tmp_path, spec):
    folder = spec.directory(tmp_path)
    folder.mkdir(parents=True)
    (folder / f"{spec.name}.so").write_bytes(b"mock shared library")
    (folder / "metadata.json").write_text(json.dumps(spec.metadata()))
    return folder


def test_build_inherits_effective_worker_config_and_cli_wins(tmp_path):
    path = tmp_path / "experiment.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "worker": {
                    "hot": 64,
                    "dim": 128,
                    "backend": "cxl",
                    "kernel_dir": "my-kernels",
                },
                "build-kernels": {"cuda_arch": "8.0"},
            }
        )
    )
    for prefix in (
        ["--config", str(path), "build-kernels"],
        ["build-kernels", "--config", str(path)],
    ):
        args = parse_args(prefix)
        assert (args.hot, args.dim, args.backend, args.kernel_dir) == (
            64,
            128,
            "cxl",
            "my-kernels",
        )
        assert args.cuda_arch == "8.0"
        assert parse_args([*prefix, "--hot", "4"]).hot == 4
    path.write_text(
        yaml.safe_dump(
            {
                "worker": {"hot": 64},
                "build-kernels": {"hot": 8},
            }
        )
    )
    assert parse_args(["build-kernels", "--config", str(path)]).hot == 8


@pytest.mark.parametrize("backend", ["dram", "rdma", "cxl"])
def test_builder_selects_only_required_libraries(backend, monkeypatch):
    built = []

    def record(spec, directory):
        built.append(spec)
        return Path(directory) / f"{spec.name}.so"

    monkeypatch.setattr(build_kernels, "build_library", record)
    args = parse_args(
        [
            "build-kernels",
            "--cuda-arch",
            "8.0",
            "--hot",
            "32",
            "--dim",
            "128",
            "--backend",
            backend,
        ]
    )
    paths = build_kernels.build_configured_kernels(args)
    assert built[0] == kernels.KernelSpec("planner", 32, "8.0")
    assert len(paths) == (2 if backend == "cxl" else 1)
    if backend == "cxl":
        assert built[1] == kernels.KernelSpec("transfer", 256, "8.0")


def test_generated_exports_cover_layer_pipeline_and_preserve_transfer_templates():
    planner = build_kernels.kernel_source(kernels.KernelSpec("planner", 32, "8.0"))
    assert planner.count("TVM_FFI_DLL_EXPORT_TYPED_FUNC") == 3
    for name in ("plan_layer", "gather_layer", "update_hot"):
        assert f"cxl_recsys::{name}<32>" in planner
    assert "load_cache_to_device_buffer" not in planner
    for width, unroll in ((128, 4), (256, 4), (1024, 2), (1152, 1)):
        source = build_kernels.kernel_source(
            kernels.KernelSpec("transfer", width, "8.0")
        )
        assert f"HiCacheKernel<{width}, {unroll}, 2, 1024>::run_one_mla" in source
        assert f"cxl_recsys::write_regions<{width}, {unroll}>" in source
        assert "write_regions" in kernels.KernelSpec("transfer", width, "8.0").exports
        assert source.count("TVM_FFI_DLL_EXPORT_TYPED_FUNC") == 7
        assert "copy_cache_planned<1024, true, false, false>" in source


@pytest.mark.parametrize("hot", [0, -1, 3, 32768, True])
def test_invalid_hot_is_rejected(hot):
    with pytest.raises(ValueError, match="hot"):
        kernels.KernelSpec("planner", hot, "8.0")


@pytest.mark.parametrize("architecture", ["auto", "sm80", "8", "8.0+PTX", "8.0;9.0"])
def test_library_requires_one_explicit_architecture(architecture):
    with pytest.raises(ValueError):
        kernels.KernelSpec("planner", 4, architecture)


@pytest.mark.parametrize(
    "failure",
    [
        "library",
        "metadata",
        "invalid_json",
        "abi",
        "upstream",
        "ffi",
        "arch",
        "export",
        "binary",
    ],
)
def test_bad_artifact_fails_without_compilation(tmp_path, monkeypatch, failure):
    spec = kernels.KernelSpec("planner", 4, "8.0")
    folder = fake_library(tmp_path, spec)
    module = SimpleNamespace(**{name: lambda: None for name in spec.exports})
    monkeypatch.setattr(tvm_ffi, "load_module", lambda _: module)
    if failure == "library":
        (folder / f"{spec.name}.so").unlink()
    elif failure == "metadata":
        (folder / "metadata.json").unlink()
    elif failure == "invalid_json":
        (folder / "metadata.json").write_text("invalid")
    elif failure == "export":
        del module.gather_layer
    elif failure == "binary":

        def broken(_):
            raise OSError("invalid ELF")

        monkeypatch.setattr(tvm_ffi, "load_module", broken)
    else:
        field = {
            "abi": "kernel_abi",
            "upstream": "upstream_commit",
            "ffi": "tvm_ffi_version",
            "arch": "architecture",
        }[failure]
        metadata = spec.metadata()
        metadata[field] = "incompatible"
        (folder / "metadata.json").write_text(json.dumps(metadata))
    forbid_compilers(monkeypatch)
    with pytest.raises(RuntimeError, match="Runtime compilation is disabled"):
        kernels.load_library(spec, tmp_path)


def test_loading_only_runtime_library_uses_process_cache(tmp_path, monkeypatch):
    spec = kernels.KernelSpec("planner", 4, "8.0")
    folder = fake_library(tmp_path, spec)
    loaded = []
    module = SimpleNamespace(**{name: lambda: None for name in spec.exports})

    def load(path):
        loaded.append(path)
        return module

    monkeypatch.setattr(tvm_ffi, "load_module", load)
    forbid_compilers(monkeypatch)
    assert kernels.load_library(spec, tmp_path) is module
    assert kernels.load_library(spec, tmp_path) is module
    assert loaded == [str(folder / f"{spec.name}.so")]


@pytest.mark.parametrize("missing", ["planner", "transfer"])
def test_worker_fails_before_pool_or_server_when_library_missing(
    tmp_path, monkeypatch, missing
):
    import cxl_recsys.mempool as mempool

    monkeypatch.setattr(kernels, "cuda_architecture", lambda *_: "8.0")
    if missing == "transfer":
        spec = kernels.KernelSpec("planner", 32, "8.0")
        fake_library(tmp_path, spec)
        module = SimpleNamespace(**{name: lambda: None for name in spec.exports})
        monkeypatch.setattr(tvm_ffi, "load_module", lambda _: module)
    monkeypatch.setattr(mempool, "MemoryPool", forbid_compile)
    forbid_compilers(monkeypatch)
    args = parse_args(["worker", "--backend", "cxl", "--kernel-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match=f"cxl_{missing}"):
        run(args)


@pytest.mark.gpu
def test_layer_shapes_execute_without_compiler(monkeypatch):
    # A prebuilt hot=32 library must already exist. Tests never build fixtures.
    torch.cuda.init()
    kernels._load_library.cache_clear()
    pool = MemoryPool(2, 4)
    try:
        pool.allocate(2)
        source = torch.arange(128 * 4).reshape(128, 4).bfloat16()
        pool.write_many(list(range(128)), source)
        forbid_compilers(monkeypatch)
        cache = HiSparseCache(pool, rows=1, max_history=128, hot=32, topk=8, dim=4)
        cache.bind(0, list(range(128)))
        for count in (1, 3, 8, 31, 32, 48, 64):
            selected = list(range(64 - count, 64))
            batch, _ = cache.fetch_layer(
                [
                    {
                        "row": 0,
                        "indices": torch.tensor(selected, device="cuda").reshape(-1, 1),
                        "history_length": 128,
                        "cached_length": 128,
                        "current_kv": torch.empty(
                            (count, 4), device="cuda", dtype=torch.bfloat16
                        ),
                    }
                ]
            )
            assert torch.equal(
                batch.values[batch.indices[:, 0]].cpu(), source[selected]
            )
    finally:
        pool.close()


@pytest.mark.gpu
def test_direct_transfer_load_and_execute_without_compiler(monkeypatch):
    torch.cuda.init()
    kernels._load_library.cache_clear()
    pool = MemoryPool(2, 128)
    pool.backend = "cxl"  # Registered anonymous mapping, not physical CXL.
    try:
        pool.allocate(2)
        source = torch.arange(3 * 128).reshape(3, 128).bfloat16().cuda()
        forbid_compilers(monkeypatch)
        pool.write_many([0, 63, 64], source)
        output = torch.empty_like(source)
        pool.read_cuda([64, 0, 63], output)
        assert torch.equal(output, source[[2, 0, 1]])
        assert pool.read_kernel_launches == pool.write_kernel_launches == 1
    finally:
        pool.close()
