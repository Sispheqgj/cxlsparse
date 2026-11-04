#!/usr/bin/env bash
# Source from the repository root after installing tools/nsa-tilelang.
NSA_RUNTIME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NSA_PYTHON="${NSA_PYTHON:-$NSA_RUNTIME_ROOT/tools/nsa-tilelang/.venv/bin/python}"
export PYTHONPATH="$NSA_RUNTIME_ROOT:$NSA_RUNTIME_ROOT/third_party${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "$NSA_PYTHON"):$CUDA_HOME/bin:$PATH"
export NSA_CACHE_DIR="${NSA_CACHE_DIR:-/dev/shm/nsa-kvpool-$(id -u)}"
export TILELANG_CACHE_DIR="$NSA_CACHE_DIR/tilelang"
export TRITON_CACHE_DIR="$NSA_CACHE_DIR/triton"
export XDG_CACHE_HOME="$NSA_CACHE_DIR/host"
export CUDA_CACHE_PATH="$NSA_CACHE_DIR/cuda"
