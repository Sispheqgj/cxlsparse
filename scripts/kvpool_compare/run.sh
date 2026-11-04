#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELIVERY_ROOT="$(cd "$TASK_DIR/../.." && pwd)"
source "$DELIVERY_ROOT/scripts/nsa_tilelang_environment.sh"
mkdir -p "$NSA_CACHE_DIR"/{tmp,tilelang,triton,host,cuda}
cd "$DELIVERY_ROOT"
launcher=()
if [[ "${NSA_SUDO:-1}" == 1 ]]; then
    launcher=(sudo -n)
fi
affinity=()
if [[ -n "${NSA_CPUS:-}" ]]; then
    affinity=(taskset -c "$NSA_CPUS")
fi
exec "${launcher[@]}" env CUDA_HOME="$CUDA_HOME" PATH="$PATH" PYTHONPATH="$PYTHONPATH" \
  NSA_CACHE_DIR="$NSA_CACHE_DIR" NSA_LOCK_PREFIX="${NSA_LOCK_PREFIX:-/tmp/nsa-kvpool-${SUDO_UID:-$(id -u)}}" \
  TILELANG_CACHE_DIR="$TILELANG_CACHE_DIR" TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  XDG_CACHE_HOME="$XDG_CACHE_HOME" CUDA_CACHE_PATH="$CUDA_CACHE_PATH" \
  TMPDIR="$NSA_CACHE_DIR/tmp" CPATH="${CPATH:-}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 "${affinity[@]}" "$NSA_PYTHON" "$TASK_DIR/runner.py" "$@"
