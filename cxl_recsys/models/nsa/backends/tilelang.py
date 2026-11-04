"""SM90 selected NSA attention and resumable FP32 online-softmax state.

One CTA processes one query / KV-head row and its complete GQA group. Each
iteration consumes exactly one original NSA block from compact HBM storage.
External IO belongs to the caller's independent producer stream. All CTA
threads remain available for HBM-to-SMEM copies, reductions and tensor cores.

``build_kernel`` returns a precompiled callable with arguments
``(Q, KV, IDs, Starts, Positions, Counts, M, L, Acc, Out)``. No tensor allocation,
device synchronization or host inspection occurs during a call. The caller owns
all buffers and must order each continuation after its preceding state update
and only the KV chunks it references. Compile before CUDA Graph capture.
"""

# TileLang tensor annotations are executable DSL expressions.
# pyright: reportInvalidTypeForm=false, reportAttributeAccessIssue=false

from functools import lru_cache
from typing import Any, cast


def pass_configs() -> dict[str, Any]:
    """Explicit SM90 MMA configuration, shared by runtime and offline lowering."""
    import tilelang

    return {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
    }


def make_prim_func(
    rows,
    groups,
    blocks,
    block_size,
    width,
    initialize=True,
    finalize=True,
    threads=32,
):
    """Build IR without querying or initializing CUDA.

    Q is BF16 [rows, groups, 64]; KV is BF16 [blocks, block_size, 128]
    containing concatenated K64/V64. IDs is int32 [rows, width] with compact
    physical block IDs in each Counts prefix; Starts is int32 [blocks] with
    original absolute token positions. Positions and Counts are int32 [rows].
    M/L are FP32 [rows, groups], Acc FP32 [rows, groups, 64], and Out BF16
    [rows, groups, 64]. Contiguous buffers on one SM90 device are required.

    Counts must be in [0, width]. Negative/invalid IDs are ignored defensively.
    The planner validates ownership, selected sets and physical block bounds.
    A final call writes Out only; a nonfinal call writes M/L/Acc only. Monolithic
    initialize+finalize does not load or store global intermediate state.
    """
    from tilelang import language as T

    geometry = (rows, groups, blocks, block_size, width, threads)
    if any(type(value) is not int for value in geometry):
        raise TypeError("NSA kernel geometry must contain integers")
    if rows < 1 or blocks < 1 or width < 1:
        raise ValueError("NSA kernel rows, block capacity and width must be positive")
    if groups not in (16, 32, 64) or block_size not in (16, 32, 64):
        raise ValueError("NSA kernel supports G16/32/64 and block16/32/64, D64")
    if threads not in (32, 64, 128):
        raise ValueError("NSA kernel supports 32, 64 or 128 threads")
    if threads // 32 > (groups // 16) * (block_size // 8):
        raise ValueError("Too many warps for the QK tile (each needs 16x8 elements)")
    if type(initialize) is not bool or type(finalize) is not bool:
        raise TypeError("initialize/finalize must be boolean")

    @T.macro
    def prefetch(row, slot, IDs, KV, k, v):
        # The entire padded block allocation exists in HBM. Copying its tail
        # never triggers external IO; the tail is zeroed before any arithmetic.
        physical = T.min(T.max(IDs[row, slot], 0), blocks - 1)
        T.async_copy(KV[physical, :, :64], k[slot % 2, :, :])
        T.async_copy(KV[physical, :, 64:], v[slot % 2, :, :])

    @T.prim_func
    def main(
        Q: T.Tensor((rows, groups, 64), "bfloat16"),
        KV: T.Tensor((blocks, block_size, 128), "bfloat16"),
        IDs: T.Tensor((rows, width), "int32"),
        Starts: T.Tensor((blocks,), "int32"),
        Positions: T.Tensor((rows,), "int32"),
        Counts: T.Tensor((rows,), "int32"),
        M: T.Tensor((rows, groups), "float32"),
        L: T.Tensor((rows, groups), "float32"),
        Acc: T.Tensor((rows, groups, 64), "float32"),
        Out: T.Tensor((rows, groups, 64), "bfloat16"),
    ):
        with T.Kernel(rows, threads=threads) as row:
            q = T.alloc_shared((groups, 64), "bfloat16")
            k = T.alloc_shared((2, block_size, 64), "bfloat16")
            v = T.alloc_shared((2, block_size, 64), "bfloat16")
            probability = T.alloc_shared((groups, block_size), "bfloat16")
            score = T.alloc_fragment((groups, block_size), "float32")
            acc = T.alloc_fragment((groups, 64), "float32")
            maximum = T.alloc_fragment((groups,), "float32")
            previous = T.alloc_fragment((groups,), "float32")
            denominator = T.alloc_fragment((groups,), "float32")
            factor = T.alloc_fragment((groups,), "float32")
            block_sum = T.alloc_fragment((groups,), "float32")

            # Match native FLA: BF16 rounding happens before QK, not afterward.
            for h, d in T.Parallel(groups, 64):
                q[h, d] = Q[row, h, d].astype("float32") * 0.125
            if initialize:
                T.fill(maximum, -T.infinity("float32"))
                T.clear(denominator)
                T.clear(acc)
            else:
                T.copy(M[row, :], maximum)
                T.copy(L[row, :], denominator)
                T.copy(Acc[row, :, :], acc)

            if Counts[row] > 0:
                prefetch(row, 0, IDs, KV, k, v)
                T.ptx_wait_group(0)
            T.sync_threads()
            for slot in T.serial(Counts[row]):
                if slot + 1 < Counts[row]:
                    prefetch(row, slot + 1, IDs, KV, k, v)
                physical = IDs[row, slot]
                if (physical >= 0) & (physical < blocks):
                    # Causal tails may never have been copied from the host.
                    # Zero both operands before GEMM: a zero P times NaN V
                    # would otherwise poison PV despite the softmax mask.
                    for n, d in T.Parallel(block_size, 64):
                        if Starts[physical] + n > Positions[row]:
                            k[slot % 2, n, d] = 0
                            v[slot % 2, n, d] = 0
                    T.sync_threads()
                    T.clear(score)
                    T.gemm(q, k[slot % 2, :, :], score, transpose_B=True)
                    for h, n in T.Parallel(groups, block_size):
                        score[h, n] = T.if_then_else(
                            Starts[physical] + n <= Positions[row],
                            score[h, n],
                            -T.infinity("float32"),
                        )
                    T.copy(maximum, previous)
                    T.reduce_max(score, maximum, dim=1, clear=False)
                    for h in T.Parallel(groups):
                        # An all-masked row remains the identity (-inf, 0, 0).
                        factor[h] = T.if_then_else(
                            previous[h] == -T.infinity("float32"),
                            0.0,
                            T.exp2((previous[h] - maximum[h]) * 1.4426950408889634),
                        )
                    for h, n in T.Parallel(groups, block_size):
                        score[h, n] = T.if_then_else(
                            score[h, n] == -T.infinity("float32"),
                            0.0,
                            T.exp2((score[h, n] - maximum[h]) * 1.4426950408889634),
                        )
                    T.reduce_sum(score, block_sum, dim=1)
                    for h in T.Parallel(groups):
                        denominator[h] = denominator[h] * factor[h] + block_sum[h]
                    for h, d in T.Parallel(groups, 64):
                        acc[h, d] *= factor[h]
                    T.copy(score, probability)
                    T.gemm(probability, v[slot % 2, :, :], acc)
                # This wait concerns HBM -> SMEM only. The IO-stream event
                # dependency is owned by the caller, once per external chunk.
                T.ptx_wait_group(0)
                T.sync_threads()

            if finalize:
                for h, d in T.Parallel(groups, 64):
                    Out[row, h, d] = acc[h, d] / T.if_then_else(
                        denominator[h] > 0, denominator[h], 1.0
                    )
            else:
                T.copy(maximum, M[row, :])
                T.copy(denominator, L[row, :])
                T.copy(acc, Acc[row, :, :])

    phase = (
        "monolithic"
        if initialize and finalize
        else ("resident" if initialize else ("final" if finalize else "missing"))
    )
    return cast(Any, main).with_attr("global_symbol", f"nsa_{phase}")


@lru_cache(maxsize=128)
def build_kernel(
    rows,
    groups,
    blocks,
    block_size,
    width,
    initialize=True,
    finalize=True,
    threads=32,
):
    """Compile one specialization before timing/capture; no silent fallbacks."""
    import tilelang

    function = make_prim_func(
        rows, groups, blocks, block_size, width, initialize, finalize, threads
    )
    return tilelang.compile(
        function,
        target={"kind": "cuda", "arch": "sm_90a"},
        pass_configs=pass_configs(),
    )
