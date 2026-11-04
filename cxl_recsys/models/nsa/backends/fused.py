"""N2: cooperative selected NSA with dedicated SM host-IO CTAs.

Consumer CTAs keep every N1 attention thread and online-softmax state. IO is
SM-issued reads from an owned CUDA-registered host allocation, never CE DMA.
The entire grid must pass CUDA cooperative admission; there is no unsafe
noncooperative fallback. Consumer-only resident completion does not stall IO.
"""
# pyright: reportInvalidTypeForm=false, reportAttributeAccessIssue=false

from functools import lru_cache
from typing import Any, cast

from cxl_recsys.models.nsa.backends.tilelang import pass_configs


@lru_cache(maxsize=64)
def fused_kernel(
    rows,
    groups,
    blocks,
    block_size,
    width,
    stages,
    resident_blocks,
    chunk_blocks,
    io_ctas=16,
    pipelined=True,
    use_io=True,
    trace=False,
    manual_sync=False,
    full_tile_fastpath=False,
    address_table=False,
):
    import tilelang
    from tilelang import language as T

    if min(rows, blocks, width, stages, chunk_blocks) < 1:
        raise ValueError("Positive fused NSA geometry required")
    if groups not in (16, 32) or block_size not in (16, 32, 64):
        raise ValueError("N2 initially supports G16/32, BS16/32/64, D64")
    if not 0 <= resident_blocks <= blocks or io_ctas < 1:
        raise ValueError("Invalid resident extent or IO CTA count")
    chunks = stages - 1
    producers = min(io_ctas, blocks - resident_blocks) if use_io else 0
    if (
        use_io
        and chunks != (blocks - resident_blocks + chunk_blocks - 1) // chunk_blocks
    ):
        raise ValueError("Stage count must match missing chunks")
    grid = rows + producers

    @T.macro
    def prefetch(stage, row, slot, IDs, KV, k, v):
        # The entire padded block allocation exists in HBM. Copying its tail
        # never triggers external IO; the tail is zeroed before any arithmetic.
        physical = T.min(T.max(IDs[stage, row, slot], 0), blocks - 1)
        T.async_copy(KV[physical, :, :64], k[slot % 2, :, :])
        T.async_copy(KV[physical, :, 64:], v[slot % 2, :, :])

    @T.macro
    def attend(
        stage,
        row,
        IDs,
        KV,
        Counts,
        Starts,
        Positions,
        q,
        k,
        v,
        score,
        acc,
        maximum,
        previous,
        denominator,
        factor,
        block_sum,
        probability,
    ):
        if Counts[stage, row] > 0:
            prefetch(stage, row, 0, IDs, KV, k, v)
            T.ptx_wait_group(0)
        T.sync_threads()
        for slot in T.serial(Counts[stage, row]):
            if slot + 1 < Counts[stage, row]:
                prefetch(stage, row, slot + 1, IDs, KV, k, v)
            physical = IDs[stage, row, slot]
            if (physical >= 0) & (physical < blocks):
                # Causal tails may never have been copied from the host.
                # Zero both operands before GEMM: a zero P times NaN V
                # would otherwise poison PV despite the softmax mask.
                if (
                    not full_tile_fastpath
                    or Starts[physical] + block_size - 1 > Positions[row]
                ):
                    for n, d in T.Parallel(block_size, 64):
                        if Starts[physical] + n > Positions[row]:
                            k[slot % 2, n, d] = 0
                            v[slot % 2, n, d] = 0
                    T.sync_threads()
                T.clear(score)
                T.gemm(q, k[slot % 2, :, :], score, transpose_B=True)
                if (
                    not full_tile_fastpath
                    or Starts[physical] + block_size - 1 > Positions[row]
                ):
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
                if manual_sync:
                    T.sync_threads()
                T.gemm(probability, v[slot % 2, :, :], acc)
            # This wait concerns HBM -> SMEM only. The IO-stream event
            # dependency is owned by the caller, once per external chunk.
            T.ptx_wait_group(0)
            T.sync_threads()

    @T.prim_func
    def main(
        Q: T.Tensor((rows, groups, 64), "bfloat16"),
        KV: T.Tensor((blocks, block_size, 128), "bfloat16"),
        IDs: T.Tensor((stages, rows, width), "int32"),
        Counts: T.Tensor((stages, rows), "int32"),
        Starts: T.Tensor((blocks,), "int32"),
        Positions: T.Tensor((rows,), "int32"),
        Valid: T.Tensor((blocks,), "int32"),
        Address: T.Tensor((blocks, block_size) if address_table else (1,), "int64"),
        Flags: T.Tensor((chunks + 1,), "int32"),
        Out: T.Tensor((rows, groups, 64), "bfloat16"),
        ComputeTrace: T.Tensor((rows, stages, 2), "uint64"),
        IOTrace: T.Tensor((blocks, 2), "uint64"),
    ):
        with T.Kernel(grid, threads=32) as bx:
            if trace:
                T.import_source(
                    '__device__ __forceinline__ unsigned long long nsa_global_timer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }\n'
                )
            io = T.alloc_shared((block_size, 128), "bfloat16")
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
            if producers > 0:
                if bx == 0:
                    for index in T.Parallel(chunks + 1):
                        Flags[index] = 0
                # This forces a cooperative launch. CUDA rejects an oversized
                # grid before execution instead of starving unscheduled CTAs.
                T.sync_grid()
            if bx < producers:
                for iteration in T.serial(
                    T.ceildiv(blocks - resident_blocks, max(1, producers))
                ):
                    physical = resident_blocks + iteration * producers + bx
                    if physical < blocks:
                        if trace:
                            if T.get_thread_binding() == 0:
                                IOTrace[physical, 0] = T.call_extern(
                                    "uint64", "nsa_global_timer"
                                )
                        for token, vector in T.Parallel(block_size, 16):
                            T.call_extern(
                                "handle",
                                "tl::cp_async_gs_conditional<16>",
                                T.address_of(io[token, vector * 8]),
                                T.reinterpret(
                                    (
                                        Address[physical, token] + vector * 16
                                        if address_table
                                        else Address[0]
                                        + (
                                            physical * block_size * 128
                                            + token * 128
                                            + vector * 8
                                        )
                                        * 2
                                    ),
                                    "handle",
                                ),
                                token < Valid[physical],
                            )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.sync_threads()
                        for token, d in T.Parallel(block_size, 128):
                            if token < Valid[physical]:
                                KV[physical, token, d] = io[token, d]
                        # Every writer fences, then thread 0 publishes through
                        # an acq_rel RMW chain covering every block in the chunk.
                        T.call_extern("handle", "__threadfence")
                        T.sync_threads()
                        if T.get_thread_binding() == 0:
                            if trace:
                                IOTrace[physical, 1] = T.call_extern(
                                    "uint64", "nsa_global_timer"
                                )
                            T.atomic_add(
                                Flags[(physical - resident_blocks) // chunk_blocks],
                                1,
                                memory_order="acq_rel",
                            )
            else:
                row = bx - producers
                if producers > 0 and not pipelined:
                    if T.get_thread_binding() == 0:
                        for chunk in T.serial(chunks):
                            while T.atomic_load(
                                Flags[chunk], memory_order="acquire"
                            ) < T.min(
                                chunk_blocks,
                                blocks - resident_blocks - chunk * chunk_blocks,
                            ):
                                T.call_extern("handle", "__nanosleep", 64)
                    T.sync_threads()
                if trace:
                    if T.get_thread_binding() == 0:
                        ComputeTrace[row, 0, 0] = T.call_extern(
                            "uint64", "nsa_global_timer"
                        )
                for h, d in T.Parallel(groups, 64):
                    q[h, d] = Q[row, h, d].astype("float32") * 0.125
                T.fill(maximum, -T.infinity("float32"))
                T.clear(denominator)
                T.clear(acc)
                attend(
                    0,
                    row,
                    IDs,
                    KV,
                    Counts,
                    Starts,
                    Positions,
                    q,
                    k,
                    v,
                    score,
                    acc,
                    maximum,
                    previous,
                    denominator,
                    factor,
                    block_sum,
                    probability,
                )
                if trace:
                    if T.get_thread_binding() == 0:
                        ComputeTrace[row, 0, 1] = T.call_extern(
                            "uint64", "nsa_global_timer"
                        )
                if producers > 0:
                    if T.get_thread_binding() == 0:
                        T.atomic_add(Flags[chunks], 1, memory_order="acq_rel")
                        while (
                            T.atomic_load(Flags[chunks], memory_order="acquire") < rows
                        ):
                            T.call_extern("handle", "__nanosleep", 64)
                    T.sync_threads()
                for stage in T.serial(1, stages):
                    if producers > 0:
                        if T.get_thread_binding() == 0:
                            while T.atomic_load(
                                Flags[stage - 1], memory_order="acquire"
                            ) < T.min(
                                chunk_blocks,
                                blocks - resident_blocks - (stage - 1) * chunk_blocks,
                            ):
                                T.call_extern("handle", "__nanosleep", 64)
                        T.sync_threads()
                    if trace:
                        if T.get_thread_binding() == 0:
                            ComputeTrace[row, stage, 0] = T.call_extern(
                                "uint64", "nsa_global_timer"
                            )
                    attend(
                        stage,
                        row,
                        IDs,
                        KV,
                        Counts,
                        Starts,
                        Positions,
                        q,
                        k,
                        v,
                        score,
                        acc,
                        maximum,
                        previous,
                        denominator,
                        factor,
                        block_sum,
                        probability,
                    )
                    if trace:
                        if T.get_thread_binding() == 0:
                            ComputeTrace[row, stage, 1] = T.call_extern(
                                "uint64", "nsa_global_timer"
                            )
                for h, d in T.Parallel(groups, 64):
                    Out[row, h, d] = acc[h, d] / T.if_then_else(
                        denominator[h] > 0, denominator[h], 1.0
                    )
                if trace:
                    T.sync_threads()
                    if T.get_thread_binding() == 0:
                        ComputeTrace[row, stages - 1, 1] = T.call_extern(
                            "uint64", "nsa_global_timer"
                        )

    function = cast(Any, main).with_attr("global_symbol", "nsa_fused")
    config = pass_configs()
    if manual_sync:
        config[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True
    return tilelang.compile(
        function, target={"kind": "cuda", "arch": "sm_90a"}, pass_configs=config
    )
