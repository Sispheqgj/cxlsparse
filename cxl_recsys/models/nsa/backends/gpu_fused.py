"""N2 with GPU-resident block routing and dynamic missing extent.

IOIDs maps missing-block ranks to bounded logical-block slots. Missing is
read on device; no CPU count download is needed for local CXL/DRAM execution.

Cooperative selected NSA with dedicated SM host-IO CTAs.

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
def gpu_fused_kernel(
    rows,
    groups,
    blocks,
    block_size,
    width,
    chunk_blocks,
    io_ctas=16,
    pipelined=True,
    use_io=True,
    trace=False,
    manual_sync=False,
    full_tile_fastpath=False,
    address_table=False,
    skip_empty_stages=False,
    block_ready=False,
):
    import tilelang
    from tilelang import language as T

    stages = 1 + (blocks + chunk_blocks - 1) // chunk_blocks
    resident_blocks = 0
    if min(rows, blocks, width, stages, chunk_blocks) < 1:
        raise ValueError("Positive fused NSA geometry required")
    if groups not in (16, 32) or block_size not in (16, 32, 64):
        raise ValueError("N2 initially supports G16/32, BS16/32/64, D64")
    if not 0 <= resident_blocks <= blocks or io_ctas < 1:
        raise ValueError("Invalid resident extent or IO CTA count")
    if block_ready and not use_io:
        raise ValueError("Per-block readiness requires the publishing IO CTAs")
    chunks = stages - 1
    producers = min(io_ctas, blocks - resident_blocks) if use_io else 0
    if (
        use_io
        and chunks != (blocks - resident_blocks + chunk_blocks - 1) // chunk_blocks
    ):
        raise ValueError("Stage count must match missing chunks")
    grid = rows + producers
    flag_count = chunks + 1 + (blocks if block_ready else 0)

    @T.macro
    def prefetch(stage, row, slot, IDs, KV, k, v):
        # The entire padded block allocation exists in HBM. Copying its tail
        # never triggers external IO; the tail is zeroed before any arithmetic.
        physical = T.min(T.max(IDs[stage, row, slot], 0), blocks - 1)
        T.async_copy(KV[physical, :, :64], k[slot % 2, :, :])
        T.async_copy(KV[physical, :, 64:], v[slot % 2, :, :])

    @T.macro
    def accumulate(
        physical,
        row,
        slot,
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
        score_shared,
        probability,
    ):
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
                    T.call_extern(
                        "float32",
                        "nsa_exp2_approx",
                        (previous[h] - maximum[h]) * 1.4426950408889634,
                    ),
                )
            for h, n in T.Parallel(groups, block_size):
                score[h, n] = T.if_then_else(
                    score[h, n] == -T.infinity("float32"),
                    0.0,
                    T.call_extern(
                        "float32",
                        "nsa_exp2_approx",
                        (score[h, n] - maximum[h]) * 1.4426950408889634,
                    ),
                )
            # Match the original MMA-layout reduction tree for BS16.
            if block_size == 16:
                T.copy(score, score_shared)
                T.sync_threads()
                for h in T.Parallel(groups):
                    block_sum[h] = (
                        (
                            (score_shared[h, 0] + score_shared[h, 1])
                            + (score_shared[h, 8] + score_shared[h, 9])
                        )
                        + (
                            (score_shared[h, 4] + score_shared[h, 5])
                            + (score_shared[h, 12] + score_shared[h, 13])
                        )
                    ) + (
                        (
                            (score_shared[h, 2] + score_shared[h, 3])
                            + (score_shared[h, 10] + score_shared[h, 11])
                        )
                        + (
                            (score_shared[h, 6] + score_shared[h, 7])
                            + (score_shared[h, 14] + score_shared[h, 15])
                        )
                    )
            else:
                T.reduce_sum(score, block_sum, dim=1)
            for h in T.Parallel(groups):
                denominator[h] = denominator[h] * factor[h] + block_sum[h]
            T.copy(score, probability)
            if manual_sync:
                T.sync_threads()
            # Triton folds dot(P,V) into the corrected accumulator MMA.
            for h, d in T.Parallel(groups, 64):
                acc[h, d] *= factor[h]
            T.gemm(probability, v[slot % 2, :, :], acc)

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
        score_shared,
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
            accumulate(
                physical,
                row,
                slot,
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
                score_shared,
                probability,
            )
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
        Flags: T.Tensor((flag_count,), "int32"),
        IOIDs: T.Tensor((blocks,), "int32"),
        Missing: T.Tensor((1,), "int32"),
        Out: T.Tensor((rows, groups, 64), "bfloat16"),
        ComputeTrace: T.Tensor((rows, stages, 2), "uint64"),
        IOTrace: T.Tensor((blocks, 2), "uint64"),
    ):
        with T.Kernel(grid, threads=32) as bx:
            T.import_source(
                '__device__ __forceinline__ float nsa_exp2_approx(float x) { float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }\n__device__ __forceinline__ float nsa_divide_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }\n'
                + (
                    '__device__ __forceinline__ unsigned long long nsa_global_timer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }\n'
                    if trace
                    else ""
                )
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
            score_shared = T.alloc_shared((groups, block_size), "float32")
            if block_ready:
                done = T.alloc_shared((width,), "int32")
                picked = T.alloc_shared((1,), "int32")
                candidates = T.alloc_fragment((1, width), "int32")
                choice = T.alloc_fragment((1,), "int32")
            if producers > 0:
                if bx == 0:
                    for index in T.Parallel(flag_count):
                        Flags[index] = 0
                # This forces a cooperative launch. CUDA rejects an oversized
                # grid before execution instead of starving unscheduled CTAs.
                T.sync_grid()
            if bx < producers:
                for iteration in T.serial(T.ceildiv(Missing[0], max(1, producers))):
                    rank = iteration * producers + bx
                    if rank < Missing[0]:
                        physical = IOIDs[rank]
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
                            if block_ready:
                                # Fence above covers all token writers. Publish
                                # this physical block independently of its IO batch.
                                T.atomic_add(
                                    Flags[chunks + 1 + physical],
                                    1,
                                    memory_order="acq_rel",
                                )
                            if not block_ready or not pipelined:
                                T.atomic_add(
                                    Flags[rank // chunk_blocks],
                                    1,
                                    memory_order="acq_rel",
                                )
            else:
                row = bx - producers
                if producers > 0 and not pipelined:
                    if T.get_thread_binding() == 0:
                        for chunk in T.serial(T.ceildiv(Missing[0], chunk_blocks)):
                            while T.atomic_load(
                                Flags[chunk], memory_order="acquire"
                            ) < T.min(
                                chunk_blocks,
                                Missing[0] - chunk * chunk_blocks,
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
                    score_shared,
                    probability,
                )
                if trace:
                    if T.get_thread_binding() == 0:
                        ComputeTrace[row, 0, 1] = T.call_extern(
                            "uint64", "nsa_global_timer"
                        )
                if block_ready:
                    # Resident rows need no inter-consumer rendezvous: producer
                    # blocks are disjoint from stage 0. Only actual source writes
                    # publish readiness; stage 1 contains every missing occurrence.
                    T.clear(done)
                    T.sync_threads()
                    for occurrence in T.serial(Counts[1, row]):
                        # DIAGNOSTIC_BLOCK_WAIT_BEGIN
                        if T.get_thread_binding() == 0:
                            picked[0] = width
                        T.sync_threads()
                        while picked[0] == width:
                            # All lanes must finish testing the previous choice
                            # before a lane publishes the next one. A single
                            # warp still has independent thread scheduling.
                            T.sync_threads()
                            for index in T.Parallel(width):
                                candidates[0, index] = width
                                if index < Counts[1, row] and done[index] == 0:
                                    candidate = IDs[1, row, index]
                                    if (
                                        T.atomic_load(
                                            Flags[chunks + 1 + candidate],
                                            memory_order="acquire",
                                        )
                                        != 0
                                    ):
                                        candidates[0, index] = index
                            T.reduce_min(candidates, choice, dim=1, clear=True)
                            T.copy(choice, picked)
                            T.sync_threads()
                            if picked[0] == width:
                                T.call_extern("handle", "__nanosleep", 64)
                        physical = IDs[1, row, picked[0]]
                        # DIAGNOSTIC_BLOCK_READY
                        T.async_copy(KV[physical, :, :64], k[0, :, :])
                        T.async_copy(KV[physical, :, 64:], v[0, :, :])
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.sync_threads()
                        # DIAGNOSTIC_BLOCK_COMPUTE_BEGIN
                        accumulate(
                            physical,
                            row,
                            0,
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
                            score_shared,
                            probability,
                        )
                        T.sync_threads()
                        # DIAGNOSTIC_BLOCK_COMPUTE_END
                        if T.get_thread_binding() == 0:
                            done[picked[0]] = 1
                        T.sync_threads()
                else:
                    if producers > 0:
                        if T.get_thread_binding() == 0:
                            T.atomic_add(Flags[chunks], 1, memory_order="acq_rel")
                            while (
                                T.atomic_load(Flags[chunks], memory_order="acquire")
                                < rows
                            ):
                                T.call_extern("handle", "__nanosleep", 64)
                        T.sync_threads()
                    for stage in T.serial(1, T.ceildiv(Missing[0], chunk_blocks) + 1):
                        # No data from an empty stage is consumed by this CTA.
                        # Producers still publish every block, and the cooperative
                        # kernel joins before the caller can reuse this workspace.
                        if not skip_empty_stages or Counts[stage, row] > 0:
                            if producers > 0:
                                if T.get_thread_binding() == 0:
                                    while T.atomic_load(
                                        Flags[stage - 1], memory_order="acquire"
                                    ) < T.min(
                                        chunk_blocks,
                                        Missing[0] - (stage - 1) * chunk_blocks,
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
                                score_shared,
                                probability,
                            )
                            if trace:
                                if T.get_thread_binding() == 0:
                                    ComputeTrace[row, stage, 1] = T.call_extern(
                                        "uint64", "nsa_global_timer"
                                    )
                for h, d in T.Parallel(groups, 64):
                    Out[row, h, d] = T.call_extern(
                        "float32",
                        "nsa_divide_full",
                        acc[h, d],
                        T.if_then_else(denominator[h] > 0, denominator[h], 1.0),
                    )
                if trace:
                    T.sync_threads()
                    if T.get_thread_binding() == 0:
                        ComputeTrace[row, stages - 1, 1] = T.call_extern(
                            "uint64", "nsa_global_timer"
                        )

    function = cast(Any, main).with_attr("global_symbol", "nsa_gpu_fused")
    config = pass_configs()
    if manual_sync:
        config[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True
    return tilelang.compile(
        function, target={"kind": "cuda", "arch": "sm_90a"}, pass_configs=config
    )
