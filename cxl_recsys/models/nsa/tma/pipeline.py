"""Experimental SM90 NSA supply paths, with identical attention arithmetic.

``staged``: cooperative IO CTAs -> shared -> HBM -> attention shared.
``direct``: each attention CTA fills its own shared ring with bulk TMA.
``cluster``: one IO CTA serves the other CTAs' shared rings, with opportunistic
multicast for matching outstanding requests. No HBM KV handoff is used.

This is an explicit operator prototype, not a replacement for Worker dispatch.
Sources must be contiguous, aligned 16-token K64/V64 BF16 blocks, pinned for the
entire launch. The caller provides a stable, compact selected-prefix plan and
the globally deduplicated IOIDs. Arbitrary per-token mixed-tier pointers are NOT
accepted by this block-address interface. Tensor TMA uses two immutable maps
(HBM/CXL) with a common physical block index. Raw-address bulk is retained as
an explicit alternative; neither route uses CUDA copy engines.

The staged path also uses bulk TMA for the HBM handoff, completing destination
writes before publishing ready. Its IO and attention CTAs use different layouts
of the same shared allocation. Ready-aware prefetch never delays current MMA
waiting for a future block. The measured default is two stages / 24 IO CTAs on
H100 PCIe; explicit io_ctas overrides remain available.

Worker integration can set consumer_ctas to reuse a bounded cooperative grid
across query rows. With resident_ready=True, External is the missing-block mask:
referenced zero entries already exist in KV and IOIDs contains every nonzero
entry once. Unreferenced capacity needs no payload initialization.
The bulk-address adapter merges mixed-tier/tail blocks before this launch.
With ready_prepared=True, the caller must initialize Flags=(External==0) before
every launch or Graph replay, ordered after its payload preparation. The kernel
then skips only the initialization grid barrier; its launch stays cooperative.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from cxl_recsys.models.nsa.backends.tilelang import pass_configs


def encode_source_maps(resident, cxl_address=None):
    """Encode two aligned BF16 maps; source allocations must outlive every use.

    Each map covers [physical block, token, packed K/V]. The caller guarantees
    that external flags select the same physical layout as the address table.
    Host encoding and copying descriptors are outside the timed kernel.
    """
    import ctypes

    import torch

    # Use the installed driver's stable ABI. Importing newer cuda-python
    # bindings probes unavailable APIs on this host's 13.2 driver and pollutes
    # memcheck with unrelated cuGetProcAddress version-negotiation errors.
    driver = ctypes.CDLL("libcuda.so.1")
    encode = driver.cuTensorMapEncodeTiled
    encode.restype = ctypes.c_int
    encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    encoded = []
    for base in (resident.data_ptr(), cxl_address or resident.data_ptr()):
        storage = ctypes.create_string_buffer(128 + 63)
        descriptor = (ctypes.addressof(storage) + 63) & ~63
        # cuda.h: CU_TENSOR_MAP_DATA_TYPE_BFLOAT16=9, all four NONE enums=0.
        status = encode(
            descriptor,
            9,
            2,
            base,
            (ctypes.c_uint64 * 2)(128, resident.shape[0] * 16),
            (ctypes.c_uint64 * 1)(256),
            (ctypes.c_uint32 * 2)(128, 16),
            (ctypes.c_uint32 * 2)(1, 1),
            0,
            0,
            0,
            0,
        )
        if status != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed: {status}")
        encoded.append(list(ctypes.string_at(descriptor, 128)))
    return torch.tensor(encoded, dtype=torch.uint8, device=resident.device)


@lru_cache(maxsize=96)
def build_tma_kernel(
    rows,
    blocks,
    width=16,
    *,
    route="direct",
    stages=2,
    io_ctas=None,
    cluster_size=4,
    diagnostic=False,
    copy_kind="tensor",
    consumer_ctas=None,
    resident_ready=False,
    ready_window=1,
    ready_prepared=False,
):
    import tilelang
    from tilelang import language as T

    if io_ctas is None:
        io_ctas = 24 if route == "staged" else 32
    if route not in {"staged", "direct", "cluster", "cpasync"}:
        raise ValueError("Unknown NSA TMA route")
    if copy_kind not in {"tensor", "bulk"}:
        raise ValueError("copy_kind must be tensor or bulk")
    if min(rows, blocks, width) < 1 or stages not in (1, 2, 4, 8):
        raise ValueError("Positive geometry and 1/2/4/8 shared stages required")
    if cluster_size not in (2, 4, 8) or io_ctas < 1:
        raise ValueError("Invalid producer/cluster geometry")
    if consumer_ctas is not None and (consumer_ctas < 1 or route != "staged"):
        raise ValueError("Bounded consumers require positive CTAs and the staged route")
    if resident_ready and route != "staged":
        raise ValueError("Resident readiness is supported only by the staged route")
    if ready_prepared and (route != "staged" or not resident_ready):
        raise ValueError("Prepared readiness requires the staged resident-aware route")
    if ready_window not in (1, 2, 4) or (
        ready_window > 1 and (route != "staged" or stages != 2 or width > 32)
    ):
        raise ValueError("Ready windows 2/4 require staged two-slot TMA and width<=32")
    groups, bs = 16, 16
    consumers = cluster_size - 1
    if route == "cluster" and consumers * stages > 32:
        raise ValueError("The cluster scheduler admits at most 32 outstanding requests")
    clusters = (rows + consumers - 1) // consumers
    staged = route in {"staged", "cpasync"}
    producers = min(io_ctas, blocks) if staged else 0
    attention_ctas = min(rows, consumer_ctas) if consumer_ctas is not None else rows
    grid = clusters * cluster_size if route == "cluster" else attention_ctas + producers
    slots = stages
    source = Path(__file__).with_suffix(".cuh").read_text()
    if ready_window > 1:
        source += (
            "\n" + (Path(__file__).parents[1] / "experimental/ready.cuh").read_text()
        )

    @T.macro
    def issue(tiles, bars, slot, physical, Address, External, TensorMaps):
        if copy_kind == "tensor":
            T.call_extern(
                "void",
                "nsa_tensor_tma",
                T.address_of(tiles[slot, 0, 0]),
                T.address_of(TensorMaps[0, 0]),
                physical,
                External[physical],
                T.address_of(bars[slot]),
            )
        else:
            T.call_extern(
                "void",
                "nsa_tma",
                T.address_of(tiles[slot, 0, 0]),
                Address[physical],
                T.address_of(bars[slot]),
            )

    @T.macro
    def consume(
        row,
        physical,
        slot,
        Starts,
        Positions,
        q,
        tiles,
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
        # Clear both K and V before tensor-core arithmetic, including NaN tails.
        for n, d in T.Parallel(bs, 128):
            if Starts[physical] + n > Positions[row]:
                tiles[slot, n, d] = 0
        T.sync_threads()
        T.clear(score)
        T.gemm(q, tiles[slot, :, :64], score, transpose_B=True)
        for h, n in T.Parallel(groups, bs):
            score[h, n] = T.if_then_else(
                Starts[physical] + n <= Positions[row],
                score[h, n],
                -T.infinity("float32"),
            )
        T.copy(maximum, previous)
        T.reduce_max(score, maximum, dim=1, clear=False)
        for h in T.Parallel(groups):
            factor[h] = T.if_then_else(
                previous[h] == -T.infinity("float32"),
                0.0,
                T.call_extern(
                    "float32",
                    "nsa_tma_exp",
                    (previous[h] - maximum[h]) * 1.4426950408889634,
                ),
            )
        for h, n in T.Parallel(groups, bs):
            score[h, n] = T.if_then_else(
                score[h, n] == -T.infinity("float32"),
                0.0,
                T.call_extern(
                    "float32",
                    "nsa_tma_exp",
                    (score[h, n] - maximum[h]) * 1.4426950408889634,
                ),
            )
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
        for h in T.Parallel(groups):
            denominator[h] = denominator[h] * factor[h] + block_sum[h]
        T.copy(score, probability)
        for h, d in T.Parallel(groups, 64):
            acc[h, d] *= factor[h]
        T.gemm(probability, tiles[slot, :, 64:], acc)
        # Every writer orders its generic tail-zeroing before a later async
        # overwrite of this slot; then all readers release the slot together.
        T.call_extern("void", "nsa_proxy_fence")
        T.sync_threads()

    if ready_window > 1:
        from cxl_recsys.models.nsa.experimental.ready import make_ready_consumer

        consume_ready_row = make_ready_consumer(consume, window=ready_window)

    @T.macro
    def body(
        bx,
        Q,
        KV,
        IDs,
        Counts,
        Starts,
        Positions,
        Address,
        External,
        Flags,
        IOIDs,
        Missing,
        Requests,
        Done,
        TensorMaps,
        Counters,
        Out,
    ):
        T.import_source(source)
        tiles = T.alloc_shared((slots, bs, 128), "bfloat16")
        # This TileLang merge pass ignores buffer align=128 for opaque extern
        # copies. Pad every control allocation to 128 bytes, so every possible
        # allocation order preserves tensor TMA's destination alignment.
        bars = T.alloc_shared((16,), "uint64")
        if route == "cluster":
            requests = T.alloc_shared(
                (((consumers * (slots + 1) + 31) // 32) * 32,), "int32"
            )
            request_bars = T.alloc_shared(
                (((consumers * (slots + 1) + 15) // 16) * 16,), "uint64"
            )
        q = T.alloc_shared((groups, 64), "bfloat16")
        probability = T.alloc_shared((groups, bs), "bfloat16")
        score_shared = T.alloc_shared((groups, bs), "float32")
        score = T.alloc_fragment((groups, bs), "float32")
        acc = T.alloc_fragment((groups, 64), "float32")
        maximum = T.alloc_fragment((groups,), "float32")
        previous = T.alloc_fragment((groups,), "float32")
        denominator = T.alloc_fragment((groups,), "float32")
        factor = T.alloc_fragment((groups,), "float32")
        block_sum = T.alloc_fragment((groups,), "float32")
        if route == "staged":
            prefetched = T.alloc_local((1,), "int32")
        # A has disjoint IO/attention CTAs. IO accesses opaque linear bytes at
        # each slot's base; attention uses the inferred MMA layout. Thus both
        # roles reuse one allocation without interpreting each other's layout.
        # Direct/cluster TMA writes into the actual MMA payload, so those paths
        # (and the scalar cp.async control) must retain the row-major layout.
        if route != "staged":
            T.annotate_layout(
                {tiles: T.Layout((slots, bs, 128), lambda s, n, d: (s, n, d))}
            )
        if T.get_thread_binding() == 0:
            for s in T.serial(slots):
                T.call_extern(
                    "void",
                    "nsa_bar_init",
                    T.address_of(bars[s]),
                    2 if route == "cluster" else 1,
                )
            if route == "cluster":
                for s in T.serial(consumers * (slots + 1)):
                    T.call_extern("void", "nsa_bar_init", T.address_of(request_bars[s]))
                T.call_extern("void", "nsa_cluster_barrier_init_fence")
            T.call_extern("void", "nsa_proxy_fence")
        T.sync_threads()
        if staged and not ready_prepared:
            if bx == 0:
                for p in T.Parallel(blocks):
                    Flags[p] = (
                        T.if_then_else(External[p] == 0, 1, 0) if resident_ready else 0
                    )
            T.sync_grid()
        if route == "cluster":
            if bx % cluster_size == 0:
                for p in T.Parallel(consumers * (slots + 1)):
                    requests[p] = 0
                if T.get_thread_binding() == 0:
                    for j in T.serial(3):
                        Counters[bx // cluster_size, j] = 0
            T.call_extern("void", "nsa_cluster_sync")
        if staged and bx < producers:
            for initial in T.serial(slots):
                first_rank = initial * producers + bx
                if first_rank < Missing[0]:
                    if route == "cpasync":
                        T.call_extern(
                            "void",
                            "nsa_cp_async_block",
                            T.address_of(tiles[initial, 0, 0]),
                            Address[IOIDs[first_rank]],
                        )
                    elif T.get_thread_binding() == 0:
                        issue(
                            tiles,
                            bars,
                            initial,
                            IOIDs[first_rank],
                            Address,
                            External,
                            TensorMaps,
                        )
            for iteration in T.serial(T.ceildiv(Missing[0], producers)):
                rank = iteration * producers + bx
                slot = iteration % slots
                if rank < Missing[0]:
                    physical = IOIDs[rank]
                    if route != "cpasync":
                        T.call_extern(
                            "void",
                            "nsa_wait",
                            T.address_of(bars[slot]),
                            (iteration // slots) % 2,
                        )
                    else:
                        T.call_extern(
                            "void",
                            f"nsa_cp_async_wait<{slots}>",
                            T.min(slots - 1, (Missing[0] - 1 - rank) // producers),
                        )
                    T.sync_threads()
                    if route == "staged":
                        if T.get_thread_binding() == 0:
                            T.call_extern(
                                "void",
                                "nsa_tma_store",
                                T.address_of(KV[physical, 0, 0]),
                                T.address_of(tiles[slot, 0, 0]),
                            )
                    else:
                        T.copy(tiles[slot, :, :], KV[physical, :, :])
                        T.call_extern("void", "__threadfence")
                    T.sync_threads()
                    if T.get_thread_binding() == 0:
                        T.call_extern(
                            "void", "nsa_release", T.address_of(Flags[physical]), 1
                        )
                    next_rank = rank + slots * producers
                    if next_rank < Missing[0]:
                        if route == "cpasync":
                            T.call_extern(
                                "void",
                                "nsa_cp_async_block",
                                T.address_of(tiles[slot, 0, 0]),
                                Address[IOIDs[next_rank]],
                            )
                        elif T.get_thread_binding() == 0:
                            issue(
                                tiles,
                                bars,
                                slot,
                                IOIDs[next_rank],
                                Address,
                                External,
                                TensorMaps,
                            )
        elif route == "cluster" and bx % cluster_size == 0:
            T.call_extern(
                "void",
                f"nsa_cluster_io<{consumers}, {slots}, {'true' if diagnostic else 'false'}, {'true' if copy_kind == 'tensor' else 'false'}>",
                T.address_of(tiles[0, 0, 0]),
                T.address_of(bars[0]),
                T.address_of(Address[0]),
                T.address_of(External[0]),
                T.address_of(TensorMaps[0, 0]),
                T.address_of(requests[0]),
                T.address_of(request_bars[0]),
                T.min(consumers, rows - (bx // cluster_size) * consumers),
                T.address_of(Counters[bx // cluster_size, 0]),
            )
        else:
            for row_iteration in T.serial(T.ceildiv(rows, attention_ctas)):
                row = (
                    bx // cluster_size * consumers + bx % cluster_size - 1
                    if route == "cluster"
                    else bx - producers + row_iteration * attention_ctas
                )
                if row < rows:
                    for h, d in T.Parallel(groups, 64):
                        q[h, d] = Q[row, h, d].astype("float32") * 0.125
                    T.fill(maximum, -T.infinity("float32"))
                    T.clear(denominator)
                    T.clear(acc)
                    if ready_window > 1:
                        consume_ready_row(
                            row,
                            KV,
                            IDs,
                            Counts,
                            Starts,
                            Positions,
                            Flags,
                            q,
                            tiles,
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
                    else:
                        if staged:
                            if Counts[row] > 0:
                                T.call_extern(
                                    "void",
                                    "nsa_wait_ready",
                                    T.address_of(Flags[IDs[row, 0]]),
                                )
                                T.async_copy(KV[IDs[row, 0], :, :], tiles[0, :, :])
                                T.ptx_wait_group(0)
                                T.sync_threads()
                        # All slots are issued before the first wait. Shared slots stay
                        # owned until their consumer finishes PV, not merely until IO.
                        if T.get_thread_binding() == 0:
                            for s in T.serial(T.min(slots, Counts[row])):
                                physical = IDs[row, s]
                                if route == "cluster":
                                    T.call_extern(
                                        "void", "nsa_expect", T.address_of(bars[s])
                                    )
                                    T.call_extern(
                                        "void",
                                        "nsa_request",
                                        T.address_of(
                                            requests[
                                                (bx % cluster_size - 1) * slots + s
                                            ]
                                        ),
                                        physical + 1,
                                        T.address_of(
                                            request_bars[
                                                (bx % cluster_size - 1) * slots + s
                                            ]
                                        ),
                                    )
                                elif route == "direct":
                                    issue(
                                        tiles,
                                        bars,
                                        s,
                                        physical,
                                        Address,
                                        External,
                                        TensorMaps,
                                    )
                        for index in T.serial(Counts[row]):
                            slot = index % slots
                            physical = IDs[row, index]
                            if staged:
                                if slots > 1:
                                    if index + 1 < Counts[row]:
                                        if route == "staged":
                                            prefetched[0] = T.call_extern(
                                                "int32",
                                                "nsa_ready_warp",
                                                T.access_ptr(
                                                    Flags,
                                                    "r",
                                                    offset=IDs[row, index + 1],
                                                ),
                                            )
                                            if prefetched[0] != 0:
                                                T.async_copy(
                                                    KV[IDs[row, index + 1], :, :],
                                                    tiles[(index + 1) % slots, :, :],
                                                )
                                        else:
                                            T.call_extern(
                                                "void",
                                                "nsa_wait_ready",
                                                T.address_of(
                                                    Flags[IDs[row, index + 1]]
                                                ),
                                            )
                                            T.async_copy(
                                                KV[IDs[row, index + 1], :, :],
                                                tiles[(index + 1) % slots, :, :],
                                            )
                            elif route == "cluster":
                                T.call_extern(
                                    "void",
                                    "nsa_wait_cluster",
                                    T.address_of(bars[slot]),
                                    (index // slots) % 2,
                                )
                            else:
                                T.call_extern(
                                    "void",
                                    "nsa_wait",
                                    T.address_of(bars[slot]),
                                    (index // slots) % 2,
                                )
                            T.sync_threads()
                            consume(
                                row,
                                physical,
                                slot,
                                Starts,
                                Positions,
                                q,
                                tiles,
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
                            if staged:
                                # Compute the available block before waiting for the
                                # next. Prefetch overlaps MMA only if all lanes acquired
                                # ready; otherwise issue it here, once, then drain it.
                                if route == "staged" and slots > 1:
                                    if index + 1 < Counts[row]:
                                        if prefetched[0] == 0:
                                            T.call_extern(
                                                "void",
                                                "nsa_wait_ready",
                                                T.address_of(
                                                    Flags[IDs[row, index + 1]]
                                                ),
                                            )
                                            T.async_copy(
                                                KV[IDs[row, index + 1], :, :],
                                                tiles[(index + 1) % slots, :, :],
                                            )
                                if slots == 1:
                                    if index + 1 < Counts[row]:
                                        T.call_extern(
                                            "void",
                                            "nsa_wait_ready",
                                            T.address_of(Flags[IDs[row, index + 1]]),
                                        )
                                        T.async_copy(
                                            KV[IDs[row, index + 1], :, :],
                                            tiles[0, :, :],
                                        )
                                T.ptx_wait_group(0)
                                T.sync_threads()
                            if index + slots < Counts[row]:
                                if T.get_thread_binding() == 0:
                                    next_physical = IDs[row, index + slots]
                                    T.call_extern("void", "nsa_proxy_fence")
                                    if route == "cluster":
                                        T.call_extern(
                                            "void",
                                            "nsa_expect",
                                            T.address_of(bars[slot]),
                                        )
                                        T.call_extern(
                                            "void",
                                            "nsa_request",
                                            T.address_of(
                                                requests[
                                                    (bx % cluster_size - 1) * slots
                                                    + slot
                                                ]
                                            ),
                                            next_physical + 1,
                                            T.address_of(
                                                request_bars[
                                                    (bx % cluster_size - 1) * slots
                                                    + slot
                                                ]
                                            ),
                                        )
                                    elif route == "direct":
                                        issue(
                                            tiles,
                                            bars,
                                            slot,
                                            next_physical,
                                            Address,
                                            External,
                                            TensorMaps,
                                        )
                    for h, d in T.Parallel(groups, 64):
                        Out[row, h, d] = T.call_extern(
                            "float32",
                            "nsa_tma_div",
                            acc[h, d],
                            T.if_then_else(denominator[h] > 0, denominator[h], 1.0),
                        )
                    if route == "cluster":
                        T.sync_threads()
                        if T.get_thread_binding() == 0:
                            T.call_extern(
                                "void",
                                "nsa_arrive_remote",
                                T.address_of(
                                    request_bars[
                                        consumers * slots + bx % cluster_size - 1
                                    ]
                                ),
                            )
        if route == "cluster":
            T.call_extern("void", "nsa_cluster_sync")

    @T.prim_func
    def main(
        Q: T.Tensor((rows, groups, 64), "bfloat16"),
        KV: T.Tensor((blocks, bs, 128), "bfloat16"),
        IDs: T.Tensor((rows, width), "int32"),
        Counts: T.Tensor((rows,), "int32"),
        Starts: T.Tensor((blocks,), "int32"),
        Positions: T.Tensor((rows,), "int32"),
        Address: T.Tensor((blocks,), "int64"),
        External: T.Tensor((blocks,), "int32"),
        Flags: T.Tensor((blocks,), "int32"),
        IOIDs: T.Tensor((blocks,), "int32"),
        Missing: T.Tensor((1,), "int32"),
        Requests: T.Tensor((clusters, consumers, slots), "int32"),
        Done: T.Tensor((clusters,), "int32"),
        TensorMaps: T.Tensor((2, 128), "uint8"),
        Counters: T.Tensor((clusters, 3), "int64"),
        Out: T.Tensor((rows, groups, 64), "bfloat16"),
    ):
        if route == "cluster":
            with T.ClusterKernel(grid, cluster_dims=cluster_size, threads=32) as bx:
                body(
                    bx,
                    Q,
                    KV,
                    IDs,
                    Counts,
                    Starts,
                    Positions,
                    Address,
                    External,
                    Flags,
                    IOIDs,
                    Missing,
                    Requests,
                    Done,
                    TensorMaps,
                    Counters,
                    Out,
                )
        else:
            with T.Kernel(grid, threads=32) as bx:
                body(
                    bx,
                    Q,
                    KV,
                    IDs,
                    Counts,
                    Starts,
                    Positions,
                    Address,
                    External,
                    Flags,
                    IOIDs,
                    Missing,
                    Requests,
                    Done,
                    TensorMaps,
                    Counters,
                    Out,
                )

    function = cast(Any, main).with_attr("global_symbol", f"nsa_tma_{route}")
    if ready_prepared:
        from cxl_recsys.models.nsa.tma.launch import cooperative_launch

        # Keep producers co-resident with polling consumers even when the
        # preceding preparation kernel replaces the initialization grid barrier.
        function = cooperative_launch(function)
    return tilelang.compile(
        function, target={"kind": "cuda", "arch": "sm_90a"}, pass_configs=pass_configs()
    )
