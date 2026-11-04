"""Opt-in, all-HBM selected attention sharing KV across consecutive queries.

Each CTA belongs to one user/layer/KV-head and a 2/4/8-query tile. The grouping
pass deduplicates *physical loads*, not each row's logical occurrences. A block
is copied to shared memory once, then a Q-tile x GQA-group GEMM updates separate
online softmax states. Transport readiness and buffer lifetime belong to the
caller. This module does not issue external IO or select a default route.

Grouping changes block processing order. BF16 Q scaling and probabilities match
the existing TileLang consumer; comparisons retain its rtol=atol=0.03 threshold.
"""

# pyright: reportInvalidTypeForm=false, reportAttributeAccessIssue=false

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch
import triton
import triton.language as tl

from cxl_recsys.models.nsa.backends.tilelang import pass_configs


@triton.jit
def _union(
    IDs,
    Counts,
    Union,
    Lengths,
    Q: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr,
    QT: tl.constexpr,
    N: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tile % HEADS
    first = tile // HEADS * QT
    offset = tl.arange(0, N)
    query = first + offset // WIDTH
    slot = offset % WIDTH
    valid = (offset < QT * WIDTH) & (query < Q)
    count = tl.load(Counts + query * HEADS + head, valid, other=0)
    physical = tl.load(
        IDs + (query * HEADS + head) * WIDTH + slot,
        valid & (slot < count),
        other=-1,
    )
    physical = tl.where((physical >= 0) & (physical < BLOCKS), physical, 2147483647)
    physical = tl.sort(physical, descending=False)
    previous = tl.gather(physical, tl.maximum(offset - 1, 0), axis=0)
    unique = (physical < 2147483647) & ((offset == 0) | (physical != previous))
    rank = tl.cumsum(unique.to(tl.int32)) - 1
    tl.store(Union + tile * QT * WIDTH + rank, physical, unique)
    tl.store(Lengths + tile, tl.sum(unique.to(tl.int32)))


@triton.jit
def _membership(
    IDs,
    Counts,
    Union,
    Lengths,
    Occurrences,
    Q: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    QT: tl.constexpr,
    W: tl.constexpr,
):
    tile, block_slot = tl.program_id(0), tl.program_id(1)
    head = tile % HEADS
    first = tile // HEADS * QT
    length = tl.load(Lengths + tile)
    if block_slot < length:
        physical = tl.load(Union + tile * QT * WIDTH + block_slot)
        q = first + tl.arange(0, QT)
        s = tl.arange(0, W)
        counts = tl.load(Counts + q * HEADS + head, q < Q, other=0)
        refs = tl.load(
            IDs + (q[:, None] * HEADS + head) * WIDTH + s[None, :],
            (q[:, None] < Q) & (s[None, :] < WIDTH) & (s[None, :] < counts[:, None]),
            other=-1,
        )
        occurrences = tl.sum((refs == physical).to(tl.int32), axis=1)
        tl.store(
            Occurrences + (tile * QT * WIDTH + block_slot) * QT + tl.arange(0, QT),
            occurrences,
        )


@dataclass
class QueryTileWorkspace:
    """Exclusive scratch lease; caller must order reuse on its CUDA stream.

    There is no global cache keyed only by shape. Allocate outside graph capture
    and do not share this workspace across overlapping users/layers/streams.
    """

    queries: int
    heads: int
    groups: int
    physical_blocks: int
    block_size: int
    width: int
    tile_queries: int
    pipelined: bool
    union: torch.Tensor
    lengths: torch.Tensor
    occurrences: torch.Tensor
    output: torch.Tensor

    @property
    def bytes(self):
        return sum(
            x.numel() * x.element_size()
            for x in (self.union, self.lengths, self.occurrences, self.output)
        )


def allocate_workspace(query, kv, ids, tile_queries=2, *, pipelined=False):
    """Allocate outside capture; prefetch stays opt-in until GPU calibration."""
    if type(pipelined) is not bool:
        raise TypeError("pipelined must be boolean")
    if query.ndim != 4 or kv.ndim != 3 or ids.ndim != 2:
        raise ValueError("Expected Q[Q,Hkv,G,64], KV[physical,BS,128], IDs[Q*Hkv,W]")
    queries, heads, groups, dim = query.shape
    physical, bs, record = kv.shape
    if (
        tile_queries not in (2, 4, 8)
        or dim != 64
        or record != 128
        or groups not in (16, 32)
        or bs not in (16, 32, 64)
        or min(queries, heads, physical, ids.shape[1]) < 1
        or ids.shape[0] != queries * heads
    ):
        raise ValueError("Unsupported query-tile geometry")
    tiles = triton.cdiv(queries, tile_queries) * heads
    capacity = tile_queries * ids.shape[1]
    return QueryTileWorkspace(
        queries,
        heads,
        groups,
        physical,
        bs,
        ids.shape[1],
        tile_queries,
        pipelined,
        torch.empty((tiles, capacity), dtype=torch.int32, device=query.device),
        torch.empty((tiles,), dtype=torch.int32, device=query.device),
        torch.empty(
            (tiles, capacity, tile_queries), dtype=torch.int32, device=query.device
        ),
        torch.empty_like(query),
    )


def group_queries(ids, counts, workspace):
    """GPU-only grouping; padded counts are zero and no dynamic length is read by CPU."""
    w = workspace
    if ids.shape != (w.queries * w.heads, w.width) or counts.shape != (len(ids),):
        raise ValueError("Row references do not match the workspace")
    if any(
        x.dtype != torch.int32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.device != w.output.device
        for x in (ids, counts)
    ):
        raise ValueError(
            "Grouping requires contiguous int32 tensors on the workspace GPU"
        )
    _union[(len(w.lengths),)](
        ids,
        counts,
        w.union,
        w.lengths,
        w.queries,
        w.heads,
        w.width,
        w.physical_blocks,
        w.tile_queries,
        triton.next_power_of_2(w.tile_queries * w.width),
    )
    _membership[(len(w.lengths), w.union.shape[1])](
        ids,
        counts,
        w.union,
        w.lengths,
        w.occurrences,
        w.queries,
        w.heads,
        w.width,
        w.tile_queries,
        triton.next_power_of_2(w.width),
    )


def make_prim_func(
    queries, heads, groups, physical, block_size, width, tile_queries, pipelined=False
):
    """Construct TileLang IR without initializing CUDA (for offline validation)."""
    from tilelang import language as T

    if (
        tile_queries not in (2, 4, 8)
        or groups not in (16, 32)
        or block_size not in (16, 32, 64)
        or min(queries, heads, physical, width) < 1
    ):
        raise ValueError("Unsupported query-tile geometry")
    if type(pipelined) is not bool:
        raise TypeError("pipelined must be boolean")
    if pipelined:
        return _make_pipelined_prim_func(
            queries, heads, groups, physical, block_size, width, tile_queries
        )
    tiles = triton.cdiv(queries, tile_queries) * heads
    capacity = tile_queries * width
    matrix_rows = tile_queries * groups
    threads = min(128, tile_queries * 32)

    @T.prim_func
    def main(
        Q: T.Tensor((queries * heads, groups, 64), "bfloat16"),
        KV: T.Tensor((physical, block_size, 128), "bfloat16"),
        Union: T.Tensor((tiles, capacity), "int32"),
        Lengths: T.Tensor((tiles,), "int32"),
        Occurrences: T.Tensor((tiles, capacity, tile_queries), "int32"),
        Starts: T.Tensor((physical,), "int32"),
        Positions: T.Tensor((queries * heads,), "int32"),
        Out: T.Tensor((queries * heads, groups, 64), "bfloat16"),
    ):
        with T.Kernel(tiles, threads=threads) as tile:
            q = T.alloc_shared((matrix_rows, 64), "bfloat16")
            k = T.alloc_shared((block_size, 64), "bfloat16")
            v = T.alloc_shared((block_size, 64), "bfloat16")
            probability = T.alloc_shared((matrix_rows, block_size), "bfloat16")
            raw_score = T.alloc_fragment((matrix_rows, block_size), "float32")
            score = T.alloc_fragment((matrix_rows, block_size), "float32")
            acc = T.alloc_fragment((matrix_rows, 64), "float32")
            maximum = T.alloc_fragment((matrix_rows,), "float32")
            previous = T.alloc_fragment((matrix_rows,), "float32")
            denominator = T.alloc_fragment((matrix_rows,), "float32")
            factor = T.alloc_fragment((matrix_rows,), "float32")
            block_sum = T.alloc_fragment((matrix_rows,), "float32")
            last_position = T.alloc_shared((1,), "int32")
            repeats = T.alloc_shared((1,), "int32")
            head = tile % heads
            first = tile // heads * tile_queries
            for m, d in T.Parallel(matrix_rows, 64):
                query = first + m // groups
                q[m, d] = T.if_then_else(
                    query < queries,
                    Q[T.min(query, queries - 1) * heads + head, m % groups, d].astype(
                        "float32"
                    )
                    * 0.125,
                    0.0,
                )
            T.fill(maximum, -T.infinity("float32"))
            T.clear(denominator)
            T.clear(acc)
            T.sync_threads()
            for slot in T.serial(Lengths[tile]):
                block = Union[tile, slot]
                if T.get_thread_binding() == 0:
                    last_position[0] = -1
                    repeats[0] = 0
                    for local in T.serial(tile_queries):
                        repeats[0] = T.max(repeats[0], Occurrences[tile, slot, local])
                        if (first + local < queries) & (
                            Occurrences[tile, slot, local] > 0
                        ):
                            last_position[0] = T.max(
                                last_position[0],
                                Positions[(first + local) * heads + head],
                            )
                # One shared-memory load for the whole query tile, including
                # repeated occurrences. Invalid global tails are sanitized so
                # masked zero probabilities cannot multiply uninitialized NaNs.
                T.async_copy(KV[block, :, :64], k)
                T.async_copy(KV[block, :, 64:], v)
                T.ptx_wait_group(0)
                T.sync_threads()
                for token, d in T.Parallel(block_size, 64):
                    if Starts[block] + token > last_position[0]:
                        k[token, d] = 0
                        v[token, d] = 0
                T.sync_threads()
                T.clear(raw_score)
                T.gemm(q, k, raw_score, transpose_B=True)
                for occurrence in T.serial(repeats[0]):
                    for m, token in T.Parallel(matrix_rows, block_size):
                        query = first + m // groups
                        row = T.min(query, queries - 1) * heads + head
                        allowed = (
                            (query < queries)
                            & (Occurrences[tile, slot, m // groups] > occurrence)
                            & (Starts[block] + token <= Positions[row])
                        )
                        score[m, token] = T.if_then_else(
                            allowed, raw_score[m, token], -T.infinity("float32")
                        )
                    T.copy(maximum, previous)
                    T.reduce_max(score, maximum, dim=1, clear=False)
                    for m in T.Parallel(matrix_rows):
                        factor[m] = T.if_then_else(
                            previous[m] == -T.infinity("float32"),
                            0.0,
                            T.exp2((previous[m] - maximum[m]) * 1.4426950408889634),
                        )
                    for m, token in T.Parallel(matrix_rows, block_size):
                        score[m, token] = T.if_then_else(
                            score[m, token] == -T.infinity("float32"),
                            0.0,
                            T.exp2((score[m, token] - maximum[m]) * 1.4426950408889634),
                        )
                    T.reduce_sum(score, block_sum, dim=1)
                    for m in T.Parallel(matrix_rows):
                        denominator[m] = denominator[m] * factor[m] + block_sum[m]
                    for m, d in T.Parallel(matrix_rows, 64):
                        acc[m, d] *= factor[m]
                    T.copy(score, probability)
                    T.sync_threads()
                    T.gemm(probability, v, acc)
                    T.sync_threads()
            for m, d in T.Parallel(matrix_rows, 64):
                if first + m // groups < queries:
                    Out[(first + m // groups) * heads + head, m % groups, d] = acc[
                        m, d
                    ] / T.if_then_else(denominator[m] > 0, denominator[m], 1.0)

    return cast(Any, main).with_attr("global_symbol", "nsa_query_tile_shared_kv")


def _make_pipelined_prim_func(
    queries, heads, groups, physical, block_size, width, tile_queries
):
    """Explicit two-buffer prefetch; preserve v1 IR when the option is off.

    The pipeline moves HBM data only. It neither extends transport lifetimes nor
    relaxes the caller's IO readiness dependency. Grouping/order/numerics match
    v1, including duplicate logical occurrences and per-block member masks.
    """
    from tilelang import language as T

    tiles = triton.cdiv(queries, tile_queries) * heads
    capacity = tile_queries * width
    matrix_rows = tile_queries * groups
    threads = min(128, tile_queries * 32)

    @T.macro
    def prefetch(tile, slot, Union, KV, k, v):
        block = Union[tile, slot]
        T.async_copy(KV[block, :, :64], k[slot % 2, :, :])
        T.async_copy(KV[block, :, 64:], v[slot % 2, :, :])

    @T.prim_func
    def main(
        Q: T.Tensor((queries * heads, groups, 64), "bfloat16"),
        KV: T.Tensor((physical, block_size, 128), "bfloat16"),
        Union: T.Tensor((tiles, capacity), "int32"),
        Lengths: T.Tensor((tiles,), "int32"),
        Occurrences: T.Tensor((tiles, capacity, tile_queries), "int32"),
        Starts: T.Tensor((physical,), "int32"),
        Positions: T.Tensor((queries * heads,), "int32"),
        Out: T.Tensor((queries * heads, groups, 64), "bfloat16"),
    ):
        with T.Kernel(tiles, threads=threads) as tile:
            q = T.alloc_shared((matrix_rows, 64), "bfloat16")
            k = T.alloc_shared((2, block_size, 64), "bfloat16")
            v = T.alloc_shared((2, block_size, 64), "bfloat16")
            probability = T.alloc_shared((matrix_rows, block_size), "bfloat16")
            raw_score = T.alloc_fragment((matrix_rows, block_size), "float32")
            score = T.alloc_fragment((matrix_rows, block_size), "float32")
            acc = T.alloc_fragment((matrix_rows, 64), "float32")
            maximum = T.alloc_fragment((matrix_rows,), "float32")
            previous = T.alloc_fragment((matrix_rows,), "float32")
            denominator = T.alloc_fragment((matrix_rows,), "float32")
            factor = T.alloc_fragment((matrix_rows,), "float32")
            block_sum = T.alloc_fragment((matrix_rows,), "float32")
            last_position = T.alloc_shared((1,), "int32")
            repeats = T.alloc_shared((1,), "int32")
            head = tile % heads
            first = tile // heads * tile_queries
            for m, d in T.Parallel(matrix_rows, 64):
                query = first + m // groups
                q[m, d] = T.if_then_else(
                    query < queries,
                    Q[T.min(query, queries - 1) * heads + head, m % groups, d].astype(
                        "float32"
                    )
                    * 0.125,
                    0.0,
                )
            T.fill(maximum, -T.infinity("float32"))
            T.clear(denominator)
            T.clear(acc)
            if Lengths[tile] > 0:
                prefetch(tile, 0, Union, KV, k, v)
                T.ptx_wait_group(0)
            T.sync_threads()
            for slot in T.serial(Lengths[tile]):
                # The opposite buffer is no longer in use after the previous
                # iteration's final barrier. Copy it while computing this one.
                if slot + 1 < Lengths[tile]:
                    prefetch(tile, slot + 1, Union, KV, k, v)
                block = Union[tile, slot]
                if T.get_thread_binding() == 0:
                    last_position[0] = -1
                    repeats[0] = 0
                    for local in T.serial(tile_queries):
                        repeats[0] = T.max(repeats[0], Occurrences[tile, slot, local])
                        if (first + local < queries) & (
                            Occurrences[tile, slot, local] > 0
                        ):
                            last_position[0] = T.max(
                                last_position[0],
                                Positions[(first + local) * heads + head],
                            )
                # Only positions of this block's member queries can establish
                # a valid KV tail. A later nonmember must not expose NaN V to
                # zero probabilities. Broadcast metadata before sanitization.
                T.sync_threads()
                for token, d in T.Parallel(block_size, 64):
                    if Starts[block] + token > last_position[0]:
                        k[slot % 2, token, d] = 0
                        v[slot % 2, token, d] = 0
                T.sync_threads()
                T.clear(raw_score)
                T.gemm(q, k[slot % 2, :, :], raw_score, transpose_B=True)
                for occurrence in T.serial(repeats[0]):
                    for m, token in T.Parallel(matrix_rows, block_size):
                        query = first + m // groups
                        row = T.min(query, queries - 1) * heads + head
                        allowed = (
                            (query < queries)
                            & (Occurrences[tile, slot, m // groups] > occurrence)
                            & (Starts[block] + token <= Positions[row])
                        )
                        score[m, token] = T.if_then_else(
                            allowed, raw_score[m, token], -T.infinity("float32")
                        )
                    T.copy(maximum, previous)
                    T.reduce_max(score, maximum, dim=1, clear=False)
                    for m in T.Parallel(matrix_rows):
                        factor[m] = T.if_then_else(
                            previous[m] == -T.infinity("float32"),
                            0.0,
                            T.exp2((previous[m] - maximum[m]) * 1.4426950408889634),
                        )
                    for m, token in T.Parallel(matrix_rows, block_size):
                        score[m, token] = T.if_then_else(
                            score[m, token] == -T.infinity("float32"),
                            0.0,
                            T.exp2((score[m, token] - maximum[m]) * 1.4426950408889634),
                        )
                    T.reduce_sum(score, block_sum, dim=1)
                    for m in T.Parallel(matrix_rows):
                        denominator[m] = denominator[m] * factor[m] + block_sum[m]
                    for m, d in T.Parallel(matrix_rows, 64):
                        acc[m, d] *= factor[m]
                    T.copy(score, probability)
                    T.sync_threads()
                    T.gemm(probability, v[slot % 2, :, :], acc)
                    T.sync_threads()
                # Drain this CTA's next-block copies before their buffer is
                # consumed. The barrier also protects the current buffer from
                # reuse even when the logical block has no occurrences.
                T.ptx_wait_group(0)
                T.sync_threads()
            for m, d in T.Parallel(matrix_rows, 64):
                if first + m // groups < queries:
                    Out[(first + m // groups) * heads + head, m % groups, d] = acc[
                        m, d
                    ] / T.if_then_else(denominator[m] > 0, denominator[m], 1.0)

    return cast(Any, main).with_attr(
        "global_symbol", "nsa_query_tile_shared_kv_pipeline"
    )


@lru_cache(maxsize=64)
def build_kernel(
    queries, heads, groups, physical, block_size, width, tile_queries, pipelined=False
):
    import tilelang

    config = pass_configs()
    # All shared-memory dependencies have explicit CTA barriers. The automatic
    # pass cannot analyze the uniform loop bound broadcast through repeats[0].
    config[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True
    return tilelang.compile(
        make_prim_func(
            queries, heads, groups, physical, block_size, width, tile_queries, pipelined
        ),
        target={"kind": "cuda", "arch": "sm_90a"},
        pass_configs=config,
    )


def consume_grouped(query, kv, starts, positions, workspace):
    """Consume a prepared group; output is borrowed until workspace reuse."""
    w = workspace
    expected = (
        (w.queries, w.heads, w.groups, 64),
        (w.physical_blocks, w.block_size, 128),
        (w.physical_blocks,),
        (w.queries * w.heads,),
    )
    for value, shape, dtype in zip(
        (query, kv, starts, positions),
        expected,
        (torch.bfloat16, torch.bfloat16, torch.int32, torch.int32),
        strict=True,
    ):
        if (
            value.shape != shape
            or value.dtype != dtype
            or not value.is_contiguous()
            or not value.is_cuda
            or value.device != w.output.device
        ):
            raise ValueError("Consumer inputs must match its contiguous GPU workspace")
    build_kernel(
        w.queries,
        w.heads,
        w.groups,
        w.physical_blocks,
        w.block_size,
        w.width,
        w.tile_queries,
        w.pipelined,
    )(
        query.view(-1, w.groups, 64),
        kv,
        w.union,
        w.lengths,
        w.occurrences,
        starts,
        positions,
        w.output.view(-1, w.groups, 64),
    )
    return w.output


def query_tile_attention(
    query,
    kv,
    ids,
    starts,
    positions,
    counts,
    *,
    tile_queries=2,
    workspace=None,
    pipelined=False,
):
    """All-HBM opt-in entry. Includes grouping; does not transfer or cache KV."""
    if workspace is None:
        workspace = allocate_workspace(
            query, kv, ids, tile_queries, pipelined=pipelined
        )
    if workspace.tile_queries != tile_queries:
        raise ValueError("Tile size differs from the supplied workspace")
    if workspace.pipelined != pipelined:
        raise ValueError("Pipeline mode differs from the supplied workspace")
    group_queries(ids, counts, workspace)
    return consume_grouped(query, kv, starts, positions, workspace)


def _cpu_rows(ids, counts, heads):
    if (
        ids.device.type != "cpu"
        or counts.device.type != "cpu"
        or ids.ndim != 2
        or counts.shape != (len(ids),)
        or heads < 1
        or len(ids) % heads
        or bool(((counts < 0) | (counts > ids.shape[1])).any())
    ):
        raise ValueError("Expected CPU rows with valid counts and Q*heads layout")
    return [
        [int(x) for x in row[: int(count)].tolist() if x >= 0]
        for row, count in zip(ids, counts, strict=True)
    ]


def reuse_statistics(ids, counts, heads, tile_queries):
    """Logical opportunity only; this is not measured L2/HBM traffic or speedup."""
    rows = _cpu_rows(ids, counts, heads)
    if tile_queries < 1:
        raise ValueError("Tile size must be positive")
    result = []
    for first in range(0, len(rows) // heads, tile_queries):
        for head in range(heads):
            chosen = [
                rows[q * heads + head]
                for q in range(first, min(first + tile_queries, len(rows) // heads))
            ]
            unique = [set(row) for row in chosen]
            union = set().union(*unique)
            references = sum(map(len, chosen))
            distinct_references = sum(map(len, unique))
            result.append(
                dict(
                    first_query=first,
                    kv_head=head,
                    queries=len(chosen),
                    distinct_references=distinct_references,
                    union_blocks=len(union),
                    occurrences=references,
                    duplicates=references - distinct_references,
                    reuse_ratio=distinct_references / len(union) if union else 0.0,
                )
            )
    union_total = sum(row["union_blocks"] for row in result)
    return dict(
        tile_queries=tile_queries,
        tiles=result,
        weighted_reuse_ratio=(
            sum(row["distinct_references"] for row in result) / union_total
            if union_total
            else 0.0
        ),
        occurrences=sum(row["occurrences"] for row in result),
        duplicates=sum(row["duplicates"] for row in result),
        union_block_loads=union_total,
        interpretation="Logical selection reuse; not measured memory traffic or speedup",
    )


def selected_reference(query, kv, ids, starts, positions, counts):
    """FP64 CPU oracle with BF16-before-QK scaling and independent row softmax."""
    if any(x.device.type != "cpu" for x in (query, kv, ids, starts, positions, counts)):
        raise ValueError("Reference is CPU-only")
    queries, heads, groups, _ = query.shape
    rows = _cpu_rows(ids, counts, heads)
    out = torch.zeros_like(query, dtype=torch.float64).view(-1, groups, 64)
    q = (query.float() * 0.125).to(query.dtype).double().view(-1, groups, 64)
    for row, physical in enumerate(rows):
        keys, values = [], []
        for block in physical:
            if block >= len(kv):
                raise ValueError("Physical ID exceeds KV capacity")
            allowed = starts[block] + torch.arange(kv.shape[1]) <= positions[row]
            keys.append(kv[block, allowed, :64].double())
            values.append(kv[block, allowed, 64:].double())
        if keys and sum(len(key) for key in keys):
            score = q[row] @ torch.cat(keys).T
            out[row] = score.softmax(-1) @ torch.cat(values)
    return out.view(queries, heads, groups, 64)
