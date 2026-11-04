"""NOSA block selections with the unchanged token-frequency hot-cache policy.

Full blocks contribute one reference count per block. Only the causal boundary
contributes per-token counts. Compact KV remains token-granular, so arbitrary
hot-cache contents, history boundaries and physical pool locations are valid.
No query-by-selected-token IDs or compact maps are materialized.
"""

import triton
import triton.language as tl

# Triton's reduction stubs also describe tuple reductions; these are tensor
# reductions compiled by Triton, not Python operations on tuples.
# pyright: reportOperatorIssue=false


@triton.jit
def _initialize(PART, FULL, LOOKUP, STATS, E: tl.constexpr, NB: tl.constexpr):
    row = tl.program_id(1)
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(PART + row * E + x, 0, x < E)
    tl.store(LOOKUP + row * E + x, -1, x < E)
    tl.store(FULL + row * NB + x, 0, x < NB)
    if tl.program_id(0) == 0:
        tl.store(STATS + row * 7 + x, 0, x < 7)


@triton.jit
def _lookup(DESC, TAGS, LOOKUP, E: tl.constexpr, HOT: tl.constexpr):
    row = tl.program_id(1)
    slot = tl.program_id(0) * 256 + tl.arange(0, 256)
    cache_row = tl.load(DESC + row * 5)
    token = tl.load(TAGS + cache_row * (HOT + 1) + slot, slot < HOT, other=-1)
    tl.store(LOOKUP + row * E + token, slot, (token >= 0) & (token < E))


@triton.jit
def _count(
    IDS,
    DESC,
    FULL,
    PART,
    STATS,
    E: tl.constexpr,
    NB: tl.constexpr,
    W: tl.constexpr,
    BW: tl.constexpr,
):
    row = tl.program_id(1)
    q = tl.program_id(0)
    count = tl.load(DESC + row * 5 + 3)
    if q < count:
        pos = tl.load(DESC + row * 5 + 1) + q
        begin = tl.load(DESC + row * 5 + 2)
        c = tl.arange(0, BW)
        block = tl.load(IDS + (begin + q) * W + c, c < W, other=-1)
        valid = (block >= 0) & (block <= pos // 64) & (block < NB)
        bad = (block != -1) & ~valid
        if (tl.sum(valid.to(tl.int32), 0) == 0) | (tl.sum(bad.to(tl.int32), 0) > 0):
            tl.atomic_or(STATS + row * 7 + 6, 1)
        # The current block is partial unless its final token is causal.
        full = valid & ((block < pos // 64) | (pos % 64 == 63))
        tl.atomic_add(FULL + row * NB + block, 1, full, sem="relaxed")
        partial = tl.sum((valid & ~full).to(tl.int32), 0)
        n = tl.arange(0, 64)
        token = (pos // 64) * 64 + n
        tl.atomic_add(
            PART + row * E + token,
            partial,
            (partial > 0) & (n <= pos % 64) & (token < E),
            sem="relaxed",
        )


@triton.jit
def _scan(
    DESC,
    FULL,
    PART,
    LOOKUP,
    RANK,
    FREQ,
    MISS,
    TILES,
    E: tl.constexpr,
    NB: tl.constexpr,
    NT: tl.constexpr,
):
    row = tl.program_id(1)
    tile = tl.program_id(0)
    token = tile * 256 + tl.arange(0, 256)
    cached = tl.load(DESC + row * 5 + 1)
    length = cached + tl.load(DESC + row * 5 + 3)
    history = tl.load(DESC + row * 5 + 4)
    refs = tl.load(FULL + row * NB + token // 64, token < E, other=0)
    refs += tl.load(PART + row * E + token, token < E, other=0)
    present = (token < length) & (token < E) & (refs > 0)
    old = present & (token < cached)
    hit = old & (tl.load(LOOKUP + row * E + token, token < E, other=-1) >= 0)
    miss = old & ~hit
    rank = tl.cumsum(present.to(tl.int32), 0) - 1
    miss_rank = tl.cumsum(miss.to(tl.int32), 0) - miss.to(tl.int32)
    tl.store(RANK + row * E + token, tl.where(present, rank, -1), token < E)
    tl.store(MISS + row * E + token, miss_rank, token < E)
    tl.store(
        FREQ + row * E + token,
        tl.where(present & (token < history), refs, 0),
        token < E,
    )
    base = TILES + (row * NT + tile) * 6
    tl.store(base, tl.sum(present.to(tl.int32), 0))
    tl.store(base + 1, tl.sum(miss.to(tl.int32), 0))
    tl.store(base + 2, tl.sum(old.to(tl.int32), 0))
    tl.store(base + 3, tl.sum(hit.to(tl.int32), 0))
    tl.store(base + 4, tl.sum(tl.where(old, refs, 0), 0))
    tl.store(base + 5, tl.sum(tl.where(present & ~old, refs, 0), 0))


@triton.jit
def _prefix(TILES, STATS, NT: tl.constexpr, BT: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, BT)
    for field in tl.static_range(6):  # pyright: ignore[reportGeneralTypeIssues]
        value = tl.load(TILES + (row * NT + tile) * 6 + field, tile < NT, other=0)
        prefix = tl.cumsum(value, 0) - value
        tl.store(TILES + (row * NT + tile) * 6 + field, prefix, tile < NT)
        # Tile fields: unique, misses, old, hits, old refs, local refs.
        out = 3 if field == 1 else 1 if field == 2 else 2 if field == 3 else field
        tl.store(STATS + row * 7 + out, tl.sum(value, 0))


@triton.jit
def _materialize(
    DESC,
    LOOKUP,
    HOST,
    LOCS,
    CACHE,
    CURRENT,
    RANK,
    MISS,
    TILES,
    LOGICAL,
    SOURCES,
    DEST,
    STATS,
    SRC_PTR,
    HBM_DST,
    E: tl.constexpr,
    NT: tl.constexpr,
    HISTORY: tl.constexpr,
    CAP: tl.constexpr,
    DIM: tl.constexpr,
    HOT: tl.constexpr,
    POOL: tl.constexpr,
):
    row = tl.program_id(1)
    tile = tl.program_id(0)
    token = tile * 256 + tl.arange(0, 256)
    rank = tl.load(RANK + row * E + token, token < E, other=-1)
    valid = (token < E) & (rank >= 0)
    cached = tl.load(DESC + row * 5 + 1)
    cache_row = tl.load(DESC + row * 5)
    begin = tl.load(DESC + row * 5 + 2)
    compact = rank + tl.load(TILES + (row * NT + tile) * 6)
    miss = tl.load(MISS + row * E + token, valid, other=0)
    miss += tl.load(TILES + (row * NT + tile) * 6 + 1)
    tl.store(RANK + row * E + token, compact, valid)
    tl.store(LOGICAL + row * E + compact, token, valid)
    slot = tl.load(LOOKUP + row * E + token, valid, other=-1)
    from_pool = valid & (token < cached) & (slot < 0)
    source = tl.load(HOST + cache_row * HISTORY + token, from_pool, other=0)
    if tl.sum((from_pool & ((source < 0) | (source >= POOL))).to(tl.int32), 0) > 0:
        tl.atomic_or(STATS + row * 7 + 6, 1)
    tl.store(SOURCES + row * E + miss, source, from_pool)
    tl.store(DEST + row * E + miss, compact, from_pool)
    hbm = valid & ~from_pool
    location = tl.load(
        LOCS + cache_row * (HOT + 1) + slot, hbm & (token < cached), other=0
    )
    address = tl.where(
        token < cached,
        CACHE + location.to(tl.int64) * DIM,
        CURRENT + (begin + token - cached).to(tl.int64) * DIM,
    )
    tl.store(SRC_PTR + row * CAP + compact - miss, address.to(tl.uint64), hbm)
    tl.store(HBM_DST + row * CAP + compact - miss, compact, hbm)


def plan(cache, blocks, descriptors, current, work, max_queries):
    """Produce the existing token compact plan without expanding query IDs."""
    import torch

    batch, extent = work["ranks"].shape
    nb, nt = triton.cdiv(extent, 64), triton.cdiv(extent, 256)
    # Reused scratch is owned and stream-ordered by HiSparseCache.
    shape = (batch, extent)
    if not hasattr(cache, "_block_scratches"):
        cache._block_scratches = {}
    scratch = cache._block_scratches.get(shape)
    if scratch is None:
        scratch = (
            torch.empty(shape, dtype=torch.int32, device=blocks.device),
            torch.empty((batch, nb), dtype=torch.int32, device=blocks.device),
            torch.empty((batch, nt, 6), dtype=torch.int32, device=blocks.device),
        )
        cache._block_scratches[shape] = scratch
    part, full, tiles = scratch
    for tensor in scratch:
        tensor.record_stream(torch.cuda.current_stream(blocks.device))
    _initialize.run(
        part,
        full,
        work["hot_lookup"],
        work["stats"],
        extent,
        nb,
        grid=(nt, batch),
        warmup=False,
    )
    _lookup.run(
        descriptors,
        cache.tags,
        work["hot_lookup"],
        extent,
        cache.hot,
        grid=(triton.cdiv(cache.hot, 256), batch),
        warmup=False,
    )
    _count.run(
        blocks,
        descriptors,
        full,
        part,
        work["stats"],
        extent,
        nb,
        blocks.shape[1],
        triton.next_power_of_2(blocks.shape[1]),
        grid=(max_queries, batch),
        warmup=False,
    )
    _scan.run(
        descriptors,
        full,
        part,
        work["hot_lookup"],
        work["ranks"],
        work["frequencies"],
        work["miss_offsets"],
        tiles,
        extent,
        nb,
        nt,
        grid=(nt, batch),
        warmup=False,
    )
    _prefix.run(
        tiles,
        work["stats"],
        nt,
        triton.next_power_of_2(nt),
        grid=(batch,),
        warmup=False,
    )
    _materialize.run(
        descriptors,
        work["hot_lookup"],
        cache.host_locs,
        cache.locs,
        cache.data,
        current,
        work["ranks"],
        work["miss_offsets"],
        tiles,
        work["logical"],
        work["sources"],
        work["destinations"],
        work["stats"],
        work["hbm_src_ptrs"],
        work["hbm_dst_local"],
        extent,
        nt,
        cache.max_history,
        work["hbm_src_ptrs"].shape[1],
        cache.dim,
        cache.hot,
        cache.pool.pages * cache.pool.page_size,
        grid=(nt, batch),
        warmup=False,
    )
