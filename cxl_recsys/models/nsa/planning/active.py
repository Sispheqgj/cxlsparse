"""Bounded selected-block planning without per-request history-table scans.

The caller serializes use of a workspace and commits its hot update once, after
attention completes. Capture/warmup only mutate scratch, so replay is idempotent.
The block-to-slot table scales with the history capacity, but only entries named
by this request are reset. Active keys are sorted before mapping their compact
slots, preserving logical adjacency for IO coalescing and stable hot-set ties.
Addresses are reconstructed from the current owner and host slot table on every
call; no backing address survives across requests.
"""

import math
from collections import Counter
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _reset_workspace(
    Active,
    Required,
    Frequencies,
    Stats,
    RegionCounts,
    InsertKeys,
    HashKeys,
    IDs,
    AllIDs,
    HEADS: tl.constexpr,
    CAP: tl.constexpr,
    SORT: tl.constexpr,
    HASH: tl.constexpr,
    REGIONS: tl.constexpr,
    N_IDS: tl.constexpr,
    N_ALL_IDS: tl.constexpr,
    B: tl.constexpr,
):
    index = tl.program_id(0) * B + tl.arange(0, B)
    # These disjoint scratch arrays have no read-after-write dependency. Keep
    # their exact previous fill values and complete before reference insertion.
    tl.store(Active + index, 0, index < HEADS)
    tl.store(Required + index, 0, index < HEADS * CAP)
    tl.store(Frequencies + index, 0, index < HEADS * SORT)
    tl.store(Stats + index, 0, index < HEADS * 7)
    tl.store(RegionCounts + index, 0, index < REGIONS)
    tl.store(InsertKeys + index, 2147483647, index < HEADS * CAP)
    tl.store(HashKeys + index, -1, index < HEADS * HASH)
    if N_IDS:
        tl.store(IDs + index, -1, index < N_IDS)
    tl.store(AllIDs + index, -1, index < N_ALL_IDS)


@triton.jit
def _clear_references(
    Blocks,
    Positions,
    Desc,
    Mapping,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BW: tl.constexpr,
    MAP: tl.constexpr,
    BS: tl.constexpr,
):
    query, head = tl.program_id(0), tl.program_id(1)
    slot = tl.arange(0, BW)
    block = tl.load(Blocks + (query * HEADS + head) * WIDTH + slot, slot < WIDTH, -1)
    position = tl.load(Positions + query)
    total = tl.load(Desc + head * 5 + 1) + tl.load(Desc + head * 5 + 4)
    valid = (slot < WIDTH) & (position >= 0) & (block >= 0) & (block < MAP)
    valid &= (block * BS <= position) & (block * BS < total)
    tl.store(Mapping + head * (MAP + 1) + block, -1, valid)
    if query == 0:
        tl.store(Mapping + head * (MAP + 1) + MAP, -2)


@triton.jit
def _insert_blocks(
    Blocks,
    Positions,
    Desc,
    Mapping,
    Keys,
    Active,
    Stats,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BW: tl.constexpr,
    MAP: tl.constexpr,
    CAP: tl.constexpr,
    BS: tl.constexpr,
):
    query, head = tl.program_id(0), tl.program_id(1)
    index = tl.arange(0, BW)
    block = tl.load(Blocks + (query * HEADS + head) * WIDTH + index, index < WIDTH, -1)
    position = tl.load(Positions + query)
    total = tl.load(Desc + head * 5 + 1) + tl.load(Desc + head * 5 + 4)
    valid = (index < WIDTH) & (position >= 0) & (block >= 0) & (block < MAP)
    valid &= (block * BS <= position) & (block * BS < total)
    pointer = Mapping + head * (MAP + 1) + tl.where(valid, block, MAP)
    previous = tl.atomic_cas(
        pointer, tl.full((BW,), -1, tl.int32), tl.full((BW,), -2, tl.int32)
    )
    winner = valid & (previous == -1)
    compact = tl.atomic_add(Active + head + tl.zeros((BW,), tl.int32), 1, winner)
    bounded = winner & (compact < CAP)
    tl.store(Keys + head * CAP + compact, block, bounded)
    tl.store(pointer, compact, bounded)
    invalid = (index < WIDTH) & (block >= 0) & (position >= 0) & ~valid
    errors = tl.sum((invalid | (winner & ~bounded)).to(tl.int32), 0)
    tl.atomic_add(Stats + head * 7 + 6, errors)


@triton.jit
def _remap_sorted(
    Keys,
    Active,
    Mapping,
    MAP: tl.constexpr,
    CAP: tl.constexpr,
    B: tl.constexpr,
):
    head, tile = tl.program_id(0), tl.program_id(1)
    compact = tile * B + tl.arange(0, B)
    active = (compact < CAP) & (compact < tl.load(Active + head))
    block = tl.load(Keys + head * CAP + compact, active, -1)
    valid = active & (block >= 0) & (block < MAP)
    tl.store(Mapping + head * (MAP + 1) + block, compact, valid)


@triton.jit
def _hot_hash(
    Tags,
    Locations,
    Desc,
    Keys,
    Values,
    HOT: tl.constexpr,
    TABLE: tl.constexpr,
    BH: tl.constexpr,
):
    head = tl.program_id(0)
    index = tl.arange(0, BH)
    row = tl.load(Desc + head * 5)
    old = tl.load(Desc + head * 5 + 1)
    token = tl.load(Tags + row * (HOT + 1) + index, index < HOT, -1)
    location = tl.load(Locations + row * (HOT + 1) + index, index < HOT, 0)
    pending = (index < HOT) & (token >= 0) & (token < old)
    hashed = (token.to(tl.uint32) * 2654435761) & (TABLE - 1)
    # At most HOT keys occupy TABLE >= 2*HOT slots, so insertion terminates.
    probes = 0
    while (probes < TABLE) & (tl.sum(pending.to(tl.int32), 0) > 0):
        previous = tl.atomic_cas(
            Keys + head * TABLE + hashed,
            tl.full((BH,), -1, tl.int32),
            tl.where(pending, token, -1),
        )
        inserted = pending & ((previous == -1) | (previous == token))
        tl.store(Values + head * TABLE + hashed, location, inserted)
        pending &= ~inserted
        hashed = (hashed + pending.to(tl.uint32)) & (TABLE - 1)
        probes += 1


@triton.jit
def _references(
    Blocks,
    Positions,
    Desc,
    Mapping,
    Required,
    Frequencies,
    Stats,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BW: tl.constexpr,
    MAP: tl.constexpr,
    CAP: tl.constexpr,
    BS: tl.constexpr,
    SORT: tl.constexpr,
):
    query, head = tl.program_id(0), tl.program_id(1)
    index = tl.arange(0, BW)
    offset = tl.arange(0, BS)
    block = tl.load(Blocks + (query * HEADS + head) * WIDTH + index, index < WIDTH, -1)
    position = tl.load(Positions + query)
    old = tl.load(Desc + head * 5 + 1)
    history = tl.load(Desc + head * 5 + 2)
    total = old + tl.load(Desc + head * 5 + 4)
    valid = (index < WIDTH) & (position >= 0) & (block >= 0) & (block < MAP)
    valid &= (block * BS <= position) & (block * BS < total)
    compact = tl.load(Mapping + head * (MAP + 1) + block, valid, -1)
    valid &= (compact >= 0) & (compact < CAP)
    required = tl.minimum(BS, tl.minimum(position + 1, total) - block * BS)
    tl.atomic_max(Required + head * CAP + compact, required, valid)
    token = block[:, None] * BS + offset[None, :]
    allowed = valid[:, None] & (token <= position) & (token < total)
    destination = head * SORT + compact[:, None] * BS + offset[None, :]
    tl.atomic_add(Frequencies + destination, 1, allowed & (token < history))
    old_count = tl.sum(tl.sum((allowed & (token < old)).to(tl.int32), 1), 0)
    local_count = tl.sum(tl.sum((allowed & (token >= old)).to(tl.int32), 1), 0)
    tl.atomic_add(Stats + head * 7 + 4, old_count)
    tl.atomic_add(Stats + head * 7 + 5, local_count)


@triton.jit
def _addresses(
    Keys,
    Active,
    Required,
    HashKeys,
    HashValues,
    Desc,
    HostSlots,
    HotData,
    Bases,
    Address,
    HBM,
    Valid,
    Starts,
    MissingFlags,
    Slots,
    Stats,
    RegionCounts,
    CAP: tl.constexpr,
    BS: tl.constexpr,
    HASH: tl.constexpr,
    MAX_HISTORY: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    REGIONS: tl.constexpr,
    LOCAL: tl.constexpr,
):
    compact, head = tl.program_id(0), tl.program_id(1)
    physical = head * CAP + compact
    active = compact < tl.load(Active + head)
    block = tl.load(Keys + physical, active, -1)
    offset = tl.arange(0, BS)
    token = block * BS + offset
    required = tl.load(Required + physical)
    row = tl.load(Desc + head * 5)
    old = tl.load(Desc + head * 5 + 1)
    pointer = tl.load(Desc + head * 5 + 3)
    valid = active & (offset < required)
    previous = valid & (token < old)
    hashed = (token.to(tl.uint32) * 2654435761) & (HASH - 1)
    pending = previous
    location = tl.full((BS,), -1, tl.int32)
    probes = 0
    while (probes < HASH) & (tl.sum(pending.to(tl.int32), 0) > 0):
        found = tl.load(HashKeys + head * HASH + hashed, pending, -1)
        match = pending & (found == token)
        found_location = tl.load(HashValues + head * HASH + hashed, match, -1)
        location = tl.where(match, found_location, location)
        pending &= (found != -1) & ~match
        hashed = (hashed + pending.to(tl.uint32)) & (HASH - 1)
        probes += 1
    external = previous & (location < 0)
    source = tl.load(
        HostSlots + row * MAX_HISTORY + token,
        external & (token >= 0) & (token < MAX_HISTORY),
        -1,
    )
    error = external & ((source < 0) | (source >= POOL_TOKENS))
    safe_external = external & ~error
    current_address = pointer + (token - old).to(tl.int64) * 256
    hot_address = HotData + location.to(tl.int64) * 128
    hbm_address = tl.where(previous, hot_address.to(tl.int64), current_address)
    hbm_address = tl.where(valid & ~external, hbm_address, 0)
    if LOCAL:
        page = source // PAGE_SIZE
        region = page % REGIONS
        local_slot = (page // REGIONS) * PAGE_SIZE + source % PAGE_SIZE
        base = tl.load(Bases + region, safe_external, 0)
        address = tl.where(safe_external, base + local_slot * 256, hbm_address)
        for r in tl.static_range(REGIONS):
            count = tl.sum((safe_external & (region == r)).to(tl.int32), 0)
            tl.atomic_add(RegionCounts + r, count)
    else:
        address = hbm_address
    tl.store(Address + physical * BS + offset, address)
    tl.store(HBM + physical * BS + offset, hbm_address)
    tl.store(Slots + physical * BS + offset, tl.where(safe_external, source, -1))
    errors = tl.sum(error.to(tl.int32), 0)
    tl.store(Valid + physical, tl.where(active & (errors == 0), required, 0))
    tl.store(Starts + physical, tl.where(active, block * BS, 0))
    tl.store(MissingFlags + physical, tl.sum(external.to(tl.int32), 0) > 0)
    tl.atomic_add(Stats + head * 7, tl.sum(valid.to(tl.int32), 0))
    tl.atomic_add(Stats + head * 7 + 1, tl.sum(previous.to(tl.int32), 0))
    tl.atomic_add(Stats + head * 7 + 2, tl.sum((previous & ~external).to(tl.int32), 0))
    tl.atomic_add(Stats + head * 7 + 3, tl.sum(external.to(tl.int32), 0))
    tl.atomic_add(Stats + head * 7 + 6, errors)


@triton.jit
def _ranks(Flags, IOIDs, Ranks, Missing, N: tl.constexpr, BN: tl.constexpr):
    physical = tl.arange(0, BN)
    missing = tl.load(Flags + physical, physical < N, 0)
    rank = tl.cumsum(missing, 0) - 1
    tl.store(IOIDs + rank, physical, (physical < N) & (missing != 0))
    tl.store(Ranks + physical, rank, physical < N)
    tl.store(Missing, tl.sum(missing, 0))


@triton.jit
def _stage_maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _stages(
    Blocks,
    Positions,
    Mapping,
    Flags,
    Ranks,
    Stats,
    IDs,
    Counts,
    RowPositions,
    AllIDs,
    AllCounts,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BW: tl.constexpr,
    MAP: tl.constexpr,
    CAP: tl.constexpr,
    BS: tl.constexpr,
    CHUNK: tl.constexpr,
    PRESERVE_ORDER: tl.constexpr = True,
    BLOCK_READY: tl.constexpr = False,
):
    row, stage = tl.program_id(0), tl.program_id(1)
    query, head = row // HEADS, row % HEADS
    index = tl.arange(0, BW)
    block = tl.load(Blocks + row * WIDTH + index, index < WIDTH, -1)
    position = tl.load(Positions + query)
    valid = (index < WIDTH) & (position >= 0) & (block >= 0) & (block < MAP)
    valid &= (block * BS <= position) & (tl.load(Stats + head * 7 + 6) == 0)
    compact = tl.load(Mapping + head * (MAP + 1) + block, valid, -1)
    valid &= (compact >= 0) & (compact < CAP)
    physical = head * CAP + compact
    missing = tl.load(Flags + physical, valid, 0)
    rank = tl.load(Ranks + physical, valid, 0)
    target = tl.where(missing != 0, 1 if BLOCK_READY else 1 + rank // CHUNK, 0)
    if PRESERVE_ORDER:
        # Strict mode delays ready occurrences behind earlier selections to
        # reproduce the original BF16 accumulation order.
        target = tl.associative_scan(tl.where(valid, target, 0), 0, _stage_maximum)
    # Otherwise use each block's actual readiness stage. Stable compaction
    # retains occurrence order within a stage, including its multiplicity.
    selected = valid & (stage == target)
    output_index = tl.cumsum(selected.to(tl.int32), 0) - 1
    tl.store(IDs + (stage * ROWS + row) * WIDTH + output_index, physical, selected)
    tl.store(Counts + stage * ROWS + row, tl.sum(selected.to(tl.int32), 0))
    if stage == 0:
        tl.store(RowPositions + row, position)
        all_index = tl.cumsum(valid.to(tl.int32), 0) - 1
        tl.store(AllIDs + row * WIDTH + all_index, physical, valid)
        tl.store(AllCounts + row, tl.sum(valid.to(tl.int32), 0))


@triton.jit
def _copy_hbm(Address, KV, N: tl.constexpr, TOKENS: tl.constexpr = 1):
    token = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    d = tl.arange(0, 128)
    pointer = tl.load(Address + token, token < N, 0)
    value = tl.load(
        pointer[:, None].to(tl.pointer_type(tl.bfloat16)) + d[None, :],
        (token[:, None] < N) & (pointer[:, None] != 0),
        0,
    )
    # Zero-source slots are still written, including inactive capacity slots.
    # Consumers see exactly the same initialized buffer as the one-token path.
    tl.store(KV + token[:, None] * 128 + d[None, :], value, token[:, None] < N)


@triton.jit
def _rdma_metadata(
    Slots,
    Flags,
    Ranks,
    Missing,
    Stats,
    Meta,
    N: tl.constexpr,
    BS: tl.constexpr,
    HEADS: tl.constexpr,
    META_WIDTH: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0) * B + tl.arange(0, B)
    physical = token // BS
    valid = (token < N) & (tl.load(Flags + physical, token < N, 0) != 0)
    rank = tl.load(Ranks + physical, valid, 0)
    source = tl.load(Slots + token, valid, -1)
    destination = 1 + rank * BS + token % BS
    tl.store(Meta + destination, source, valid)
    tl.store(Meta + META_WIDTH + destination, token, valid)
    if tl.program_id(0) == 0:
        tl.store(Meta, tl.load(Missing))
        for i in tl.static_range(HEADS * 7):
            tl.store(Meta + 1 + N + i, tl.load(Stats + i))


@triton.jit
def _retain_hot(
    Desc,
    Chosen,
    Frequencies,
    Keys,
    KV,
    Tags,
    Locations,
    Data,
    CAP: tl.constexpr,
    BS: tl.constexpr,
    SORT: tl.constexpr,
    HOT: tl.constexpr,
):
    head, slot = tl.program_id(0), tl.program_id(1)
    row = tl.load(Desc + head * 5)
    compact_token = tl.load(Chosen + head * HOT + slot)
    selected = tl.load(Frequencies + head * SORT + compact_token) > 0
    compact = compact_token // BS
    block = tl.load(Keys + head * CAP + compact, selected & (compact < CAP), -1)
    token = block * BS + compact_token % BS
    index = row * (HOT + 1) + slot
    location = tl.load(Locations + index)
    d = tl.arange(0, 128)
    values = tl.load(KV + (head * CAP * BS + compact_token) * 128 + d, selected, 0)
    tl.store(Tags + index, tl.where(selected, token, -1))
    tl.store(Data + location * 128 + d, values, selected)


@triton.jit
def _sort_active_keys(Keys, Sorted, CAP: tl.constexpr, B: tl.constexpr):
    head = tl.program_id(0)
    i = tl.arange(0, B)
    keys = tl.load(Keys + head * CAP + i, i < CAP, 2147483647)
    keys = tl.sort(keys, descending=False)
    tl.store(Sorted + head * CAP + i, keys, i < CAP)


@triton.jit
def _hot_tile_top(
    Frequencies,
    Partial,
    N: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    B: tl.constexpr,
):
    head, tile = tl.program_id(0), tl.program_id(1)
    i = tile * B + tl.arange(0, B)
    f = tl.load(Frequencies + head * N + i, i < N, -1)
    # Unique priority preserves descending frequency / ascending compact token.
    priority = (f.to(tl.int64) << 32) + (2147483647 - i).to(tl.int64)
    priority = tl.sort(priority, descending=True)
    j = tl.arange(0, B)
    tl.store(Partial + (head * TILES + tile) * K + j, priority, j < K)


@triton.jit
def _hot_merge_top(
    Partial,
    Chosen,
    TILES: tl.constexpr,
    K: tl.constexpr,
    HOT: tl.constexpr,
    B: tl.constexpr,
    STEPS: tl.constexpr,
):
    head, source = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, K)
    tile = tl.arange(0, (B // K))
    value = tl.load(Partial + (head * TILES + source) * K + i)
    # Each tile is already sorted with unique priorities. Count greater keys
    # by binary search in every tile, then sum to obtain the exact global rank.
    # This exposes all tiles to independent CTAs instead of sorting 4096 keys
    # in one CTA. Truncating each tile to K cannot discard a global top-HOT key.
    low = tl.full(((B // K), K), 0, tl.int32)
    high = tl.full(((B // K), K), K, tl.int32)
    for _ in range(STEPS):
        mid = (low + high) // 2
        probe = tl.load(
            Partial + (head * TILES + tile[:, None]) * K + mid,
            (tile[:, None] < TILES) & (mid < K),
            -9223372036854775807,
        )
        greater = probe > value[None, :]
        low = tl.where(greater, mid + 1, low)
        high = tl.where(greater, high, mid)
    rank = tl.sum(low, axis=0)
    token = 2147483647 - (value & 4294967295).to(tl.int32)
    tl.store(Chosen + head * HOT + rank, token, rank < HOT)


class ActivePlanAdapter:
    """Scratch-only GPU plan; owner/stream ordering is enforced by transport.

    ``map_blocks`` is a host-known capacity, not the dynamic union length. It
    can be a history bucket. ``per_head`` is the compact active-block capacity.
    CXL never downloads counts, selections, tags or addresses during planning.
    """

    def __init__(self, cache, *, chunk_blocks=64, rdma_coalesce=True):
        self.cache, self.pool = cache, cache.pool
        self.chunk_blocks, self.rdma_coalesce = chunk_blocks, rdma_coalesce
        if chunk_blocks < 1 or cache.hot < 1:
            raise ValueError("Positive chunk and hot capacities are required")

    def geometry(self, query, blocks, fetches, bs, *, map_blocks=None, all_hbm=False):
        heads, width = query.shape[1], blocks.shape[-1]
        if bs < 1 or bs & (bs - 1) or not heads or not width or not len(query):
            raise ValueError("Nonempty geometry and a power-of-two block size required")
        if len(fetches) != heads or blocks.shape[:2] != query.shape[:2]:
            raise ValueError("Selected geometry does not match fetch descriptors")
        needed = math.ceil(
            max(f["cached_length"] + len(f["current_kv"]) for f in fetches) / bs
        )
        if map_blocks is None:
            map_blocks = needed
        if map_blocks < needed or map_blocks < 1:
            raise ValueError("History mapping capacity cannot truncate a request")
        if map_blocks * bs > torch.iinfo(torch.int32).max:
            raise ValueError("Absolute token addressing must fit int32")
        per_head = min(map_blocks, len(query) * width)
        physical, rows = heads * per_head, len(query) * heads
        chunk = (
            physical
            if all_hbm or (self.pool.backend == "rdma" and self.rdma_coalesce)
            else self.chunk_blocks
        )
        return dict(
            heads=heads,
            per_head=per_head,
            map_blocks=map_blocks,
            map_stride=map_blocks + 1,
            extent=per_head * bs,
            sort_extent=max(per_head * bs, self.cache.hot),
            hash_capacity=triton.next_power_of_2(2 * self.cache.hot),
            bs=bs,
            rows=rows,
            width=width,
            physical=physical,
            stages=1 + math.ceil(physical / chunk),
            chunk=chunk,
        )

    def allocate(
        self,
        query,
        blocks,
        fetches,
        bs,
        *,
        map_blocks=None,
        all_hbm=False,
        read_order="physical",
        occurrence_only=False,
    ):
        w = self.geometry(
            query, blocks, fetches, bs, map_blocks=map_blocks, all_hbm=all_hbm
        )
        h, c, p, rows, width, stages = (
            w[k] for k in ("heads", "per_head", "physical", "rows", "width", "stages")
        )
        stage_tables = 0 if occurrence_only else stages

        def empty(shape, dtype=torch.int32):
            return torch.empty(shape, dtype=dtype, device=query.device)

        w.update(
            occurrence_only=occurrence_only,
            mapping=empty((h, w["map_stride"])),
            insert_keys=empty((h, c)),
            active_keys=empty((h, c)),
            key_order=empty((h, c), torch.int64),
            active_counts=empty((h,)),
            required=empty((h, c)),
            hot_hash_keys=empty((h, w["hash_capacity"])),
            hot_hash_values=empty((h, w["hash_capacity"])),
            frequencies=empty((h, w["sort_extent"])),
            stats=empty((h, 7)),
            address=empty((p, bs), torch.int64),
            hbm=empty((p, bs), torch.int64),
            valid=empty((p,)),
            starts=empty((p,)),
            flags=empty((p,)),
            slots=empty((p * bs,), torch.int64),
            ioids=empty((p,)),
            ranks=empty((p,)),
            missing=empty((1,)),
            ids=empty((stage_tables, rows, width)),
            counts=empty((stage_tables, rows)),
            all_ids=empty((rows, width)),
            all_counts=empty((rows,)),
            positions=empty((rows,)),
            region_counts=empty((self.pool.region_count,)),
            kv=empty((p, bs, 128), torch.bfloat16),
            output=torch.empty_like(query.contiguous().flatten(0, 1)),
            barriers=empty((stage_tables,)),
            compute_trace=empty((rows, stage_tables, 2), torch.uint64),
            io_trace=empty((0 if occurrence_only else p, 2), torch.uint64),
        )
        if (
            c <= 4096
            and self.cache.hot <= 1024
            and triton.cdiv(w["sort_extent"], 1024)
            * triton.next_power_of_2(self.cache.hot)
            <= 4096
        ):
            k = triton.next_power_of_2(self.cache.hot)
            tile = max(1024, k)
            tiles = triton.cdiv(w["sort_extent"], tile)
            w["hot_partial"] = empty((h, tiles, k), torch.int64)
            w["hot_chosen"] = empty((h, self.cache.hot), torch.int64)
            w["hot_tile"] = tile
        if self.pool.backend == "rdma":
            meta_width = 1 + p * bs + h * 7
            w["meta"] = empty((2, meta_width), torch.int64)
            w["meta_host"] = torch.empty(
                (2, meta_width), dtype=torch.int64, pin_memory=True
            )
            w["rdma_values"] = w["kv"].flatten(0, 1)
            w["maximum"] = empty((rows, query.shape[2]), torch.float32)
            w["denominator"] = torch.empty_like(w["maximum"])
            w["accumulator"] = torch.empty_like(w["output"], dtype=torch.float32)
        w["persistent_mapping_bytes"] = (
            w["mapping"].numel() * w["mapping"].element_size()
        )
        w["scratch_bytes"] = sum(
            v.numel() * v.element_size()
            for k, v in w.items()
            if isinstance(v, torch.Tensor) and k not in ("mapping", "rdma_values")
        )
        w["reset_bytes"] = sum(
            w[k].numel() * w[k].element_size()
            for k in (
                "active_counts",
                "required",
                "frequencies",
                "stats",
                "region_counts",
                "insert_keys",
                "hot_hash_keys",
                "ids",
                "all_ids",
            )
        )
        if read_order != "physical":
            from cxl_recsys.models.nsa.planning.read_order import allocate_read_order

            allocate_read_order(w, query.device, order=read_order)
            if read_order == "query_first_use_region":
                from cxl_recsys.models.nsa.planning.read_order import (
                    allocate_region_order,
                )

                allocate_region_order(
                    w, query.device, self.pool.region_count, self.pool.page_size
                )
        return w

    def plan(
        self,
        blocks,
        positions,
        desc,
        bases,
        w,
        *,
        consume_order="strict",
        read_order="physical",
        read_group_size=16,
    ):
        """Prepare compact sources and occurrences; no external IO.

        TMA's occurrence-only path leaves payload initialization to its source
        adapter, which combines resident copies with partial external gathers.
        """
        if consume_order not in ("strict", "resident_first", "block_ready"):
            raise ValueError("Unknown NSA consumption order")
        from cxl_recsys.models.nsa.planning.read_order import READ_ORDERS, reorder_reads

        if read_order not in READ_ORDERS or read_group_size < 1:
            raise ValueError("Invalid NSA read priority configuration")
        if read_order != "physical" and (
            self.pool.backend == "rdma"
            or (len(blocks) > 128 and "consumer_ctas" not in w)
        ):
            raise ValueError("Read priority supports only local cooperative Q<=128")
        heads, cap, bs, mapping = (
            w[k] for k in ("heads", "per_head", "bs", "map_blocks")
        )
        reset_elements = max(
            heads * max(cap, w["sort_extent"], w["hash_capacity"], 7),
            self.pool.region_count,
            w["ids"].numel(),
            w["all_ids"].numel(),
        )
        _reset_workspace[(triton.cdiv(reset_elements, 256),)](
            w["active_counts"],
            w["required"],
            w["frequencies"],
            w["stats"],
            w["region_counts"],
            w["insert_keys"],
            w["hot_hash_keys"],
            w["ids"],
            w["all_ids"],
            heads,
            cap,
            w["sort_extent"],
            w["hash_capacity"],
            self.pool.region_count,
            w["ids"].numel(),
            w["all_ids"].numel(),
            256,
            num_warps=4,
        )
        geometry = (heads, w["width"], triton.next_power_of_2(w["width"]), mapping)
        _clear_references[(len(blocks), heads)](
            blocks, positions, desc, w["mapping"], *geometry, bs, num_warps=1
        )
        _insert_blocks[(len(blocks), heads)](
            blocks,
            positions,
            desc,
            w["mapping"],
            w["insert_keys"],
            w["active_counts"],
            w["stats"],
            *geometry,
            cap,
            bs,
            num_warps=1,
        )
        if "hot_partial" in w:
            _sort_active_keys[(heads,)](
                w["insert_keys"],
                w["active_keys"],
                cap,
                triton.next_power_of_2(cap),
                num_warps=4,
            )
        else:
            torch.sort(w["insert_keys"], dim=1, out=(w["active_keys"], w["key_order"]))
        _remap_sorted[(heads, triton.cdiv(cap, 256))](
            w["active_keys"], w["active_counts"], w["mapping"], mapping, cap, 256
        )
        _hot_hash[(heads,)](
            self.cache.tags,
            self.cache.locs,
            desc,
            w["hot_hash_keys"],
            w["hot_hash_values"],
            self.cache.hot,
            w["hash_capacity"],
            triton.next_power_of_2(self.cache.hot),
            num_warps=4,
        )
        _references[(len(blocks), heads)](
            blocks,
            positions,
            desc,
            w["mapping"],
            w["required"],
            w["frequencies"],
            w["stats"],
            *geometry,
            cap,
            bs,
            w["sort_extent"],
            num_warps=4,
        )
        _addresses[(cap, heads)](
            w["active_keys"],
            w["active_counts"],
            w["required"],
            w["hot_hash_keys"],
            w["hot_hash_values"],
            desc,
            self.cache.host_locs,
            self.cache.data,
            bases,
            w["address"],
            w["hbm"],
            w["valid"],
            w["starts"],
            w["flags"],
            w["slots"],
            w["stats"],
            w["region_counts"],
            cap,
            bs,
            w["hash_capacity"],
            self.cache.max_history,
            self.pool.pages * self.pool.page_size,
            self.pool.page_size,
            self.pool.region_count,
            self.pool.backend != "rdma",
            num_warps=1,
        )
        _ranks[(1,)](
            w["flags"],
            w["ioids"],
            w["ranks"],
            w["missing"],
            w["physical"],
            triton.next_power_of_2(w["physical"]),
            num_warps=4,
        )

        def stages():
            _stages[(w["rows"], w["stages"])](
                blocks,
                positions,
                w["mapping"],
                w["flags"],
                w["ranks"],
                w["stats"],
                w["ids"],
                w["counts"],
                w["positions"],
                w["all_ids"],
                w["all_counts"],
                heads,
                w["rows"],
                w["width"],
                triton.next_power_of_2(w["width"]),
                mapping,
                cap,
                bs,
                w["chunk"],
                PRESERVE_ORDER=consume_order == "strict",
                BLOCK_READY=consume_order == "block_ready",
                num_warps=1,
            )

        if w["occurrence_only"]:
            from cxl_recsys.models.nsa.planning.occurrences import build_occurrences

            build_occurrences(blocks, positions, w)
        else:
            stages()
        if read_order != "physical":
            reorder_reads(w, read_order, read_group_size)
            if (
                not w["occurrence_only"]
                and consume_order != "block_ready"
                and "consumer_ctas" not in w
            ):
                stages()  # Batch consumers use the newly assigned ranks.
        # Compact token order is absolute token order within each head. Stable
        # int32 frequency sorting therefore gives the baseline tie rule without
        # encoding absolute IDs into an int64 priority or using int64 topk.
        if "hot_partial" in w:
            _, tiles, k = w["hot_partial"].shape
            _hot_tile_top[(heads, tiles)](
                w["frequencies"],
                w["hot_partial"],
                w["sort_extent"],
                k,
                tiles,
                w["hot_tile"],
                num_warps=4,
            )
            chosen = w["hot_chosen"]
            _hot_merge_top[(heads, tiles)](
                w["hot_partial"],
                chosen,
                tiles,
                k,
                self.cache.hot,
                triton.next_power_of_2(tiles * k),
                k.bit_length(),
                num_warps=8,
            )
        else:
            chosen = torch.argsort(
                w["frequencies"], dim=1, descending=True, stable=True
            )[:, : self.cache.hot].contiguous()
        if not w["occurrence_only"]:
            _copy_hbm[(triton.cdiv(w["physical"] * bs, 8),)](
                w["hbm"], w["kv"], w["physical"] * bs, TOKENS=8, num_warps=4
            )
        if self.pool.backend == "rdma":
            _rdma_metadata[(triton.cdiv(w["physical"] * bs, 256),)](
                w["slots"],
                w["flags"],
                w["ranks"],
                w["missing"],
                w["stats"],
                w["meta"],
                w["physical"] * bs,
                bs,
                heads,
                w["meta"].shape[1],
                256,
            )
        return chosen

    def retain_hot(self, desc, chosen, w):
        """Commit once after successful attention, outside Graph warmup/capture."""
        _retain_hot[(w["heads"], self.cache.hot)](
            desc,
            chosen,
            w["frequencies"],
            w["active_keys"],
            w["kv"],
            self.cache.tags,
            self.cache.locs,
            self.cache.data,
            w["per_head"],
            w["bs"],
            w["sort_extent"],
            self.cache.hot,
            num_warps=1,
        )


@dataclass(frozen=True)
class ReferenceActivePlan:
    """Independent CPU specification; source maps are supplied by the fixture."""

    keys: tuple[tuple[int, int], ...]
    row_keys: tuple[tuple[tuple[int, int], ...], ...]
    required: tuple[int, ...]
    sources: tuple[tuple[tuple[str, int], ...], ...]
    hot: tuple[tuple[int, ...], ...]
    stats: tuple[tuple[int, ...], ...]


def reference_active_plan(
    blocks, positions, fetches, *, block_size, hot, capacity=None
):
    """Reference with CPU descriptors: cached/history/total, hot and backing maps.

    Each fetch is ``{cached_length, history_length, total_length, hot_locations,
    host_slots}``. This oracle does not call any production planner or kernel.
    Negative positions represent padded rows; they have no observable effect.
    """
    if blocks.device.type != "cpu" or positions.device.type != "cpu":
        raise ValueError("The independent reference accepts CPU inputs only")
    if (
        blocks.ndim != 3
        or len(positions) != len(blocks)
        or len(fetches) != blocks.shape[1]
    ):
        raise ValueError("Invalid reference geometry")
    if min(block_size, hot) < 1:
        raise ValueError("Positive block and hot capacities required")
    required, rows = {}, []
    frequencies = [Counter() for _ in fetches]
    stats = [[0] * 7 for _ in fetches]
    for choices, position in zip(blocks.tolist(), positions.tolist(), strict=True):
        for head, ids in enumerate(choices):
            f = fetches[head]
            row = []
            for block in ids:
                if position < 0 or block == -1:
                    continue
                if (
                    block < -1
                    or block * block_size > position
                    or block * block_size >= f["total_length"]
                ):
                    raise ValueError("Invalid selected block")
                length = min(
                    block_size,
                    position + 1 - block * block_size,
                    f["total_length"] - block * block_size,
                )
                key = (head, block)
                row.append(key)
                required[key] = max(required.get(key, 0), length)
                for token in range(block * block_size, block * block_size + length):
                    stats[head][4 if token < f["cached_length"] else 5] += 1
                    if token < f["history_length"]:
                        frequencies[head][token] += 1
            rows.append(tuple(row))
    if capacity is not None and any(
        sum(h == head for h, _ in required) > capacity for head in range(len(fetches))
    ):
        raise ValueError("Active-block capacity overflow")
    sources = []
    for (head, block), length in required.items():
        f, source = fetches[head], []
        for token in range(block * block_size, block * block_size + length):
            stats[head][0] += 1
            if token >= f["cached_length"]:
                source.append(("current", token - f["cached_length"]))
                continue
            stats[head][1] += 1
            if token in f["hot_locations"]:
                stats[head][2] += 1
                source.append(("hot", f["hot_locations"][token]))
            else:
                stats[head][3] += 1
                source.append(("external", f["host_slots"][token]))
        sources.append(tuple(source))
    chosen = tuple(
        tuple(sorted(f, key=lambda token: (-f[token], token))[:hot])
        for f in frequencies
    )
    return ReferenceActivePlan(
        tuple(required),
        tuple(rows),
        tuple(required.values()),
        tuple(sources),
        chosen,
        tuple(tuple(row) for row in stats),
    )
