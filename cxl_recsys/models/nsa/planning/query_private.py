"""Query-private KV transfers with the same layer-wide hot-token policy.

Each (query, KV head, block) owns a payload slot. Repeated references inside
one query keep their multiplicity but share that query's slot. The separate
absolute-token frequency table chooses the hot set only; it never supplies a
miss to another query. All payloads are fetched into their own destinations.
"""

import math

import torch
import triton
import triton.language as tl

from cxl_recsys.models.nsa.planning.active import (
    ActivePlanAdapter,
    _addresses,
    _copy_hbm,
    _hot_hash,
    _hot_merge_top,
    _hot_tile_top,
    _ranks,
    _rdma_metadata,
    _reset_workspace,
    _stage_maximum,
)


@triton.jit
def _private_references(
    Blocks,
    Positions,
    Desc,
    Keys,
    Required,
    Occurrences,
    Frequencies,
    Canonical,
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
    i = tl.arange(0, BW)
    block = tl.load(Blocks + (query * HEADS + head) * WIDTH + i, i < WIDTH, -1)
    position = tl.load(Positions + query)
    old = tl.load(Desc + head * 5 + 1)
    history = tl.load(Desc + head * 5 + 2)
    total = old + tl.load(Desc + head * 5 + 4)
    valid = (i < WIDTH) & (position >= 0) & (block >= 0) & (block < MAP)
    valid &= (block * BS <= position) & (block * BS < total)
    first = tl.min(
        tl.where(
            (block[:, None] == block[None, :]) & (i[None, :] <= i[:, None]),
            i[None, :],
            BW,
        ),
        1,
    )
    own = valid & (first == i)
    private = query * WIDTH + i
    # Keep address order inside each query, so disabling cross-query sharing
    # does not also disable adjacent-range merging in the RDMA implementation.
    keys = tl.sort(tl.where(own, block, 2147483647), descending=False)
    local = tl.sum(((block[None, :] < block[:, None]) & own[None, :]).to(tl.int32), 1)
    origin = query * WIDTH + local
    required = tl.minimum(BS, tl.minimum(position + 1, total) - keys * BS)
    tl.store(Keys + head * CAP + private, keys, i < WIDTH)
    tl.store(
        Required + head * CAP + private,
        tl.where(keys != 2147483647, required, 0),
        i < WIDTH,
    )
    tl.store(
        Occurrences + (query * HEADS + head) * WIDTH + i,
        tl.where(valid, origin, -1),
        i < WIDTH,
    )
    offset = tl.arange(0, BS)
    token = block[:, None] * BS + offset[None, :]
    allowed = valid[:, None] & (token <= position) & (token < total)
    historical = allowed & (token < history)
    tl.atomic_add(Frequencies + head * SORT + token, 1, historical)
    # Canonical is only a source for the final hot update, after every private
    # transfer has completed. It is never used to resolve a consumer miss.
    tl.atomic_min(
        Canonical + head * SORT + token,
        origin[:, None] * BS + offset[None, :],
        historical,
    )
    tl.atomic_add(
        Stats + head * 7 + 4,
        tl.sum(tl.sum((allowed & (token < old)).to(tl.int32), 1), 0),
    )
    tl.atomic_add(
        Stats + head * 7 + 5,
        tl.sum(tl.sum((allowed & (token >= old)).to(tl.int32), 1), 0),
    )
    invalid = (i < WIDTH) & (block >= 0) & (position >= 0) & ~valid
    tl.atomic_add(Stats + head * 7 + 6, tl.sum(invalid.to(tl.int32), 0))


@triton.jit
def _compact_private(
    Keys,
    Required,
    Sorted,
    CompactRequired,
    Mapping,
    Counts,
    CAP: tl.constexpr,
    B: tl.constexpr,
):
    head = tl.program_id(0)
    i = tl.arange(0, B)
    length = tl.load(Required + head * CAP + i, i < CAP, 0)
    valid = (i < CAP) & (length > 0)
    rank = tl.cumsum(valid.to(tl.int32), 0) - 1
    block = tl.load(Keys + head * CAP + i, valid, -1)
    tl.store(Sorted + head * CAP + rank, block, valid)
    tl.store(CompactRequired + head * CAP + rank, length, valid)
    tl.store(Mapping + head * CAP + i, tl.where(valid, rank, -1), i < CAP)
    tl.store(Counts + head, tl.sum(valid.to(tl.int32), 0))


@triton.jit
def _private_occurrences(
    Occurrences,
    Mapping,
    Positions,
    Stats,
    AllIDs,
    AllCounts,
    RowPositions,
    HEADS: tl.constexpr,
    CAP: tl.constexpr,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % HEADS
    i = tl.arange(0, B)
    original = tl.load(Occurrences + row * WIDTH + i, i < WIDTH, -1)
    valid = (i < WIDTH) & (original >= 0) & (tl.load(Stats + head * 7 + 6) == 0)
    compact = tl.load(Mapping + head * CAP + original, valid, -1)
    valid &= compact >= 0
    rank = tl.cumsum(valid.to(tl.int32), 0) - 1
    tl.store(AllIDs + row * WIDTH + rank, head * CAP + compact, valid)
    tl.store(AllCounts + row, tl.sum(valid.to(tl.int32), 0))
    tl.store(RowPositions + row, tl.load(Positions + row // HEADS))


@triton.jit
def _private_stages(
    AllIDs,
    AllCounts,
    Flags,
    Ranks,
    IDs,
    Counts,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
    CHUNK: tl.constexpr,
    STRICT: tl.constexpr,
    BLOCK_READY: tl.constexpr,
):
    row, stage = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, B)
    valid = (i < WIDTH) & (i < tl.load(AllCounts + row))
    physical = tl.load(AllIDs + row * WIDTH + i, valid, -1)
    missing = tl.load(Flags + physical, valid, 0)
    rank = tl.load(Ranks + physical, valid, 0)
    target = tl.where(missing != 0, 1 if BLOCK_READY else 1 + rank // CHUNK, 0)
    if STRICT:
        target = tl.associative_scan(tl.where(valid, target, 0), 0, _stage_maximum)
    selected = valid & (target == stage)
    destination = tl.cumsum(selected.to(tl.int32), 0) - 1
    tl.store(IDs + (stage * ROWS + row) * WIDTH + destination, physical, selected)
    tl.store(Counts + stage * ROWS + row, tl.sum(selected.to(tl.int32), 0))


@triton.jit
def _retain_private_hot(
    Desc,
    Chosen,
    Frequencies,
    Canonical,
    Mapping,
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
    token = tl.load(Chosen + head * HOT + slot)
    selected = tl.load(Frequencies + head * SORT + token) > 0
    original = tl.load(Canonical + head * SORT + token, selected, 0)
    compact = tl.load(Mapping + head * CAP + original // BS, selected, 0)
    source = (head * CAP + compact) * BS + original % BS
    index = row * (HOT + 1) + slot
    location = tl.load(Locations + index)
    d = tl.arange(0, 128)
    values = tl.load(KV + source * 128 + d, selected, 0)
    tl.store(Tags + index, tl.where(selected, token, -1))
    tl.store(Data + location * 128 + d, values, selected)


class QueryPrivatePlanAdapter(ActivePlanAdapter):
    """ActivePlanAdapter interface without sharing missing KV across queries."""

    def geometry(self, query, blocks, fetches, bs, *, map_blocks=None, all_hbm=False):
        w = super().geometry(
            query, blocks, fetches, bs, map_blocks=map_blocks, all_hbm=all_hbm
        )
        cap = len(query) * w["width"]
        physical = w["heads"] * cap
        chunk = (
            physical
            if all_hbm or (self.pool.backend == "rdma" and self.rdma_coalesce)
            else self.chunk_blocks
        )
        w.update(
            per_head=cap,
            physical=physical,
            extent=cap * bs,
            sort_extent=max(w["map_blocks"] * bs, self.cache.hot),
            chunk=chunk,
            stages=1 + math.ceil(physical / chunk),
        )
        return w

    def allocate(self, query, blocks, fetches, bs, **kwargs):
        w = super().allocate(query, blocks, fetches, bs, **kwargs)
        h, cap = w["heads"], w["per_head"]
        w["mapping"] = torch.empty((h, cap), device=query.device, dtype=torch.int32)
        w["private_required"] = torch.empty_like(w["required"])
        w["private_occurrences"] = torch.empty_like(w["all_ids"])
        w["hot_canonical"] = torch.empty_like(w["frequencies"])
        w["persistent_mapping_bytes"] = w["mapping"].numel() * 4
        w["scratch_bytes"] += sum(
            w[k].numel() * w[k].element_size()
            for k in ("private_required", "private_occurrences", "hot_canonical")
        )
        w["reset_bytes"] += w["hot_canonical"].numel() * 4
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
        from cxl_recsys.models.nsa.planning.read_order import READ_ORDERS, reorder_reads

        if consume_order not in ("strict", "resident_first", "block_ready"):
            raise ValueError("Unknown NSA consumption order")
        if read_order not in READ_ORDERS or read_group_size < 1:
            raise ValueError("Invalid NSA read priority configuration")
        h, cap, bs = w["heads"], w["per_head"], w["bs"]
        reset = max(
            h * max(cap, w["sort_extent"], w["hash_capacity"], 7),
            self.pool.region_count,
            w["ids"].numel(),
            w["all_ids"].numel(),
        )
        _reset_workspace[(triton.cdiv(reset, 256),)](
            w["active_counts"],
            w["required"],
            w["frequencies"],
            w["stats"],
            w["region_counts"],
            w["insert_keys"],
            w["hot_hash_keys"],
            w["ids"],
            w["all_ids"],
            h,
            cap,
            w["sort_extent"],
            w["hash_capacity"],
            self.pool.region_count,
            w["ids"].numel(),
            w["all_ids"].numel(),
            256,
            num_warps=4,
        )
        w["hot_canonical"].fill_(2147483647)
        _private_references[(len(blocks), h)](
            blocks,
            positions,
            desc,
            w["insert_keys"],
            w["private_required"],
            w["private_occurrences"],
            w["frequencies"],
            w["hot_canonical"],
            w["stats"],
            h,
            w["width"],
            triton.next_power_of_2(w["width"]),
            w["map_blocks"],
            cap,
            bs,
            w["sort_extent"],
            num_warps=4,
        )
        _compact_private[(h,)](
            w["insert_keys"],
            w["private_required"],
            w["active_keys"],
            w["required"],
            w["mapping"],
            w["active_counts"],
            cap,
            triton.next_power_of_2(cap),
            num_warps=4,
        )
        _hot_hash[(h,)](
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
        _addresses[(cap, h)](
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
        _private_occurrences[(w["rows"],)](
            w["private_occurrences"],
            w["mapping"],
            positions,
            w["stats"],
            w["all_ids"],
            w["all_counts"],
            w["positions"],
            h,
            cap,
            w["width"],
            triton.next_power_of_2(w["width"]),
            num_warps=1,
        )
        reorder_reads(w, read_order, read_group_size)
        if not w["occurrence_only"]:
            _private_stages[(w["rows"], w["stages"])](
                w["all_ids"],
                w["all_counts"],
                w["flags"],
                w["ranks"],
                w["ids"],
                w["counts"],
                w["rows"],
                w["width"],
                triton.next_power_of_2(w["width"]),
                w["chunk"],
                consume_order == "strict",
                consume_order == "block_ready",
                num_warps=1,
            )
        if "hot_partial" in w:
            _, tiles, k = w["hot_partial"].shape
            _hot_tile_top[(h, tiles)](
                w["frequencies"],
                w["hot_partial"],
                w["sort_extent"],
                k,
                tiles,
                w["hot_tile"],
                num_warps=4,
            )
            chosen = w["hot_chosen"]
            _hot_merge_top[(h, tiles)](
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
                h,
                w["meta"].shape[1],
                256,
            )
        return chosen

    def retain_hot(self, desc, chosen, w):
        _retain_private_hot[(w["heads"], self.cache.hot)](
            desc,
            chosen,
            w["frequencies"],
            w["hot_canonical"],
            w["mapping"],
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
