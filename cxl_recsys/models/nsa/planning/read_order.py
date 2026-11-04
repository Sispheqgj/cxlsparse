"""GPU-only ordering of already deduplicated physical reads.

No KV storage, addresses, consumer occurrences or readiness protocol change.
Group numbers count valid token positions, not GQA heads or padded rows.
"""

import torch
import triton
import triton.language as tl

DEPTH_ORDERS = ("consumption_depth", "consumption_depth_shared")
READ_ORDERS = (
    "physical",
    "head_interleaved",
    "query_first_use",
    "query_first_use_region",
    *DEPTH_ORDERS,
)


def allocate_region_order(w, device, regions, page_size):
    """Only scheduling metadata; source slots and target storage remain fixed."""
    if "read_region_ids" not in w:
        w["read_region_ids"] = torch.empty(
            w["physical"], dtype=torch.int32, device=device
        )
        w["read_order_bytes"] += w["physical"] * 4
        w["scratch_bytes"] += w["physical"] * 4
    w["read_region_count"], w["read_page_size"] = regions, page_size


@triton.jit
def _block_regions(
    Slots, Valid, Regions, BS: tl.constexpr, PAGE: tl.constexpr, REGIONS: tl.constexpr
):
    b = tl.program_id(0)
    t = tl.arange(0, BS)
    slot = tl.load(Slots + b * BS + t)
    valid = (t < tl.load(Valid + b)) & (slot >= 0)
    region = (slot // PAGE) % REGIONS
    best, chosen = -1, 0
    # A mixed-source block is assigned to its largest effective-byte source.
    # This changes only priority, never its per-token address or read content.
    for r in tl.static_range(REGIONS):
        count = tl.sum((valid & (region == r)).to(tl.int32), 0)
        chosen = tl.where(count > best, r, chosen)
        best = tl.maximum(best, count)
    tl.store(Regions + b, chosen)


@triton.jit
def _region_priority_ranks(
    Flags,
    First,
    Regions,
    IOIDs,
    Ranks,
    Missing,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    REGIONS: tl.constexpr,
    B: tl.constexpr,
):
    group, device = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, B)
    missing = tl.load(Flags + i, i < N, 0) != 0
    priority = tl.minimum(tl.load(First + i, i < N, GROUPS), GROUPS)
    region = tl.load(Regions + i, i < N, -1)
    selected = missing & (priority == group)
    own = selected & (region == device)
    ordinal = tl.cumsum(own.to(tl.int32)) - 1
    rank = tl.full((B,), 0, tl.int32)
    for other in tl.static_range(REGIONS):
        count = tl.sum((selected & (region == other)).to(tl.int32), 0)
        rank += tl.minimum(count, ordinal)
        rank += ((other < device) & (count > ordinal)).to(tl.int32)
    rank += tl.sum((missing & (priority < group)).to(tl.int32), 0)
    tl.store(IOIDs + rank, i, own)
    tl.store(Ranks + i, rank, own)
    if (group == 0) & (device == 0):
        count = tl.sum(missing.to(tl.int32), 0)
        tl.store(Missing, count)
        tl.store(IOIDs + i, -1, (i < N) & (i >= count))
        tl.store(Ranks + i, -1, (i < N) & ~missing)


def allocate_read_order(w, device, *, head_ids=None, order="query_first_use"):
    """Allocate once, before capture; replay resets every dynamic priority."""
    if order in DEPTH_ORDERS:
        from cxl_recsys.models.nsa.planning.dependencies import (
            allocate_dependency_order,
        )

        allocate_dependency_order(w, device)
        return
    if "read_first_group" in w:
        return
    p, tokens = w["physical"], w["rows"] // w["heads"]
    w["read_first_group"] = torch.empty(p, dtype=torch.int32, device=device)
    w["read_token_group"] = torch.empty(tokens, dtype=torch.int32, device=device)
    # An explicit map also supports compact trace slices with unequal head sizes.
    w["read_head_ids"] = (
        torch.arange(p, device=device, dtype=torch.int32) // w["per_head"]
        if head_ids is None
        else head_ids
    )
    w["read_order_bytes"] = (2 * p + tokens) * 4
    w["scratch_bytes"] = w.get("scratch_bytes", 0) + w["read_order_bytes"]


@triton.jit
def _reset_priority(
    Positions,
    First,
    Groups,
    N: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.arange(0, B)
    tl.store(First + i, TOKENS + 1, i < N)
    pos = tl.load(Positions + i * HEADS, i < TOKENS, -1)
    ordinal = tl.cumsum(((i < TOKENS) & (pos >= 0)).to(tl.int32)) - 1
    tl.store(Groups + i, tl.where(pos >= 0, ordinal // GROUP, -1), i < TOKENS)


@triton.jit
def _first_use(
    IDs,
    Counts,
    Flags,
    Groups,
    First,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    group = tl.load(Groups + row // HEADS)
    valid = (i < WIDTH) & (i < tl.load(Counts + row)) & (group >= 0)
    physical = tl.load(IDs + row * WIDTH + i, valid, -1)
    valid &= physical >= 0
    missing = tl.load(Flags + physical, valid, 0)
    tl.atomic_min(First + physical, group, valid & (missing != 0))


@triton.jit
def _priority_ranks(
    Flags,
    First,
    IOIDs,
    Ranks,
    Missing,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    B: tl.constexpr,
):
    # Each program compacts one priority bucket; outputs are disjoint. A final
    # sentinel bucket preserves even unclaimed flags on an invalid input plan.
    group = tl.program_id(0)
    i = tl.arange(0, B)
    missing = tl.load(Flags + i, i < N, 0) != 0
    priority = tl.minimum(tl.load(First + i, i < N, GROUPS), GROUPS)
    own = missing & (priority == group)
    base = tl.sum((missing & (priority < group)).to(tl.int32), 0)
    rank = base + tl.cumsum(own.to(tl.int32)) - 1
    tl.store(IOIDs + rank, i, own)
    tl.store(Ranks + i, rank, own)
    if group == 0:
        count = tl.sum(missing.to(tl.int32), 0)
        tl.store(Missing, count)
        tl.store(IOIDs + i, -1, (i < N) & (i >= count))
        tl.store(Ranks + i, -1, (i < N) & ~missing)


@triton.jit
def _interleaved_ranks(
    Flags,
    Heads,
    IOIDs,
    Ranks,
    Missing,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    B: tl.constexpr,
):
    h = tl.program_id(0)
    i = tl.arange(0, B)
    missing = tl.load(Flags + i, i < N, 0) != 0
    head = tl.load(Heads + i, i < N, -1)
    own = missing & (head == h)
    ordinal = tl.cumsum(own.to(tl.int32)) - 1
    rank = tl.full((B,), 0, tl.int32)
    for other in tl.static_range(HEADS):
        count = tl.sum((missing & (head == other)).to(tl.int32), 0)
        rank += tl.minimum(count, ordinal)
        rank += ((other < h) & (count > ordinal)).to(tl.int32)
    tl.store(IOIDs + rank, i, own)
    tl.store(Ranks + i, rank, own)
    if h == 0:
        count = tl.sum(missing.to(tl.int32), 0)
        tl.store(Missing, count)
        tl.store(IOIDs + i, -1, (i < N) & (i >= count))
        tl.store(Ranks + i, -1, (i < N) & ~missing)


def reorder_reads(w, order, group_size=16):
    """Call after valid AllIDs/Counts/row positions and actual Flags exist.

    The caller rebuilds rank-dependent batch stages if using a batch consumer;
    block_ready stages do not depend on ranks. No device-to-host operations.
    """
    if order not in READ_ORDERS or group_size < 1:
        raise ValueError("Invalid NSA read order or token supply group size")
    if order == "physical":
        return
    if order in DEPTH_ORDERS:
        from cxl_recsys.models.nsa.planning.dependencies import reorder_dependency_reads

        reorder_dependency_reads(w, shared=order == "consumption_depth_shared")
        return
    n, heads, rows, width = (w[k] for k in ("physical", "heads", "rows", "width"))
    if order in ("query_first_use", "query_first_use_region"):
        _reset_priority[(1,)](
            w["positions"],
            w["read_first_group"],
            w["read_token_group"],
            n,
            rows // heads,
            heads,
            group_size,
            triton.next_power_of_2(max(n, rows // heads)),
            num_warps=4,
        )
        _first_use[(rows,)](
            w["all_ids"],
            w["all_counts"],
            w["flags"],
            w["read_token_group"],
            w["read_first_group"],
            heads,
            width,
            triton.next_power_of_2(width),
            num_warps=1,
        )
        groups = triton.cdiv(rows // heads, group_size)
        if order == "query_first_use_region":
            _block_regions[(n,)](
                w["slots"],
                w["valid"],
                w["read_region_ids"],
                w["bs"],
                w["read_page_size"],
                w["read_region_count"],
                num_warps=1,
            )
            _region_priority_ranks[(groups + 1, w["read_region_count"])](
                w["flags"],
                w["read_first_group"],
                w["read_region_ids"],
                w["ioids"],
                w["ranks"],
                w["missing"],
                n,
                groups,
                w["read_region_count"],
                triton.next_power_of_2(n),
                num_warps=4,
            )
        else:
            _priority_ranks[(groups + 1,)](
                w["flags"],
                w["read_first_group"],
                w["ioids"],
                w["ranks"],
                w["missing"],
                n,
                groups,
                triton.next_power_of_2(n),
                num_warps=4,
            )
    else:
        _interleaved_ranks[(heads,)](
            w["flags"],
            w["read_head_ids"],
            w["ioids"],
            w["ranks"],
            w["missing"],
            n,
            heads,
            triton.next_power_of_2(n),
            num_warps=4,
        )
