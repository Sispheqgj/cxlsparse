"""Static read priorities for the bounded staged-TMA consumer grid.

Depth counts every consumed block, including resident blocks, along one CTA's
serial rows. Sharing counts distinct logical consumer rows within four steps of
a block's earliest depth; rows remain distinct when they reuse the same CTA.
Only IOIDs/Ranks change. Source addresses, missing membership and attention's
original occurrence order are preserved.
"""

import torch
import triton
import triton.language as tl

NEAR_WINDOW = 4


def allocate_dependency_order(w, device):
    """Allocate fixed-shape metadata before capture; CTA count is needed later."""
    if "dependency_sort_keys" in w:
        return
    n, rows = w["physical"], w["rows"]
    for name, size, dtype in (
        ("dependency_row_depth", rows, torch.int32),
        ("dependency_first_depth", n, torch.int32),
        ("dependency_near_consumers", n, torch.int32),
        ("dependency_sort_keys", n, torch.int64),
        ("dependency_sorted_keys", n, torch.int64),
        ("dependency_sort_indices", n, torch.int64),
    ):
        w[name] = torch.empty(size, dtype=dtype, device=device)
    size_bytes = 4 * rows + 32 * n
    w["dependency_order_bytes"] = size_bytes
    w["read_order_bytes"] = w.get("read_order_bytes", 0) + size_bytes
    w["scratch_bytes"] = w.get("scratch_bytes", 0) + size_bytes


@triton.jit
def _reset_dependency(
    First, Near, N: tl.constexpr, SENTINEL: tl.constexpr, B: tl.constexpr
):
    block = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(First + block, SENTINEL, block < N)
    tl.store(Near + block, 0, block < N)


@triton.jit
def _first_depth(
    IDs,
    Counts,
    Positions,
    Flags,
    Depth,
    First,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    CONSUMERS: tl.constexpr,
    ROW_STEPS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    previous = row % CONSUMERS + tl.arange(0, ROW_STEPS) * CONSUMERS
    valid_previous = (previous < row) & (
        tl.load(Positions + previous, previous < row, -1) >= 0
    )
    # Count resident steps too; padded rows never advance the CTA's clock.
    depth = tl.sum(tl.load(Counts + previous, valid_previous, 0), 0)
    tl.store(Depth + row, depth)
    col = tl.arange(0, B)
    valid = (
        (col < WIDTH) & (col < tl.load(Counts + row)) & (tl.load(Positions + row) >= 0)
    )
    block = tl.load(IDs + row * WIDTH + col, valid, -1)
    valid &= (block >= 0) & (block < N)
    missing = tl.load(Flags + block, valid, 0) != 0
    tl.atomic_min(First + block, depth + col, valid & missing)


@triton.jit
def _near_consumers(
    IDs,
    Counts,
    Positions,
    Flags,
    Depth,
    First,
    Near,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    valid = (
        (col < WIDTH) & (col < tl.load(Counts + row)) & (tl.load(Positions + row) >= 0)
    )
    block = tl.load(IDs + row * WIDTH + col, valid, -1)
    valid &= (block >= 0) & (block < N)
    missing = tl.load(Flags + block, valid, 0) != 0
    first = tl.load(First + block, valid, 0)
    depth = tl.load(Depth + row) + col
    near = valid & missing & (depth >= first) & (depth < first + WINDOW)
    unique = tl.full((B,), True, tl.int1)
    # Any earlier occurrence of this block in this row must be at most
    # WINDOW-1 steps back: otherwise it would lower First beyond this window.
    # Checking only that bounded prefix counts each logical row once.
    for previous in tl.static_range(1, WINDOW):
        prior = tl.load(
            IDs + row * WIDTH + col - previous, near & (col >= previous), -1
        )
        unique &= prior != block
    tl.atomic_add(Near + block, 1, near & unique)


@triton.jit
def _priority_keys(
    Flags,
    First,
    Near,
    block,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHARED: tl.constexpr,
    INACTIVE_BASE: tl.constexpr,
    KEY32: tl.constexpr,
):
    dtype: tl.constexpr = tl.int32 if KEY32 else tl.int64
    valid = block < N
    missing = tl.load(Flags + block, valid, 0) != 0
    first = tl.load(First + block, valid, 0).to(dtype)
    shared = tl.load(Near + block, valid, 0) if SHARED else 0
    # Unique physical tail keys make stability independent of the sort backend.
    key = (first * (ROWS + 1) + ROWS - shared) * N + block
    inactive = INACTIVE_BASE + block.to(dtype)
    padding: tl.constexpr = 0x7FFFFFFF if KEY32 else 0x7FFFFFFFFFFFFFFF
    return tl.where(valid, tl.where(missing, key, inactive), padding)


@triton.jit
def _pack_keys(
    Flags,
    First,
    Near,
    Keys,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHARED: tl.constexpr,
    INACTIVE_BASE: tl.constexpr,
    B: tl.constexpr,
):
    block = tl.program_id(0) * B + tl.arange(0, B)
    key = _priority_keys(
        Flags, First, Near, block, N, ROWS, SHARED, INACTIVE_BASE, False
    )
    tl.store(Keys + block, key, block < N)


@triton.jit
def _sort_publish(
    Flags,
    First,
    Near,
    IOIDs,
    Ranks,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHARED: tl.constexpr,
    INACTIVE_BASE: tl.constexpr,
    KEY32: tl.constexpr,
    B: tl.constexpr,
):
    rank = tl.arange(0, B)
    key = _priority_keys(
        Flags, First, Near, rank, N, ROWS, SHARED, INACTIVE_BASE, KEY32
    )
    key = tl.sort(key, descending=False)
    # Both missing and inactive keys encode the physical block as key % N.
    block = key % N
    missing = key < INACTIVE_BASE
    tl.store(IOIDs + rank, tl.where(missing, block, -1), rank < N)
    tl.store(Ranks + block, tl.where(missing, rank, -1), rank < N)


@triton.jit
def _publish_order(Order, Flags, IOIDs, Ranks, N: tl.constexpr, B: tl.constexpr):
    rank = tl.program_id(0) * B + tl.arange(0, B)
    valid = rank < N
    block = tl.load(Order + rank, valid, 0)
    missing = tl.load(Flags + block, valid, 0) != 0
    tl.store(IOIDs + rank, tl.where(missing, block, -1), valid)
    tl.store(Ranks + block, tl.where(missing, rank, -1), valid)


def reorder_dependency_reads(w, *, shared=False):
    """Rebuild static priorities after AllIDs/Counts/Positions and Flags exist.

    Requires the actual staged-TMA ``consumer_ctas``; no implicit legacy-grid
    assumption is made. ``missing`` is unchanged because Flags membership is
    unchanged. All reset, reduction and sorting work stays on the GPU; sort
    outputs are allocated once, so replay needs no data-dependent host work.
    """
    n, rows, width = (w[k] for k in ("physical", "rows", "width"))
    consumers = w["consumer_ctas"]
    # No CTA can consume more than this many blocks, including resident steps.
    max_depth = triton.cdiv(rows, consumers) * width
    grid = (triton.cdiv(n, 256),)
    _reset_dependency[grid](
        w["dependency_first_depth"],
        w["dependency_near_consumers"],
        n,
        max_depth,
        256,
        num_warps=4,
    )
    args = (
        w["all_ids"],
        w["all_counts"],
        w["positions"],
        w["flags"],
        w["dependency_row_depth"],
        w["dependency_first_depth"],
    )
    _first_depth[(rows,)](
        *args,
        n,
        width,
        consumers,
        triton.next_power_of_2(triton.cdiv(rows, consumers)),
        triton.next_power_of_2(width),
        num_warps=1,
    )
    if not shared and n <= 4096 and max_depth <= 64:
        from cxl_recsys.models.nsa.planning.read_order import _priority_ranks

        # Compact the small depth buckets in parallel, preserving physical order
        # within each bucket. The sentinel bucket retains unreferenced flags.
        _priority_ranks[(max_depth + 1,)](
            w["flags"],
            w["dependency_first_depth"],
            w["ioids"],
            w["ranks"],
            w["missing"],
            n,
            max_depth,
            triton.next_power_of_2(n),
            num_warps=4,
        )
        return
    if shared:
        _near_consumers[(rows,)](
            *args,
            w["dependency_near_consumers"],
            n,
            width,
            NEAR_WINDOW,
            triton.next_power_of_2(width),
            num_warps=1,
        )
    inactive_base = ((max_depth + 1) * (rows + 1) + rows) * n
    if n <= 4096:
        _sort_publish[(1,)](
            w["flags"],
            w["dependency_first_depth"],
            w["dependency_near_consumers"],
            w["ioids"],
            w["ranks"],
            n,
            rows,
            shared,
            inactive_base,
            inactive_base + n - 1 < 2**31,
            triton.next_power_of_2(n),
            num_warps=8 if n > 2048 else 4,
        )
        return
    _pack_keys[grid](
        w["flags"],
        w["dependency_first_depth"],
        w["dependency_near_consumers"],
        w["dependency_sort_keys"],
        n,
        rows,
        shared,
        inactive_base,
        256,
        num_warps=4,
    )
    torch.sort(
        w["dependency_sort_keys"],
        out=(w["dependency_sorted_keys"], w["dependency_sort_indices"]),
    )
    _publish_order[grid](
        w["dependency_sort_indices"],
        w["flags"],
        w["ioids"],
        w["ranks"],
        n,
        256,
        num_warps=4,
    )
