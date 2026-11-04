"""Original-order consumption rows for block-ready TMA consumers."""

import triton
import triton.language as tl


@triton.jit
def _occurrences(
    Blocks,
    Positions,
    Mapping,
    Stats,
    IDs,
    Counts,
    RowPositions,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BW: tl.constexpr,
    MAP: tl.constexpr,
    CAP: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0)
    query, head = row // HEADS, row % HEADS
    index = tl.arange(0, BW)
    block = tl.load(Blocks + row * WIDTH + index, index < WIDTH, -1)
    position = tl.load(Positions + query)
    valid = (index < WIDTH) & (position >= 0) & (block >= 0) & (block < MAP)
    valid &= (block * BS <= position) & (tl.load(Stats + head * 7 + 6) == 0)
    compact = tl.load(Mapping + head * (MAP + 1) + block, valid, -1)
    valid &= (compact >= 0) & (compact < CAP)
    output = tl.cumsum(valid.to(tl.int32), 0) - 1
    tl.store(IDs + row * WIDTH + output, head * CAP + compact, valid)
    tl.store(Counts + row, tl.sum(valid.to(tl.int32), 0))
    tl.store(RowPositions + row, position)


def build_occurrences(blocks, positions, work):
    """Retain every valid occurrence without constructing readiness stages."""
    _occurrences[(work["rows"],)](
        blocks,
        positions,
        work["mapping"],
        work["stats"],
        work["all_ids"],
        work["all_counts"],
        work["positions"],
        work["heads"],
        work["width"],
        triton.next_power_of_2(work["width"]),
        work["map_blocks"],
        work["per_head"],
        work["bs"],
        num_warps=1,
    )
