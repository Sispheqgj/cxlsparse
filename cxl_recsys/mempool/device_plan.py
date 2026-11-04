"""Device-resident completion of the existing HiSparse layer plan.

Uses bounded capacity buffers and GPU counts. Host validation/accounting is
deferred to the forward boundary, never silently discarded. Invalid rows
produce masked attention IDs and zero IO counts before any payload loads.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class DeferredLayerCounters:
    statistics: torch.Tensor
    launches: int
    backend: str = "cxl"

    def materialize(self, rows):
        if any(row[6] for row in rows):
            raise ValueError(
                "Top-K must contain causal positions or -1 padding, with bound history slots"
            )
        owner = next((i for i, row in enumerate(rows) if row[3]), None)
        return [
            dict(
                hits=r[2],
                misses=r[3],
                old_references=r[4],
                deduplicated_references=r[4] - r[1],
                local_tokens=r[5],
                planner_calls=int(i == 0),
                h2d_transfers=int(self.backend == "dram" and i == owner),
                cxl_read_kernel_launches=(
                    self.launches if self.backend == "cxl" and i == 0 else 0
                ),
            )
            for i, r in enumerate(rows)
        ]


@dataclass
class DeviceLayerKVBatch:
    values: torch.Tensor
    indices: torch.Tensor
    # Capacity, not logical length. Inspection is deliberately explicit.
    logical_capacity: torch.Tensor
    statistics: torch.Tensor
    blocks: torch.Tensor | None = None
    descriptors: torch.Tensor | None = None

    @property
    def sequence_indptr(self):
        counts = self.statistics[:, 0].tolist()
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        return offsets

    @property
    def logical_ids(self):
        counts = self.statistics[:, 0].tolist()
        return torch.cat([row[:n] for row, n in zip(self.logical_capacity, counts)])

    @property
    def logical_positions(self):
        return self.logical_ids.tolist()

    @property
    def cached_tokens(self):
        return int(self.statistics[:, 1].sum().item())


@triton.jit
def _metadata(
    STATS,
    COUNTS,
    DEST,
    FREQUENCIES,
    EXTENT: tl.constexpr,
    BATCH: tl.constexpr,
    BB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    item = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    prev = tl.arange(0, BB)
    offset = tl.sum(
        tl.load(STATS + prev * 7, (prev < row) & (prev < BATCH), other=0), 0
    )
    error = tl.load(STATS + row * 7 + 6)
    misses = tl.load(STATS + row * 7 + 3)
    if tl.program_id(0) == 0:
        tl.store(COUNTS + row, tl.where(error == 0, misses, 0))
    dst = tl.load(DEST + row * EXTENT + item, item < misses, other=0)
    tl.store(DEST + row * EXTENT + item, dst + offset, item < misses)
    freq = tl.load(FREQUENCIES + row * EXTENT + item, item < EXTENT, other=0)
    freq = tl.where(error == 0, freq, 0)
    tl.store(FREQUENCIES + row * EXTENT + item, freq, item < EXTENT)


@triton.jit
def _map(
    IDS,
    DESC,
    RANKS,
    STATS,
    OUT,
    EXTENT: tl.constexpr,
    WIDTH: tl.constexpr,
    BATCH: tl.constexpr,
    BB: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(1)
    item = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(DESC + row * 5 + 2)
    queries = tl.load(DESC + row * 5 + 3)
    token = tl.load(IDS + begin * WIDTH + item, item < queries * WIDTH, other=-1)
    token *= BLOCK_SIZE
    error = tl.load(STATS + row * 7 + 6)
    prev = tl.arange(0, BB)
    offset = tl.sum(
        tl.load(STATS + prev * 7, (prev < row) & (prev < BATCH), other=0), 0
    )
    valid = (error == 0) & (token >= 0) & (token < EXTENT)
    rank = tl.load(RANKS + row * EXTENT + token, valid, other=-1)
    tl.store(
        OUT + begin * WIDTH + item,
        tl.where(valid, offset + rank, -1),
        item < queries * WIDTH,
    )


@triton.jit
def _copy(
    SRC,
    DST,
    STATS,
    VALUES,
    CAPACITY: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    BATCH: tl.constexpr,
    BB: tl.constexpr,
):
    row = tl.program_id(1)
    job = tl.program_id(0) * 4 + tl.arange(0, 4)
    col = tl.arange(0, BD)
    count = tl.load(STATS + row * 7) - tl.load(STATS + row * 7 + 3)
    valid = (job < count) & (tl.load(STATS + row * 7 + 6) == 0)
    address = tl.load(SRC + row * CAPACITY + job, valid, other=0)
    target = tl.load(DST + row * CAPACITY + job, valid, other=0)
    prev = tl.arange(0, BB)
    offset = tl.sum(
        tl.load(STATS + prev * 7, (prev < row) & (prev < BATCH), other=0), 0
    )
    source = address.to(tl.pointer_type(tl.bfloat16))
    data = tl.load(
        source[:, None] + col[None, :], valid[:, None] & (col[None, :] < DIM), other=0
    )
    tl.store(
        VALUES + (offset + target[:, None]) * DIM + col[None, :],
        data,
        valid[:, None] & (col[None, :] < DIM),
    )


def gather(indices, descriptors, work, dim, max_queries, hot, *, block_size=1):
    batch, extent = work["ranks"].shape
    values = torch.empty(
        (batch * extent, dim), dtype=torch.bfloat16, device=indices.device
    )
    mapped = torch.empty_like(indices)
    bb = triton.next_power_of_2(batch)
    _metadata.run(
        work["stats"],
        work["counts"],
        work["destinations"],
        work["frequencies"],
        extent,
        batch,
        bb,
        256,
        grid=(triton.cdiv(extent, 256), batch),
        warmup=False,
    )
    _map.run(
        indices,
        descriptors,
        work["ranks"],
        work["stats"],
        mapped,
        extent,
        indices.shape[1],
        batch,
        bb,
        256,
        block_size,
        grid=(triton.cdiv(max_queries * indices.shape[1], 256), batch),
        warmup=False,
    )
    capacity = work["hbm_src_ptrs"].shape[1]
    _copy.run(
        work["hbm_src_ptrs"],
        work["hbm_dst_local"],
        work["stats"],
        values,
        capacity,
        dim,
        triton.next_power_of_2(dim),
        batch,
        bb,
        grid=(triton.cdiv(capacity, 4), batch),
        warmup=False,
    )
    # Stable int32 sorting keeps lower logical positions first on ties, without
    # the expensive int64 composite-key Top-K. Preserve exact hot-set order.
    selected = torch.argsort(work["frequencies"], dim=1, descending=True, stable=True)[
        :, :hot
    ].contiguous()
    return values, mapped, selected
