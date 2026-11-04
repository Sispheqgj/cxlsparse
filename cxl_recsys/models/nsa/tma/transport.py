"""Adapt the live token-address plan to the staged TMA supply path.

Full contiguous blocks are read directly from registered backing memory with
bulk TMA. Other active blocks gather their resident and external tokens once;
direct and inactive blocks need no HBM payload initialization. No source
pointer, selection or dynamic count is downloaded to the CPU. The surrounding
GR transport owns registration, graph lifetimes, stream ordering and hot commit.
"""

import torch
import triton
import triton.language as tl

from cxl_recsys.models.nsa.tma.pipeline import build_tma_kernel


@triton.jit
def _prepare_sources(Address, Valid, MissingFlags, KV, Sources, Kinds, Ready):
    block = tl.program_id(0)
    token = tl.arange(0, 16)
    dim = tl.arange(0, 128)
    address = tl.load(Address + block * 16 + token)
    first = tl.load(Address + block * 16)
    valid = tl.load(Valid + block)
    missing = tl.load(MissingFlags + block) != 0
    contiguous = tl.sum((address == first + token * 256).to(tl.int32), 0) == 16
    full = missing & (valid == 16) & (first != 0) & (first % 16 == 0) & contiguous
    destination = KV + block * 2048
    tl.store(Sources + block, tl.where(full, first, destination.to(tl.int64)))
    # 0: resident, 1: direct full-block TMA, 2: mixed/tail external gather.
    tl.store(Kinds + block, tl.where(missing, tl.where(full, 1, 2), 0))
    if (valid > 0) & ~full:
        values = tl.load(
            address[:, None].to(tl.pointer_type(tl.bfloat16)) + dim[None, :],
            ((token < valid) & (address != 0))[:, None],
            0,
        )
        # Initialize the tail too: subsequent copies move a complete block.
        tl.store(destination + token[:, None] * 128 + dim[None, :], values)
    # The following fused launch observes both payload and flags through its
    # stream/Graph dependency. Mixed blocks retain their existing IO queue entry.
    tl.store(Ready + block, tl.where(missing, 0, 1))


def configure_workspace(query, work, *, ready_window=1):
    """Bound the cooperative grid independently of the number of query rows."""
    if query.shape[2:] != (16, 64) or work["bs"] != 16:
        raise ValueError("The TMA worker adapter requires G16, D64 and BS16")
    rows, blocks = work["rows"], work["physical"]
    clusters = (rows + 2) // 3

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=query.device)

    work.update(
        tma_sources=empty((blocks,), torch.int64),
        tma_source_kinds=empty((blocks,), torch.int32),
        tma_ready=empty((blocks,), torch.int32),
        tma_requests=empty((clusters, 3, 2), torch.int32),
        tma_done=empty((clusters,), torch.int32),
        tma_maps=empty((2, 128), torch.uint8),
        tma_counters=empty((clusters, 3), torch.int64),
        consumer_ctas=min(rows, 256),
        io_ctas=min(blocks, 24),
        ready_window=ready_window,
        ready_prepared=True,
    )
    work["kernel"] = build_tma_kernel(
        rows,
        blocks,
        work["width"],
        route="staged",
        stages=2,
        io_ctas=24,
        copy_kind="bulk",
        consumer_ctas=256,
        resident_ready=True,
        ready_window=ready_window,
        ready_prepared=True,
    )


def execute(query, work):
    """Prepare payload and readiness after the planner has resolved every source."""
    _prepare_sources[(work["physical"],)](
        work["address"],
        work["valid"],
        work["flags"],
        work["kv"],
        work["tma_sources"],
        work["tma_source_kinds"],
        work["tma_ready"],
        num_warps=4,
    )
    work["kernel"](
        query.flatten(0, 1),
        work["kv"],
        work["all_ids"],
        work["all_counts"],
        work["starts"],
        work["positions"],
        work["tma_sources"],
        work["flags"],
        work["tma_ready"],
        work["ioids"],
        work["missing"],
        work["tma_requests"],
        work["tma_done"],
        work["tma_maps"],
        work["tma_counters"],
        work["output"],
    )
