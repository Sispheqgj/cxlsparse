"""GPU block plans for mapped CXL and CPU-posted GPUDirect RDMA.

Logical block slots have bounded capacity. The GPU builds a missing-block
permutation and stage IDs without downloading selections, positions or hot
tags. CXL counts stay on device until the forward accounting boundary. RDMA
downloads one packed address plan; payloads never pass through CPU memory.
"""

import math
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch
import triton
import triton.language as tl

from cxl_recsys.mempool.device_plan import DeferredLayerCounters
from cxl_recsys.models.cuda_graph import CUDAGraphCache
from cxl_recsys.models.nsa.backends.gpu_fused import gpu_fused_kernel
from cxl_recsys.models.nsa.backends.legacy_transport import _copy_hbm
from cxl_recsys.models.nsa.backends.tilelang import build_kernel


@triton.jit
def _lookup(
    Tags,
    Locations,
    Desc,
    Lookup,
    HOT: tl.constexpr,
    EXTENT: tl.constexpr,
    BH: tl.constexpr,
):
    head = tl.program_id(0)
    slot = tl.arange(0, BH)
    row = tl.load(Desc + head * 5)
    old = tl.load(Desc + head * 5 + 1)
    token = tl.load(Tags + row * (HOT + 1) + slot, slot < HOT, other=-1)
    loc = tl.load(Locations + row * (HOT + 1) + slot, slot < HOT, other=0)
    tl.store(Lookup + head * EXTENT + token, loc, (token >= 0) & (token < old))


@triton.jit
def _references(
    Blocks,
    Positions,
    Desc,
    Required,
    Frequencies,
    Stats,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BW: tl.constexpr,
    BLOCKS: tl.constexpr,
    BS: tl.constexpr,
    EXTENT: tl.constexpr,
):
    query, head = tl.program_id(0), tl.program_id(1)
    slot = tl.arange(0, BW)
    token_in_block = tl.arange(0, BS)
    position = tl.load(Positions + query)
    old = tl.load(Desc + head * 5 + 1)
    history = tl.load(Desc + head * 5 + 2)
    total = old + tl.load(Desc + head * 5 + 4)
    block = tl.load(
        Blocks + (query * HEADS + head) * WIDTH + slot, slot < WIDTH, other=-1
    )
    valid = (block >= 0) & (block < BLOCKS) & (block * BS <= position)
    required = tl.minimum(BS, position + 1 - block * BS)
    tl.atomic_max(Required + head * BLOCKS + block, required, valid)
    token = block[:, None] * BS + token_in_block[None, :]
    allowed = valid[:, None] & (token <= position) & (token < total)
    tl.atomic_add(Frequencies + head * EXTENT + token, 1, allowed & (token < history))
    old_refs = tl.sum(tl.sum((allowed & (token < old)).to(tl.int32), 1), 0)
    local_refs = tl.sum(tl.sum((allowed & (token >= old)).to(tl.int32), 1), 0)
    tl.atomic_add(Stats + head * 7 + 4, old_refs)
    tl.atomic_add(Stats + head * 7 + 5, local_refs)
    error = tl.sum(
        (
            (slot < WIDTH)
            & (block >= 0)
            & ((block >= BLOCKS) | (block * BS > position))
        ).to(tl.int32),
        0,
    )
    tl.atomic_add(Stats + head * 7 + 6, error)


@triton.jit
def _addresses(
    Required,
    Lookup,
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
    BLOCKS: tl.constexpr,
    BS: tl.constexpr,
    EXTENT: tl.constexpr,
    MAX_HISTORY: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    REGIONS: tl.constexpr,
    LOCAL: tl.constexpr,
):
    block, head = tl.program_id(0), tl.program_id(1)
    physical = head * BLOCKS + block
    token = block * BS + tl.arange(0, BS)
    required = tl.load(Required + physical)
    row = tl.load(Desc + head * 5)
    old = tl.load(Desc + head * 5 + 1)
    pointer = tl.load(Desc + head * 5 + 3)
    valid = token < block * BS + required
    location = tl.load(Lookup + head * EXTENT + token, valid, other=-1)
    previous = valid & (token < old)
    external = previous & (location < 0)
    source = tl.load(
        HostSlots + row * MAX_HISTORY + token,
        external & (token < MAX_HISTORY),
        other=-1,
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
        base = tl.load(Bases + region, safe_external, other=0)
        source_address = base + local_slot * 256
        address = tl.where(safe_external, source_address, hbm_address)
        for r in tl.static_range(REGIONS):
            count = tl.sum((safe_external & (region == r)).to(tl.int32), 0)
            tl.atomic_add(RegionCounts + r, count)
    else:
        address = hbm_address
    tl.store(Address + physical * BS + tl.arange(0, BS), address)
    tl.store(HBM + physical * BS + tl.arange(0, BS), hbm_address)
    tl.store(
        Slots + physical * BS + tl.arange(0, BS), tl.where(safe_external, source, -1)
    )
    errors = tl.sum(error.to(tl.int32), 0)
    tl.store(Valid + physical, tl.where(errors == 0, required, 0))
    tl.store(Starts + physical, block * BS)
    tl.store(MissingFlags + physical, tl.sum(external.to(tl.int32), 0) > 0)
    tl.atomic_add(Stats + head * 7, tl.sum(valid.to(tl.int32), 0))
    tl.atomic_add(Stats + head * 7 + 1, tl.sum(previous.to(tl.int32), 0))
    tl.atomic_add(Stats + head * 7 + 2, tl.sum((previous & ~external).to(tl.int32), 0))
    tl.atomic_add(Stats + head * 7 + 3, tl.sum(external.to(tl.int32), 0))
    tl.atomic_add(Stats + head * 7 + 6, errors)


@triton.jit
def _ranks(Flags, IOIDs, IORanks, Missing, N: tl.constexpr, BN: tl.constexpr):
    physical = tl.arange(0, BN)
    missing = tl.load(Flags + physical, physical < N, other=0)
    rank = tl.cumsum(missing, 0) - 1
    tl.store(IOIDs + rank, physical, (physical < N) & (missing != 0))
    tl.store(IORanks + physical, rank, physical < N)
    tl.store(Missing, tl.sum(missing, 0))


@triton.jit
def _stages(
    Blocks,
    Positions,
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
    BLOCKS: tl.constexpr,
    BS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row, stage = tl.program_id(0), tl.program_id(1)
    query, head = row // HEADS, row % HEADS
    slot = tl.arange(0, BW)
    block = tl.load(Blocks + row * WIDTH + slot, slot < WIDTH, other=-1)
    position = tl.load(Positions + query)
    physical = head * BLOCKS + block
    valid = (slot < WIDTH) & (block >= 0) & (block < BLOCKS) & (block * BS <= position)
    valid &= tl.load(Stats + head * 7 + 6) == 0
    missing = tl.load(Flags + physical, valid, other=0)
    rank = tl.load(Ranks + physical, valid, other=0)
    target_stage = tl.where(missing != 0, 1 + rank // CHUNK, 0)
    chosen = valid & (stage == target_stage)
    rank_in_row = tl.cumsum(chosen.to(tl.int32), 0) - 1
    tl.store(IDs + (stage * ROWS + row) * WIDTH + rank_in_row, physical, chosen)
    tl.store(Counts + stage * ROWS + row, tl.sum(chosen.to(tl.int32), 0))
    if stage == 0:
        tl.store(RowPositions + row, position)
        all_rank = tl.cumsum(valid.to(tl.int32), 0) - 1
        tl.store(AllIDs + row * WIDTH + all_rank, physical, valid)
        tl.store(AllCounts + row, tl.sum(valid.to(tl.int32), 0))


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
    valid = (token < N) & (tl.load(Flags + physical, token < N, other=0) != 0)
    rank = tl.load(Ranks + physical, valid, other=0)
    source = tl.load(Slots + token, valid, other=-1)
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
    KV,
    Tags,
    Locations,
    Data,
    EXTENT: tl.constexpr,
    HOT: tl.constexpr,
):
    head, slot = tl.program_id(0), tl.program_id(1)
    row = tl.load(Desc + head * 5)
    token = tl.load(Chosen + head * HOT + slot)
    selected = tl.load(Frequencies + head * EXTENT + token) > 0
    index = row * (HOT + 1) + slot
    location = tl.load(Locations + index)
    d = tl.arange(0, 128)
    values = tl.load(KV + (head * EXTENT + token) * 128 + d, selected, other=0)
    tl.store(Tags + index, tl.where(selected, token, -1))
    tl.store(Data + location * 128 + d, values, selected)


class GPUSelectedTransport:
    """Serialized bounded plans; capture only scratch mutation and local reads."""

    def __init__(
        self,
        cache,
        *,
        mode="overlap",
        chunk_blocks=64,
        io_ctas=64,
        cuda_graphs=True,
        capacity=4,
        rdma_coalesce=True,
    ):
        if mode not in ("serial", "overlap") or min(chunk_blocks, io_ctas) < 1:
            raise ValueError("Invalid GPU selected transport configuration")
        self.cache, self.pool = cache, cache.pool
        self.mode, self.chunk_blocks, self.io_ctas = mode, chunk_blocks, io_ctas
        self.graphs = CUDAGraphCache(capacity) if cuda_graphs else None
        self.capacity = capacity
        self.rdma_coalesce = rdma_coalesce
        self.attention_graphs = CUDAGraphCache(capacity) if cuda_graphs else None
        self.workspaces = OrderedDict()
        self.descriptors = OrderedDict()
        self.local_bases = None
        self.producer = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="nsa-rdma")
            if self.pool.backend == "rdma"
            else None
        )
        self.io_stream = torch.cuda.Stream(device=cache.data.device)
        self.last_done = torch.cuda.Event()
        self.submitted = False
        self.calls = 0
        self.last_stats = {}

    def close(self):
        if self.producer is not None:
            self.producer.shutdown(wait=True)
        if self.submitted:
            self.last_done.synchronize()
        self.io_stream.synchronize()
        if self.graphs is not None:
            self.graphs.close()
        if self.attention_graphs is not None:
            self.attention_graphs.close()
        for work in self.workspaces.values():
            if work.get("registration") is not None:
                work["registration"].close()
        self.workspaces.clear()
        self.descriptors.clear()

    def _workspace(self, query, blocks, fetches, bs):
        heads, width = len(fetches), blocks.shape[-1]
        per_head = math.ceil(
            max(f["cached_length"] + len(f["current_kv"]) for f in fetches) / bs
        )
        # The stable hot sort may include zero-frequency padding entries.
        per_head = max(per_head, math.ceil(self.cache.hot / bs))
        key = (len(query), heads, per_head, bs, width, query.shape[2])
        if key in self.workspaces:
            self.workspaces.move_to_end(key)
            return key, self.workspaces[key]
        if len(self.workspaces) >= self.capacity:
            if self.graphs is not None:
                self.graphs.close()
            if self.attention_graphs is not None:
                self.attention_graphs.close()
            if self.submitted:
                self.last_done.synchronize()
            _, previous = self.workspaces.popitem(last=False)
            if previous.get("registration") is not None:
                previous["registration"].close()
        extent, physical, rows = per_head * bs, per_head * heads, len(query) * heads
        chunk = (
            physical
            if self.pool.backend == "rdma" and self.rdma_coalesce
            else self.chunk_blocks
        )
        stages = 1 + math.ceil(physical / chunk)
        device = query.device

        def empty(shape, dtype=torch.int32):
            return torch.empty(shape, dtype=dtype, device=device)

        work = dict(
            required=empty((heads, per_head)),
            lookup=empty((heads, extent)),
            frequencies=empty((heads, extent)),
            stats=empty((heads, 7)),
            address=empty((physical, bs), torch.int64),
            hbm=empty((physical, bs), torch.int64),
            valid=empty((physical,)),
            starts=empty((physical,)),
            flags=empty((physical,)),
            slots=empty((physical * bs,), torch.int64),
            ioids=empty((physical,)),
            ranks=empty((physical,)),
            missing=empty((1,)),
            ids=empty((stages, rows, width)),
            counts=empty((stages, rows)),
            all_ids=empty((rows, width)),
            all_counts=empty((rows,)),
            positions=empty((rows,)),
            region_counts=empty((self.pool.region_count,)),
            kv=empty((physical, bs, 128), torch.bfloat16),
            output=torch.empty_like(query.contiguous().flatten(0, 1)),
            barriers=empty((stages,)),
            compute_trace=empty((rows, stages, 2), torch.uint64),
            io_trace=empty((physical, 2), torch.uint64),
            heads=heads,
            per_head=per_head,
            extent=extent,
            bs=bs,
            rows=rows,
            width=width,
            physical=physical,
            stages=stages,
            chunk=chunk,
        )
        if self.pool.backend == "rdma":
            meta_width = 1 + physical * bs + heads * 7
            work["meta"] = empty((2, meta_width), torch.int64)
            work["meta_host"] = torch.empty(
                (2, meta_width), dtype=torch.int64, pin_memory=True
            )
            work["rdma_values"] = work["kv"].flatten(0, 1)
            work["registration"] = self.pool.remote.region.register_cuda(
                work["rdma_values"]
            )
            work["kernels"] = {
                (initial, final): build_kernel(
                    rows,
                    query.shape[2],
                    physical,
                    bs,
                    width,
                    initialize=initial,
                    finalize=final,
                )
                for initial, final in (
                    (True, True),
                    (True, False),
                    (False, False),
                    (False, True),
                )
            }
            work["maximum"] = empty((rows, query.shape[2]), torch.float32)
            work["denominator"] = torch.empty_like(work["maximum"])
            work["accumulator"] = torch.empty_like(work["output"], dtype=torch.float32)
        else:
            work["kernel"] = gpu_fused_kernel(
                rows,
                query.shape[2],
                physical,
                bs,
                width,
                self.chunk_blocks,
                io_ctas=self.io_ctas,
                pipelined=self.mode == "overlap",
                manual_sync=True,
                address_table=True,
            )
        self.workspaces[key] = work
        return key, work

    def _plan(self, query, blocks, positions, desc, bases, w):
        heads, per_head, bs, extent = w["heads"], w["per_head"], w["bs"], w["extent"]
        for name in ("required", "frequencies", "stats", "region_counts"):
            w[name].zero_()
        w["lookup"].fill_(-1)
        w["ids"].fill_(-1)
        w["all_ids"].fill_(-1)
        _lookup[(heads,)](
            self.cache.tags,
            self.cache.locs,
            desc,
            w["lookup"],
            self.cache.hot,
            extent,
            triton.next_power_of_2(self.cache.hot),
        )
        _references[(len(query), heads)](
            blocks,
            positions,
            desc,
            w["required"],
            w["frequencies"],
            w["stats"],
            heads,
            w["width"],
            triton.next_power_of_2(w["width"]),
            per_head,
            bs,
            extent,
            num_warps=4,
        )
        _addresses[(per_head, heads)](
            w["required"],
            w["lookup"],
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
            per_head,
            bs,
            extent,
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
        _stages[(w["rows"], w["stages"])](
            blocks,
            positions,
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
            per_head,
            bs,
            w["chunk"],
            num_warps=1,
        )
        _copy_hbm[(w["physical"] * bs,)](w["hbm"], w["kv"], w["physical"] * bs)
        chosen = torch.argsort(w["frequencies"], dim=1, descending=True, stable=True)
        chosen = chosen[:, : self.cache.hot].contiguous()
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
        else:
            w["kernel"](
                query.contiguous().flatten(0, 1),
                w["kv"],
                w["ids"],
                w["counts"],
                w["starts"],
                w["positions"],
                w["valid"],
                w["address"],
                w["barriers"],
                w["ioids"],
                w["missing"],
                w["output"],
                w["compute_trace"],
                w["io_trace"],
            )
        return w["output"], chosen

    def _rdma(self, query, w, key):
        stream = torch.cuda.current_stream(query.device)
        w["meta_host"].copy_(w["meta"], non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(stream)
        ready.synchronize()
        metadata = w["meta_host"].numpy()
        missing = int(metadata[0, 0])
        counters = DeferredLayerCounters(w["stats"], 0, "rdma").materialize(
            metadata[0, 1 + w["physical"] * w["bs"] :].reshape(w["heads"], 7).tolist()
        )
        chunks = math.ceil(missing / w["chunk"])
        if self.rdma_coalesce:
            sources, destinations = metadata[:, 1 : 1 + missing * w["bs"]]
            valid = sources >= 0
            with torch.cuda.nvtx.range("nsa.rdma.coalesced"):
                self.pool.read_rdma_cuda(
                    sources[valid],
                    w["rdma_values"],
                    destinations[valid],
                    registration=w["registration"],
                )

            def attention(q):
                w["kernels"][(True, True)](
                    q,
                    w["kv"],
                    w["all_ids"],
                    w["starts"],
                    w["positions"],
                    w["all_counts"],
                    w["maximum"],
                    w["denominator"],
                    w["accumulator"],
                    w["output"],
                )
                return (w["output"],)

            with torch.cuda.nvtx.range("nsa.selected.tilelang.coalesced"):
                q = query.contiguous().flatten(0, 1)
                if self.attention_graphs is None:
                    attention(q)
                else:
                    self.attention_graphs.run(key, attention, (q,), clone_outputs=False)
            return counters
        events = [torch.cuda.Event() for _ in range(chunks)]
        arrived = [threading.Event() for _ in range(chunks)]
        failure = []

        def produce():
            try:
                with torch.cuda.device(query.device), torch.cuda.stream(self.io_stream):
                    self.io_stream.wait_event(ready)
                    for i in range(chunks):
                        lo = 1 + i * w["chunk"] * w["bs"]
                        hi = 1 + min(missing, (i + 1) * w["chunk"]) * w["bs"]
                        sources, destinations = metadata[:, lo:hi]
                        valid = sources >= 0
                        with torch.cuda.nvtx.range(f"nsa.rdma.chunk.{i}"):
                            self.pool.read_rdma_cuda(
                                sources[valid],
                                w["rdma_values"],
                                destinations[valid],
                                registration=w["registration"],
                            )
                        events[i].record(self.io_stream)
                        arrived[i].set()
            except BaseException as exc:
                failure.append(exc)
                for ticket in arrived:
                    ticket.set()
                raise

        future = self.producer.submit(produce) if chunks else None

        def wait(i):
            arrived[i].wait()
            if failure:
                future.result()
            stream.wait_event(events[i])

        def attend(i):
            kernel = w["kernels"][(i == 0, i == chunks)]
            with torch.cuda.nvtx.range(
                "nsa.selected.tilelang.resident"
                if i == 0
                else f"nsa.selected.tilelang.chunk.{i - 1}"
            ):
                kernel(
                    query.contiguous().flatten(0, 1),
                    w["kv"],
                    w["ids"][i],
                    w["starts"],
                    w["positions"],
                    w["counts"][i],
                    w["maximum"],
                    w["denominator"],
                    w["accumulator"],
                    w["output"],
                )

        try:
            if chunks and self.mode == "serial":
                wait(chunks - 1)
            attend(0)
            for i in range(chunks):
                wait(i)
                attend(i + 1)
        finally:
            if future is not None:
                future.result()
        return counters

    def run(self, query, blocks, positions, fetches, block_size, *, trace=False):
        if trace:
            raise ValueError(
                "Use unprofiled CUDA event timelines for the GPU transport"
            )
        if (
            blocks.shape[:2] != query.shape[:2]
            or len(fetches) != query.shape[1]
            or blocks.dtype != torch.int32
            or not blocks.is_contiguous()
        ):
            raise ValueError("Invalid GPU block selection geometry")
        begin = time.perf_counter_ns()
        stream = torch.cuda.current_stream(query.device)
        if self.submitted:
            stream.wait_event(self.last_done)
        if self.cache.has_last_use:
            stream.wait_event(self.cache.last_use)
        key, w = self._workspace(query, blocks, fetches, block_size)
        layout = tuple(
            (
                f["row"],
                f["cached_length"],
                f["history_length"],
                f["current_kv"].data_ptr(),
                len(f["current_kv"]),
            )
            for f in fetches
        )
        descriptors = self.descriptors.get(layout)
        if descriptors is None:
            descriptors = torch.tensor(layout, dtype=torch.int64, pin_memory=True).to(
                query.device, non_blocking=True
            )
            self.descriptors[layout] = descriptors
            if len(self.descriptors) > 128:
                self.descriptors.popitem(last=False)
        else:
            self.descriptors.move_to_end(layout)
        descriptors.record_stream(stream)
        if self.pool.backend == "rdma":
            bases = w["address"].flatten()[:1]  # unused by the RDMA specialization
        elif self.pool.cxl is not None:
            self.pool.cxl._initialize_cuda(query.device)
            bases = self.pool.cxl.region_bases
        else:
            if self.local_bases is None:
                self.local_bases = torch.tensor(
                    [self.pool._cuda_mapping(query.device).address],
                    dtype=torch.int64,
                    device=query.device,
                )
            bases = self.local_bases
        for f in fetches:
            f["current_kv"].record_stream(stream)
        inputs = (query, blocks, positions, descriptors, bases)
        for tensor in (query, blocks, positions, descriptors, bases):
            tensor.record_stream(stream)
        with torch.cuda.nvtx.range("nsa.selected.gpu_plan_and_io"):

            def function(*args):
                return self._plan(*args, w)

            output, chosen = (
                self.graphs.run(key, function, inputs, clone_outputs=False)
                if self.graphs is not None
                else function(*inputs)
            )
        prepared = time.perf_counter_ns()
        if self.pool.backend == "rdma":
            counters = self._rdma(query, w, key)
        else:
            counters = DeferredLayerCounters(w["stats"].clone(), 1, self.pool.backend)
            self.pool._pending_read_counts.append((w["region_counts"].clone(), 1))
            if self.pool.cxl is not None:
                self.pool.cxl.last_use.record(stream)
        with torch.cuda.nvtx.range("nsa.selected.update_hot"):
            _retain_hot[(w["heads"], self.cache.hot)](
                descriptors,
                chosen,
                w["frequencies"],
                w["kv"],
                self.cache.tags,
                self.cache.locs,
                self.cache.data,
                w["extent"],
                self.cache.hot,
                num_warps=1,
            )
        self.last_done.record(stream)
        self.cache.last_use.record(stream)
        self.cache.has_last_use = True
        self.cache.planner_calls += 1
        self.submitted = True
        self.calls += 1
        self.last_stats = dict(
            backend=self.pool.backend,
            mode=self.mode,
            planner="gpu",
            cpu_selection_downloads=0,
            persistent_rdma_registration=self.producer is not None,
            rdma_policy="coalesced"
            if self.pool.backend == "rdma" and self.rdma_coalesce
            else "chunks",
            prepare_submit_wall_ms=(prepared - begin) / 1e6,
            total_submit_wall_ms=(time.perf_counter_ns() - begin) / 1e6,
        )
        return output.view_as(query), counters
