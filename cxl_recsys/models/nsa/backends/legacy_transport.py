"""Real local SM reads / GPUDirect chunk producer for TileLang selected NSA.

Planning is dynamic and counted in Worker time. No CPU KV packing, synthetic
residency or remote-address-as-local-pointer shortcut is used. The existing
token-frequency HiSparse hot policy is maintained, including partial blocks.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import triton
import triton.language as tl

from cxl_recsys.models.nsa.backends.fused import fused_kernel
from cxl_recsys.models.nsa.backends.io_plan import plan_blocks
from cxl_recsys.models.nsa.backends.tilelang import build_kernel


@triton.jit
def _copy_hbm(Address, KV, N: tl.constexpr, D: tl.constexpr = 128):
    token = tl.program_id(0)
    d = tl.arange(0, D)
    pointer = tl.load(Address + token)
    value = tl.load(pointer.to(tl.pointer_type(tl.bfloat16)) + d, pointer != 0, other=0)
    tl.store(KV + token * D + d, value)


@triton.jit
def _update_hot(Rows, Chosen, Compact, KV, Tags, Locations, Data, HOT: tl.constexpr):
    head, slot = tl.program_id(0), tl.program_id(1)
    row = tl.load(Rows + head)
    index = row * (HOT + 1) + slot
    chosen = tl.load(Chosen + head * HOT + slot)
    compact = tl.load(Compact + head * HOT + slot)
    tl.store(Tags + index, chosen)
    location = tl.load(Locations + index)
    d = tl.arange(0, 128)
    values = tl.load(KV + tl.maximum(compact, 0) * 128 + d, chosen >= 0, other=0)
    tl.store(Data + location * 128 + d, values, chosen >= 0)


class SelectedTransport:
    """Serialized model executor; each call owns its sparse metadata and buffers.

    The RDMA thread owns its dedicated CUDA stream. CPU chunk completion is
    signalled only after CQ completion AND CUDA flush; compute then waits on
    the recorded stream event. Errors unblock every CPU ticket and are raised
    before the corresponding continuation is launched. No persistent GPU flag
    is used to infer NIC-write visibility.
    """

    def __init__(self, cache, *, mode="overlap", chunk_blocks=64, io_ctas=64):
        if mode not in ("serial", "overlap") or min(chunk_blocks, io_ctas) < 1:
            raise ValueError("Invalid TileLang transport configuration")
        self.cache = cache
        self.pool = cache.pool
        self.mode = mode
        self.chunk_blocks = chunk_blocks
        self.io_ctas = io_ctas
        self.producer = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="nsa-rdma")
            if self.pool.backend == "rdma"
            else None
        )
        self.io_stream = torch.cuda.Stream(device=cache.data.device)
        self.last_done = torch.cuda.Event()
        self.submitted = False
        self.keepalive = None
        self.calls = 0
        self.last_stats = {}

    def close(self):
        if self.producer is not None:
            self.producer.shutdown(wait=True)
        if self.submitted:
            self.last_done.synchronize()
        self.io_stream.synchronize()
        self.keepalive = None

    def _prepare(self, query, blocks, positions, fetches, block_size):
        cache = self.cache
        stream = torch.cuda.current_stream(query.device)
        if self.submitted:
            stream.wait_event(self.last_done)
        if cache.has_last_use:
            stream.wait_event(cache.last_use)
        rows = [f["row"] for f in fetches]
        # Only selection/hot metadata reaches CPU; never a KV payload.
        selections = blocks.cpu().to(torch.int32)
        pos = positions.cpu().to(torch.int32)
        tags = cache.tags[rows, : cache.hot].cpu().tolist()
        # Cache locations are assigned once; only tags/data are mutated.
        locations = [
            list(range(row * (cache.hot + 1), row * (cache.hot + 1) + cache.hot))
            for row in rows
        ]
        hot = [dict(zip(t, loc, strict=True)) for t, loc in zip(tags, locations)]
        total = max(f["cached_length"] + len(f["current_kv"]) for f in fetches)
        resident = np.zeros(
            (len(fetches), (total + block_size - 1) // block_size), dtype=np.bool_
        )
        required, frequencies = {}, [{} for _ in fetches]
        occurrence_old, occurrence_local = [0] * len(fetches), [0] * len(fetches)
        for selected, position in zip(selections.tolist(), pos.tolist(), strict=True):
            for head, choices in enumerate(selected):
                for block in choices:
                    if block < 0 or block * block_size > position:
                        continue
                    length = min(block_size, position + 1 - block * block_size)
                    key = (head, block)
                    required[key] = max(required.get(key, 0), length)
        token_ids = selections.numpy()[..., None].astype(
            np.int64
        ) * block_size + np.arange(block_size)
        allowed = (selections.numpy()[..., None] >= 0) & (
            token_ids <= pos.numpy()[:, None, None, None]
        )
        for head, fetch in enumerate(fetches):
            tokens = token_ids[:, head][allowed[:, head]]
            occurrence_old[head] = int(
                np.count_nonzero(tokens < fetch["cached_length"])
            )
            occurrence_local[head] = len(tokens) - occurrence_old[head]
            frequency = np.bincount(
                tokens[tokens < fetch["history_length"]],
                minlength=fetch["history_length"],
            )
            frequencies[head] = {
                int(token): int(frequency[token]) for token in np.flatnonzero(frequency)
            }
        for (head, block), length in required.items():
            resident[head, block] = all(
                token >= fetches[head]["cached_length"] or token in hot[head]
                for token in range(block * block_size, block * block_size + length)
            )
        plan = plan_blocks(
            selections, pos, torch.from_numpy(resident), block_size, self.chunk_blocks
        )
        capacity = max(1, len(plan.block_keys))
        addresses = [[0] * block_size for _ in range(capacity)]
        hbm = [[0] * block_size for _ in range(capacity)]
        source_slots, destination_rows = [], []
        token_to_compact = [{} for _ in fetches]
        hit_count, miss_count = [0] * len(fetches), [0] * len(fetches)
        for physical, ((head, block), length) in enumerate(
            zip(plan.block_keys, plan.valid_tokens.tolist(), strict=True)
        ):
            fetch = fetches[head]
            for offset in range(length):
                token = block * block_size + offset
                destination = physical * block_size + offset
                token_to_compact[head][token] = destination
                if token >= fetch["cached_length"]:
                    pointer = (
                        fetch["current_kv"].data_ptr()
                        + (token - fetch["cached_length"]) * self.pool.token_bytes
                    )
                elif token in hot[head]:
                    pointer = (
                        cache.data.data_ptr() + hot[head][token] * self.pool.token_bytes
                    )
                    hit_count[head] += 1
                else:
                    page, offset = divmod(token, self.pool.page_size)
                    source_slots.append(
                        fetch["pages"][page] * self.pool.page_size + offset
                    )
                    destination_rows.append(destination)
                    miss_count[head] += 1
                    continue
                addresses[physical][offset] = pointer
                hbm[physical][offset] = pointer
        # Owned CPU page metadata avoids scalar GPU downloads and correctly
        # follows the allocator's page / multi-region layout.
        if self.pool.backend != "rdma" and source_slots:
            external = self.pool.registered_token_addresses(source_slots, query.device)
            for destination, pointer in zip(destination_rows, external, strict=True):
                physical, offset = divmod(destination, block_size)
                addresses[physical][offset] = pointer

        kv = torch.empty(
            (capacity, block_size, 128), dtype=torch.bfloat16, device=query.device
        )

        def upload(values, dtype=torch.int32):
            tensor = torch.as_tensor(values, dtype=dtype).contiguous()
            return tensor.to(query.device, non_blocking=True)

        hbm_addresses = upload(hbm, torch.int64)
        _copy_hbm[(capacity * block_size,)](
            hbm_addresses, kv, capacity * block_size, num_warps=4
        )
        starts = upload(plan.starts if len(plan.starts) else [0])
        row_positions = upload(plan.row_positions)
        width = max(stage.ids.shape[1] for stage in plan.stages)
        ids_cpu = torch.full(
            (len(plan.stages), len(query) * len(fetches), width), -1, dtype=torch.int32
        )
        for i, stage in enumerate(plan.stages):
            ids_cpu[i, :, : stage.ids.shape[1]] = stage.ids
        ids = upload(ids_cpu)
        counts = upload(torch.stack([s.counts for s in plan.stages]))
        q = query.contiguous().flatten(0, 1)
        output = torch.empty_like(q)
        chunk_sources = []
        for begin, end in plan.chunk_ranges:
            pairs = [
                (s, d)
                for s, d in zip(source_slots, destination_rows, strict=True)
                if begin * block_size <= d < end * block_size
            ]
            chunk_sources.append(([s for s, _ in pairs], [d for _, d in pairs]))
        counters = [
            {
                "hits": hit_count[head],
                "misses": miss_count[head],
                "old_references": occurrence_old[head],
                "deduplicated_references": occurrence_old[head]
                - hit_count[head]
                - miss_count[head],
                "local_tokens": occurrence_local[head],
                "planner_calls": int(head == 0),
                "h2d_transfers": 0,
                "cxl_read_kernel_launches": 0,
            }
            for head in range(len(fetches))
        ]
        context = dict(
            plan=plan,
            kv=kv,
            q=q,
            ids=ids,
            counts=counts,
            starts=starts,
            positions=row_positions,
            output=output,
            addresses=upload(addresses, torch.int64),
            valid=upload(plan.valid_tokens if len(plan.valid_tokens) else [0]),
            source_slots=source_slots,
            chunk_sources=chunk_sources,
            counters=counters,
        )
        # Retain original tensors until all producers/consumers and hot copies finish.
        context["owners"] = (query, blocks, positions, fetches, hbm_addresses)
        context["frequencies"], context["token_to_compact"] = (
            frequencies,
            token_to_compact,
        )
        return context

    def _local(self, c, trace):
        plan, q, kv, ids = (c[k] for k in ("plan", "q", "kv", "ids"))
        profile_kernel = getattr(self, "profile_kernel", None)
        trace = trace or profile_kernel is not None
        kernel = (profile_kernel or fused_kernel)(
            len(q),
            q.shape[1],
            len(kv),
            plan.block_size,
            ids.shape[2],
            len(plan.stages),
            plan.resident_blocks,
            self.chunk_blocks,
            io_ctas=self.io_ctas,
            pipelined=self.mode == "overlap",
            use_io=bool(plan.chunk_ranges),
            manual_sync=True,
            full_tile_fastpath=False,
            address_table=True,
            trace=trace,
        )
        flags = torch.empty(len(plan.stages), device=q.device, dtype=torch.int32)
        compute_trace = torch.empty(
            (len(q), len(plan.stages), 2), device=q.device, dtype=torch.uint64
        )
        trace_blocks = len(kv) + (
            len(q) + min(self.io_ctas, len(kv) - plan.resident_blocks)
            if profile_kernel is not None
            else 0
        )
        io_trace = torch.empty((trace_blocks, 2), device=q.device, dtype=torch.uint64)
        c["traces"] = (compute_trace, io_trace)
        with torch.cuda.nvtx.range("nsa.selected.tilelang.cxl"):
            kernel(
                q,
                kv,
                ids,
                c["counts"],
                c["starts"],
                c["positions"],
                c["valid"],
                c["addresses"],
                flags,
                c["output"],
                compute_trace,
                io_trace,
            )
        if c["source_slots"]:
            self.pool.record_fused_read(c["source_slots"])
            c["counters"][0]["cxl_read_kernel_launches"] = 1

    def _rdma(self, c):
        plan, q, kv, ids = (c[k] for k in ("plan", "q", "kv", "ids"))
        geometry = (len(q), q.shape[1], len(kv), plan.block_size, ids.shape[2])
        kernels = [
            build_kernel(
                *geometry, initialize=i == 0, finalize=i == len(plan.stages) - 1
            )
            for i in range(len(plan.stages))
        ]
        maximum = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        denominator = torch.empty_like(maximum)
        accumulator = torch.empty_like(q, dtype=torch.float32)
        stream = torch.cuda.current_stream(q.device)
        fork = torch.cuda.Event()
        fork.record(stream)
        events = [torch.cuda.Event() for _ in plan.chunk_ranges]
        arrived = [threading.Event() for _ in plan.chunk_ranges]
        failure = []

        def produce():
            try:
                with torch.cuda.device(q.device), torch.cuda.stream(self.io_stream):
                    self.io_stream.wait_event(fork)
                    for i, (slots, destinations) in enumerate(c["chunk_sources"]):
                        with torch.cuda.nvtx.range(f"nsa.rdma.chunk.{i}"):
                            if slots:
                                self.pool.read_rdma_cuda(
                                    slots, kv.flatten(0, 1), destinations
                                )
                            events[i].record(self.io_stream)
                        arrived[i].set()
            except BaseException as exc:
                failure.append(exc)
                for ticket in arrived:
                    ticket.set()
                raise

        future = self.producer.submit(produce) if arrived else None

        def attention(i):
            with torch.cuda.nvtx.range(
                "nsa.selected.tilelang.resident"
                if i == 0
                else f"nsa.selected.tilelang.chunk.{i - 1}"
            ):
                kernels[i](
                    q,
                    kv,
                    ids[i],
                    c["starts"],
                    c["positions"],
                    c["counts"][i],
                    maximum,
                    denominator,
                    accumulator,
                    c["output"],
                )

        def wait(i):
            arrived[i].wait()
            if failure:
                future.result()
            stream.wait_event(events[i])

        try:
            if arrived and self.mode == "serial":
                wait(len(arrived) - 1)
            attention(0)
            for i in range(len(arrived)):
                wait(i)
                attention(i + 1)
        finally:
            if future is not None:
                future.result()
        c["state"] = (maximum, denominator, accumulator, fork, events)

    def run(self, query, blocks, positions, fetches, block_size, *, trace=False):
        begin = time.perf_counter_ns()
        with torch.cuda.nvtx.range("nsa.selected.prepare"):
            c = self._prepare(query, blocks, positions, fetches, block_size)
        prepared = time.perf_counter_ns()
        if self.pool.backend == "rdma":
            self._rdma(c)
        else:
            self._local(c, trace)
        submitted = time.perf_counter_ns()
        with torch.cuda.nvtx.range("nsa.selected.update_hot"):
            chosen_rows, compact_rows = [], []
            for head, fetch in enumerate(fetches):
                frequency = c["frequencies"][head]
                chosen = sorted(frequency, key=lambda t: (-frequency[t], t))[
                    : self.cache.hot
                ]
                padding = [-1] * (self.cache.hot - len(chosen))
                chosen_rows.append(chosen + padding)
                compact_rows.append(
                    [c["token_to_compact"][head][t] for t in chosen] + padding
                )
            rows_gpu = torch.tensor(
                [f["row"] for f in fetches], dtype=torch.int32, device=query.device
            )
            hot_gpu = torch.tensor(
                [chosen_rows, compact_rows], dtype=torch.int32, device=query.device
            )
            _update_hot[(len(fetches), self.cache.hot)](
                rows_gpu,
                hot_gpu[0],
                hot_gpu[1],
                c["kv"],
                self.cache.tags,
                self.cache.locs,
                self.cache.data,
                self.cache.hot,
                num_warps=1,
            )
        stream = torch.cuda.current_stream(query.device)
        self.last_done.record(stream)
        self.cache.last_use.record(stream)
        self.cache.has_last_use = True
        self.cache.planner_calls += 1
        self.submitted = True
        self.keepalive = c
        self.calls += 1
        self.last_stats = {
            **c["plan"].stats(),
            "external_tokens": len(c["source_slots"]),
            "backend": self.pool.backend,
            "mode": self.mode,
            "prepare_wall_ms": (prepared - begin) / 1e6,
            "submit_wall_ms": (submitted - prepared) / 1e6,
            "hot_update_wall_ms": (time.perf_counter_ns() - submitted) / 1e6,
        }
        return c["output"].view_as(query), c["counters"]
