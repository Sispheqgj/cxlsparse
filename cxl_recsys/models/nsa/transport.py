"""Opt-in GR selected execution: compact planning and bounded query consumers.

The short, single-query consumer retains the admitted cooperative CXL kernel.
Batch consumers fetch their planned working set before bounded all-HBM groups;
query-private plans keep separate targets. No cooperative grid is enlarged for
a long suffix.
Persistent hot state is committed once, after graph warmup/capture/replay.
"""

import math
import time

import torch
import triton
import triton.language as tl

from cxl_recsys.mempool.device_plan import DeferredLayerCounters
from cxl_recsys.models.nsa.backends.gpu_fused import gpu_fused_kernel
from cxl_recsys.models.nsa.backends.gpu_transport import GPUSelectedTransport
from cxl_recsys.models.nsa.backends.tilelang import build_kernel
from cxl_recsys.models.nsa.planning.active import ActivePlanAdapter


class GRWorkspaceLimit(RuntimeError):
    """Capacity refusal before any plan/IO/hot state has been submitted."""


@triton.jit
def _read_external(
    Addresses, Slots, IOIDs, Missing, KV, BS: tl.constexpr, CTAS: tl.constexpr
):
    """Dedicated IO CTAs; kernel completion publishes all planned payloads."""
    token = tl.arange(0, BS)
    dim = tl.arange(0, 128)
    for rank in range(tl.program_id(0), tl.load(Missing), CTAS):
        physical = tl.load(IOIDs + rank)
        source = tl.load(Slots + physical * BS + token)
        address = tl.load(Addresses + physical * BS + token)
        external = (source >= 0) & (address != 0)
        value = tl.load(
            address[:, None].to(tl.pointer_type(tl.bfloat16)) + dim[None, :],
            external[:, None],
            other=0,
        )
        # Partially resident blocks retain the latest HBM token values.
        tl.store(
            KV + (physical * BS + token[:, None]) * 128 + dim[None, :],
            value,
            external[:, None],
        )


@triton.jit
def _expand_native_indices(
    IDs,
    Counts,
    Starts,
    Positions,
    Tokens,
    WIDTH: tl.constexpr,
    BS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.arange(0, B)
    valid = (token < WIDTH * BS) & (token // BS < tl.load(Counts + row))
    physical = tl.load(IDs + row * WIDTH + token // BS, valid, -1)
    valid &= physical >= 0
    absolute = tl.load(Starts + physical, valid, 0) + token % BS
    valid &= absolute <= tl.load(Positions + row)
    tl.store(
        Tokens + row * WIDTH * BS + token,
        tl.where(valid, physical * BS + token % BS, -1),
        token < WIDTH * BS,
    )


def _tensor_bytes(work):
    seen, result = set(), 0
    for value in work.values():
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            identity = (value.device, storage.data_ptr())
            if identity not in seen:
                seen.add(identity)
                result += storage.nbytes()
    return result


def workspace_estimate(
    queries,
    cached,
    heads,
    groups,
    width,
    bs,
    *,
    rdma=False,
    chunk_blocks=64,
    all_hbm=False,
    cross_query_dedup=True,
    native=False,
):
    """Host-known conservative bound, also used before selecting a model route."""
    from cxl_recsys.models.nsa.dispatch import query_bucket

    q = query_bucket(queries)
    logical = max(64, math.ceil((cached + q) / (64 * bs)) * 64)
    physical = heads * (min(logical, q * width) if cross_query_dedup else q * width)
    rows = q * heads
    if chunk_blocks < 1:
        raise ValueError("Positive chunk capacity required")
    stages = 2 if rdma or q > 128 or all_hbm else 1 + math.ceil(physical / chunk_blocks)
    # The bounded native consumer expands compact block occurrences to int32
    # token indices. Including this for shared-query long buckets is conservative.
    token_indices = rows * width * bs * 4 if rdma or q > 128 or native else 0
    return (
        physical * bs * 384
        + stages * rows * (width + 5) * 4
        + rows * groups * 64 * 16
        + logical * heads * 16
        + token_indices
        + 4 * 1024**2
    )


class GRSelectedTransport(GPUSelectedTransport):
    """Select strict, resident-first batch, or individual-block readiness.

    Reordering is limited to the existing cooperative short-query route.
    Later layers may select different blocks because of changed rounding.
    Numerical acceptance is separate from source and consumption correctness.
    """

    def __init__(
        self,
        cache,
        *,
        query_group=1,
        workspace_bytes=512 * 1024**2,
        consume_order="strict",
        read_order="physical",
        read_group_size=16,
        consumer_backend="legacy",
        ready_window=1,
        cross_query_dedup=True,
        rdma_registration="persistent",
        **kwargs,
    ):
        from cxl_recsys.models.nsa.planning.read_order import DEPTH_ORDERS, READ_ORDERS

        if consumer_backend not in ("legacy", "tma", "native"):
            raise ValueError("Unknown GR consumer backend")
        if ready_window not in (1, 2, 4):
            raise ValueError("NSA ready_window must be 1, 2 or 4")
        if ready_window > 1 and consumer_backend != "tma":
            raise ValueError("NSA ready windows require the TMA consumer")
        if consumer_backend == "tma" and (
            cache.pool.backend not in ("dram", "cxl")
            or query_group != 1
            or consume_order != "strict"
            or read_order not in ("physical", *DEPTH_ORDERS)
            or kwargs.get("mode", "overlap") != "overlap"
        ):
            raise ValueError(
                "TMA requires local backing, strict order and overlap mode"
            )
        if read_order in DEPTH_ORDERS and consumer_backend != "tma":
            raise ValueError("Consumption-depth ordering requires the TMA consumer")
        if consumer_backend == "native" and (
            consume_order != "strict" or read_order != "physical" or query_group != 1
        ):
            raise ValueError("Native batch consumption requires strict physical order")
        if rdma_registration not in ("persistent", "per_call"):
            raise ValueError("Unknown RDMA registration policy")
        if not cross_query_dedup and consumer_backend == "legacy":
            raise ValueError("Query-private plans require native or TMA consumers")
        self.consumer_backend = consumer_backend
        self.cross_query_dedup = cross_query_dedup
        self.rdma_registration = rdma_registration
        self.ready_window = ready_window
        super().__init__(cache, **kwargs)
        if query_group not in (1, 2, 4, 8) or workspace_bytes < 1:
            raise ValueError("Invalid GR selected resource policy")
        if consume_order not in ("strict", "resident_first", "block_ready"):
            raise ValueError("Unknown NSA consumption order")
        if read_order not in READ_ORDERS or read_group_size < 1:
            raise ValueError("Invalid NSA read priority configuration")
        if read_order != "physical" and (
            self.pool.backend == "rdma" or query_group != 1
        ):
            raise ValueError(
                "Read priority requires local cooperative single-token consumers"
            )
        self.read_order, self.read_group_size = read_order, read_group_size
        self.consume_order = consume_order
        from cxl_recsys.models.nsa.dispatch import NSABucketInputs

        self.buckets = NSABucketInputs(capacity=self.capacity)
        adapter = ActivePlanAdapter
        if not cross_query_dedup:
            from cxl_recsys.models.nsa.planning.query_private import (
                QueryPrivatePlanAdapter,
            )

            adapter = QueryPrivatePlanAdapter
        self.active_plan = adapter(
            cache, chunk_blocks=self.chunk_blocks, rdma_coalesce=True
        )
        self.query_group = query_group
        self.workspace_budget = workspace_bytes
        self.workspace_allocations = self.workspace_evictions = 0
        self.mr_registrations = 0

    def refusal(self, queries, cached, heads, groups, width, bs):
        if (
            self.read_order != "physical"
            and queries > 128
            and self.consumer_backend != "tma"
        ):
            raise ValueError("Read priority supports Q<=128; refusing silent fallback")
        if not 1 <= queries <= 1024:
            return "query_bucket_limit"
        estimate = workspace_estimate(
            queries,
            cached,
            heads,
            groups,
            width,
            bs,
            rdma=self.pool.backend == "rdma",
            chunk_blocks=self.chunk_blocks,
            all_hbm=self.query_group > 1 or self.consumer_backend != "legacy",
            cross_query_dedup=self.cross_query_dedup,
            native=self.consumer_backend == "native",
        )
        return "workspace_budget" if estimate > self.workspace_budget else None

    def close(self):
        super().close()
        self.buckets.close()

    def _workspace(self, query, blocks, fetches, bs):
        if (
            self.read_order != "physical"
            and len(query) > 128
            and self.consumer_backend != "tma"
        ):
            raise ValueError("Read priority supports Q<=128; refusing silent fallback")
        # Bucket the logical extent, as well as Q. No effective Q enters a key.
        logical = max(f["cached_length"] + len(query) for f in fetches)
        map_blocks = max(64, math.ceil(logical / (64 * bs)) * 64)
        geometry = self.active_plan.geometry(
            query,
            blocks,
            fetches,
            bs,
            map_blocks=map_blocks,
            all_hbm=len(query) > 128
            or self.query_group > 1
            or self.consumer_backend != "legacy",
        )
        key = (len(query), query.shape[1:], blocks.shape[-1], bs, map_blocks)
        if key in self.workspaces:
            self.workspaces.move_to_end(key)
            return key, self.workspaces[key]
        # A conservative upper bound includes IDs, addresses, KV, sort scratch,
        # graph-owned input/output copies and per-consumer online-softmax state.
        p, rows, width = geometry["physical"], geometry["rows"], geometry["width"]
        estimate = workspace_estimate(
            len(query),
            max(f["cached_length"] for f in fetches),
            query.shape[1],
            query.shape[2],
            width,
            bs,
            rdma=self.pool.backend == "rdma",
            chunk_blocks=self.chunk_blocks,
            all_hbm=self.query_group > 1 or self.consumer_backend != "legacy",
            cross_query_dedup=self.cross_query_dedup,
            native=self.consumer_backend == "native",
        )
        if estimate > self.workspace_budget:
            raise GRWorkspaceLimit(
                f"GR workspace estimate {estimate} exceeds {self.workspace_budget} bytes"
            )
        resident = sum(w["allocated_bytes"] for w in self.workspaces.values())
        while self.workspaces and (
            len(self.workspaces) >= self.capacity
            or resident + estimate > self.workspace_budget
        ):
            # Graphs close before any referenced buffer or MR is released.
            if self.graphs is not None:
                self.graphs.close()
            if self.attention_graphs is not None:
                self.attention_graphs.close()
            if self.submitted:
                self.last_done.synchronize()
            _, old = self.workspaces.popitem(last=False)
            if old.get("registration") is not None:
                old["registration"].close()
            resident -= old["allocated_bytes"]
            self.workspace_evictions += 1
        w = self.active_plan.allocate(
            query,
            blocks,
            fetches,
            bs,
            map_blocks=map_blocks,
            all_hbm=len(query) > 128
            or self.query_group > 1
            or self.consumer_backend != "legacy",
            read_order=self.read_order,
            occurrence_only=self.consumer_backend == "tma",
        )
        w["cooperative"] = self.consumer_backend == "tma" or (
            self.consumer_backend == "legacy"
            and self.pool.backend != "rdma"
            and len(query) <= 128
            and self.query_group == 1
        )
        # Keep long suffixes, RDMA and experimental grouped consumers unchanged.
        w["consume_order"] = self.consume_order if w["cooperative"] else "strict"
        if self.consumer_backend == "tma":
            from cxl_recsys.models.nsa.tma.transport import configure_workspace

            configure_workspace(query, w, ready_window=self.ready_window)
        elif w["cooperative"]:
            if w["consume_order"] == "block_ready":
                # Batch counters remain for the legacy non-overlap mode.
                w["barriers"] = torch.empty(
                    w["stages"] + p, dtype=torch.int32, device=query.device
                )
                w["scratch_bytes"] += p * w["barriers"].element_size()
            w["kernel"] = gpu_fused_kernel(
                rows,
                query.shape[2],
                p,
                bs,
                width,
                self.chunk_blocks,
                io_ctas=self.io_ctas,
                pipelined=self.mode == "overlap",
                manual_sync=True,
                address_table=True,
                skip_empty_stages=w["consume_order"] != "strict",
                block_ready=w["consume_order"] == "block_ready",
            )
        else:
            w["native_bounded"] = (
                self.consumer_backend == "native"
                or self.pool.backend == "rdma"
                or len(query) > 128
            ) and self.query_group == 1
            if w["native_bounded"]:
                w["token_ids"] = torch.empty(
                    (rows, width * bs), dtype=torch.int32, device=query.device
                )
                # RDMA's planner allocates continuation state for its legacy
                # consumer. Native indexed attention owns no such global state.
                for name in ("maximum", "denominator", "accumulator"):
                    w.pop(name, None)
            else:
                w["maximum"] = torch.empty(
                    (rows, query.shape[2]), dtype=torch.float32, device=query.device
                )
                w["denominator"] = torch.empty_like(w["maximum"])
                w["accumulator"] = torch.empty_like(w["output"], dtype=torch.float32)
            w["groups"] = []
            for begin in range(0, len(query), 128):
                end = min(begin + 128, len(query))
                kernel = (
                    build_kernel(
                        (end - begin) * w["heads"],
                        query.shape[2],
                        p,
                        bs,
                        width,
                    )
                    if self.query_group == 1 and not w["native_bounded"]
                    else None
                )
                tile_work = None
                if self.query_group > 1:
                    from cxl_recsys.models.nsa.experimental.query_tile import (
                        allocate_workspace,
                    )

                    tile_work = allocate_workspace(
                        query[begin:end],
                        w["kv"],
                        w["all_ids"][begin * w["heads"] : end * w["heads"]],
                        tile_queries=self.query_group,
                    )
                w["groups"].append((begin, end, kernel, tile_work))
        if self.pool.backend == "rdma":
            w["rdma_values"] = w["kv"].flatten(0, 1)
            w["registration"] = None
            if self.rdma_registration == "persistent":
                w["registration"] = self.pool.remote.region.register_cuda(
                    w["rdma_values"]
                )
                self.mr_registrations += 1
        w["allocated_bytes"] = _tensor_bytes(w) + sum(
            group[3].bytes for group in w.get("groups", ()) if group[3] is not None
        )
        self.workspaces[key] = w
        self.workspace_allocations += 1
        return key, w

    def _consume(self, query, w):
        if w.get("native_bounded"):
            _expand_native_indices[(w["rows"],)](
                w["all_ids"],
                w["all_counts"],
                w["starts"],
                w["positions"],
                w["token_ids"],
                w["width"],
                w["bs"],
                triton.next_power_of_2(w["width"] * w["bs"]),
                num_warps=1,
            )
        for begin, end, kernel, tile_work in w["groups"]:
            lo, hi = begin * w["heads"], end * w["heads"]
            if w.get("native_bounded"):
                from cxl_recsys.models.nsa.attention import indexed_attention

                # Keep baseline arithmetic for RDMA and long suffixes: small
                # BF16 differences can otherwise flip the next layer's Top-K.
                # Each group consumes its planned shared or query-private KV.
                value = indexed_attention(
                    query[begin:end].flatten(0, 1),
                    w["kv"].flatten(0, 1),
                    w["token_ids"][lo:hi],
                    w["bs"],
                )
                w["output"][lo:hi].copy_(value)
            elif self.query_group > 1:
                from cxl_recsys.models.nsa.experimental.query_tile import (
                    query_tile_attention,
                )

                value = query_tile_attention(
                    query[begin:end],
                    w["kv"],
                    w["all_ids"][lo:hi],
                    w["starts"],
                    w["positions"][lo:hi],
                    w["all_counts"][lo:hi],
                    tile_queries=self.query_group,
                    workspace=tile_work,
                )
                w["output"][lo:hi].copy_(value.flatten(0, 1))
            else:
                kernel(
                    query[begin:end].flatten(0, 1),
                    w["kv"],
                    w["all_ids"][lo:hi],
                    w["starts"],
                    w["positions"][lo:hi],
                    w["all_counts"][lo:hi],
                    w["maximum"][lo:hi],
                    w["denominator"][lo:hi],
                    w["accumulator"][lo:hi],
                    w["output"][lo:hi],
                )
        return (w["output"],)

    def _plan(self, query, blocks, positions, desc, bases, w):
        chosen = self.active_plan.plan(
            blocks,
            positions,
            desc,
            bases,
            w,
            consume_order=w["consume_order"],
            read_order=self.read_order,
            read_group_size=self.read_group_size,
        )
        if self.pool.backend != "rdma":
            if self.consumer_backend == "tma":
                from cxl_recsys.models.nsa.tma.transport import execute

                execute(query, w)
            elif w["cooperative"]:
                w["kernel"](
                    query.flatten(0, 1),
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
            else:
                ctas = min(self.io_ctas, w["physical"])
                _read_external[(ctas,)](
                    w["address"],
                    w["slots"],
                    w["ioids"],
                    w["missing"],
                    w["kv"],
                    w["bs"],
                    ctas,
                )
                self._consume(query, w)
        return w["output"], chosen

    def _rdma(self, query, w, key):
        # The required metadata download is the sole CPU inspection boundary.
        w["meta_host"].copy_(w["meta"], non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(query.device))
        ready.synchronize()
        meta = w["meta_host"].numpy()
        missing = int(meta[0, 0])
        counters = DeferredLayerCounters(w["stats"], 0, "rdma").materialize(
            meta[0, 1 + w["physical"] * w["bs"] :].reshape(w["heads"], 7).tolist()
        )
        sources, destinations = meta[:, 1 : 1 + missing * w["bs"]]
        valid = sources >= 0
        if self.rdma_registration == "per_call" and valid.any():
            self.mr_registrations += 1
        self.pool.read_rdma_cuda(
            sources[valid],
            w["rdma_values"],
            destinations[valid],
            registration=w["registration"],
        )
        # read_rdma_cuda has completed CQ and GPU visibility before consumption.
        if self.attention_graphs is None:
            self._consume(query, w)
        else:
            self.attention_graphs.run(
                key, lambda q: self._consume(q, w), (query,), clone_outputs=False
            )
        return counters

    def run(self, query, blocks, positions, fetches, block_size, *, trace=False):
        if trace:
            raise ValueError("Collect device activity separately from GR timing")
        if (
            query.ndim != 4
            or blocks.ndim != 3
            or query.shape[:2] != blocks.shape[:2]
            or len(fetches) != query.shape[1]
            or positions.shape != (len(query),)
            or blocks.dtype != torch.int32
            or positions.dtype not in (torch.int32, torch.int64)
            or any(t.device != query.device for t in (blocks, positions))
        ):
            raise ValueError("Invalid GR query/selection geometry")
        begin = time.perf_counter_ns()
        stream = torch.cuda.current_stream(query.device)
        if self.submitted:
            stream.wait_event(self.last_done)
        if self.cache.has_last_use:
            stream.wait_event(self.cache.last_use)
        from cxl_recsys.models.nsa.dispatch import BucketOwner, query_bucket

        lease = None
        if len(query) == query_bucket(len(query)) and all(
            tensor.is_contiguous() for tensor in (query, blocks, positions)
        ):
            q, b, pos = query, blocks, positions
        else:
            owner = BucketOwner(
                user=tuple(f["row"] for f in fetches),
                layer=-1,
                backing=id(self.pool),
                epoch=self.calls + 1,
            )
            lease = self.buckets.prepare(query, blocks, positions, owner=owner)
            q, b, pos = lease.query, lease.blocks, lease.positions
        try:
            key, w = self._workspace(q, b, fetches, block_size)
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
            desc = self.descriptors.get(layout)
            if desc is None:
                desc = torch.tensor(layout, dtype=torch.int64, pin_memory=True).to(
                    query.device, non_blocking=True
                )
                self.descriptors[layout] = desc
                if len(self.descriptors) > 128:
                    self.descriptors.popitem(last=False)
            else:
                self.descriptors.move_to_end(layout)
            if self.pool.backend == "rdma":
                bases = w["address"].flatten()[:1]
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
            inputs = (q, b, pos, desc, bases)
            for tensor in inputs:
                tensor.record_stream(stream)

            def function(*args):
                return self._plan(*args, w)

            output, chosen = (
                self.graphs.run(key, function, inputs, clone_outputs=False)
                if self.graphs is not None
                else function(*inputs)
            )
            if self.pool.backend == "rdma":
                counters = self._rdma(q, w, key)
            else:
                counters = DeferredLayerCounters(
                    w["stats"].clone(), 1, self.pool.backend
                )
                self.pool._pending_read_counts.append((w["region_counts"].clone(), 1))
                if self.pool.cxl is not None:
                    self.pool.cxl.last_use.record(stream)
            self.active_plan.retain_hot(desc, chosen, w)
            self.last_done.record(stream)
            self.cache.last_use.record(stream)
            self.cache.has_last_use = self.submitted = True
            self.cache.planner_calls += 1
            self.calls += 1
            self.last_stats = dict(
                backend=self.pool.backend,
                planner="active_gpu" if self.cross_query_dedup else "query_private_gpu",
                cross_query_dedup=self.cross_query_dedup,
                rdma_registration=self.rdma_registration,
                consume_order=w["consume_order"],
                read_order=self.read_order,
                read_group_size=self.read_group_size,
                read_order_metadata_bytes=w.get("read_order_bytes", 0),
                effective_q=len(query),
                query_bucket=len(q),
                bucket_staging=lease is not None,
                history=[f["history_length"] for f in fetches],
                query_group=self.query_group,
                consumer_groups=1 if w["cooperative"] else len(w["groups"]),
                io_policy=(
                    "cooperative"
                    if w["cooperative"]
                    else "union_once"
                    if self.cross_query_dedup
                    else "batch_once"
                ),
                consumer=(
                    "tma_bounded_rows"
                    if self.consumer_backend == "tma"
                    else "native_indexed_bounded"
                    if w.get("native_bounded")
                    else "tilelang_query_shared"
                    if self.query_group > 1
                    else "tilelang_single_query"
                ),
                cpu_selection_downloads=0,
                workspace_allocations=self.workspace_allocations,
                workspace_evictions=self.workspace_evictions,
                workspace_bytes=sum(
                    x["allocated_bytes"] for x in self.workspaces.values()
                ),
                logical_block_capacity=w["map_blocks"],
                active_block_capacity=w["physical"],
                persistent_mapping_bytes=w["persistent_mapping_bytes"],
                plan_reset_bytes=w["reset_bytes"],
                mr_registrations=self.mr_registrations,
                attention_graphs=self.attention_graphs.stats()
                if self.attention_graphs
                else None,
                total_submit_wall_ms=(time.perf_counter_ns() - begin) / 1e6,
            )
            if self.consumer_backend == "tma":
                self.last_stats.update(
                    io_policy="tma_block_ready",
                    tma_copy="bulk_1d",
                    io_ctas=w["io_ctas"],
                    consumer_ctas=w["consumer_ctas"],
                    ready_window=w["ready_window"],
                    partial_block_policy="external_token_gather",
                )
            return output.view_as(q)[: len(query)], counters
        finally:
            if lease is not None:
                lease.release()
