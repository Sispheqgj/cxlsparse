"""Whole-layer GPU planning with upstream HiSparse backing-pool transport.

The attention working set is independent of persistent hot-cache capacity.
Hits are gathered and misses are copied before the hot set is updated once.
Failures invalidate the run; there is no retry/rollback.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass

import torch

from .kernels import DEFAULT_KERNEL_DIR, load_planner

logger = logging.getLogger(__name__)


@dataclass
class LayerKVBatch:
    """Independent compact attention buffer; metadata inspection is lazy."""

    values: torch.Tensor
    indices: torch.Tensor
    logical_ids: torch.Tensor
    cached_tokens: int
    sequence_indptr: list[int]

    @property
    def logical_positions(self) -> list[int]:
        # Only explicit diagnostics/tests copy logical IDs to Python.
        return self.logical_ids.tolist()


class HiSparseCache:
    """Token-granular hot replicas plus a whole-layer working-set planner.

    Retain this layer's most frequently selected cacheable history tokens.
    Count references before deduplication, including newly appended history but
    excluding temporary candidates. Ties prefer lower logical positions. There
    is no accumulated frequency or retention of unselected old hot entries.
    CUDA events order workspace/cache reuse across serialized stream calls.
    """

    def __init__(
        self,
        pool,
        rows=32,
        max_history=4096,
        hot=32,
        topk=8,
        dim=64,
        *,
        device_resident=False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Worker requires CUDA; there is no fake HiSparse fallback"
            )
        if not (0 < topk <= hot <= 16384) or hot & (hot - 1):
            raise ValueError("hot must be a power of two, topk <= hot <= 16384")
        self.pool = pool
        if device_resident and pool.backend not in ("cxl", "dram"):
            raise ValueError(
                "Device-resident HiSparse requires local DRAM or CXL reads"
            )
        self.device_resident = device_resident
        self.rows = rows
        self.max_history = max_history
        self.hot = hot
        self.topk = topk
        self.dim = dim
        self.module = load_planner(hot, getattr(pool, "kernel_dir", DEFAULT_KERNEL_DIR))
        self.tags = torch.full((rows, hot + 1), -1, dtype=torch.int32, device="cuda")
        self.locs = torch.arange(
            rows * (hot + 1), dtype=torch.int32, device="cuda"
        ).reshape(rows, hot + 1)
        self.host_locs = torch.full(
            (rows, max_history), -1, dtype=torch.int64, device="cuda"
        )
        self.data = torch.empty(
            (rows * (hot + 1), dim), dtype=torch.bfloat16, device="cuda"
        )
        self.workspace = {}
        self.workspace_batch = 0
        self.workspace_extent = 0
        self.last_use = torch.cuda.Event()
        self.last_use.record(torch.cuda.current_stream(self.data.device))
        self.has_last_use = True
        self.planner_calls = 0
        self.bound_lengths = [0] * rows
        self._device_descriptors = OrderedDict()
        self._device_request_counts = {}
        self._workspace_views = {}
        self._block_scratches = {}
        self.plan_graphs = None

    def enable_plan_graphs(self, capacity):
        from ..models.cuda_graph import CUDAGraphCache

        if not self.device_resident:
            raise ValueError("Planning graphs require device-resident cache")
        self.close_plan_graphs()
        self.plan_graphs = CUDAGraphCache(capacity) if capacity else None

    def close_plan_graphs(self):
        if self.plan_graphs is not None:
            self.plan_graphs.close()
        self.plan_graphs = None

    def bind(self, row, slots):
        stream = torch.cuda.current_stream(self.data.device)
        stream.wait_event(self.last_use)
        self.host_locs[row].fill_(-1)
        self.host_locs[row, : len(slots)] = torch.tensor(
            slots, dtype=torch.int64, device=self.data.device
        )
        self.host_locs.record_stream(stream)
        self.last_use.record(stream)
        self.bound_lengths[row] = len(slots)

    def append_slots(self, row, start, slots):
        """Upload only an append-only history suffix; existing slots stay valid."""
        if (
            not 0 <= row < self.rows
            or start != self.bound_lengths[row]
            or start + len(slots) > self.max_history
        ):
            raise ValueError("Invalid append-only history binding")
        if not slots:
            return
        stream = torch.cuda.current_stream(self.data.device)
        stream.wait_event(self.last_use)
        self.host_locs[row, start : start + len(slots)] = torch.tensor(
            slots, dtype=torch.int64, pin_memory=True
        ).to(self.data.device, non_blocking=True)
        self.host_locs.record_stream(stream)
        self.last_use.record(stream)
        self.bound_lengths[row] += len(slots)

    def invalidate(self):
        torch.cuda.synchronize(self.data.device)
        self.tags.fill_(-1)
        self.last_use.record()
        self.has_last_use = True
        logger.warning("hisparse_invalidating rows=%d", self.rows)

    def fetch(self, row, indices, history_length):
        """Normalize trusted unique, ascending IDs to the layer-batch path."""
        indices = list(indices)
        if (
            not 0 <= row < self.rows
            or not 0 <= history_length <= self.max_history
            or len(indices) > self.topk
            or any(type(i) is not int or not 0 <= i < history_length for i in indices)
        ):
            raise ValueError("fetch needs a valid row and in-range history indices")
        if not indices:
            return self.data[:0], {"hits": 0, "misses": 0}
        batch, counters = self.fetch_layer(
            [
                {
                    "row": row,
                    "indices": torch.tensor(
                        [indices], dtype=torch.int64, device=self.data.device
                    ),
                    "history_length": history_length,
                    "cached_length": history_length,
                    "current_kv": torch.empty(
                        (1, self.dim), dtype=self.data.dtype, device=self.data.device
                    ),
                }
            ]
        )
        from .device_plan import DeferredLayerCounters

        if isinstance(counters, DeferredLayerCounters):
            counters = counters.materialize(counters.statistics.tolist())
            self.pool.flush_read_stats()
        return batch.values[batch.indices[0]], {
            name: counters[0][name] for name in ("hits", "misses")
        }

    def _validate_layer(
        self, row, indices, history_length, *, cached_length, current_kv, block_size=1
    ):
        # Shape/device metadata is CPU-resident. Value checks are fused into
        # plan_layer and reported once for the batch, not through many .item().
        if (
            not 0 <= row < self.rows
            or not 0 <= cached_length <= history_length <= self.max_history
            or indices.ndim != 2
            or indices.shape[0] != len(current_kv)
            or not 0 < indices.shape[1] * block_size <= self.topk
            or not 0 < len(current_kv)
            or history_length > cached_length + len(current_kv)
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.device != self.data.device
            or current_kv.device != self.data.device
            or current_kv.dtype != self.data.dtype
            or current_kv.shape != (len(indices), self.dim)
        ):
            raise ValueError("Invalid layer Top-K/KV shapes, dtype, device or prefix")

    def _workspace(self, batch_size, max_queries):
        extent = self.max_history + max_queries
        if batch_size > self.workspace_batch or extent > self.workspace_extent:
            # Captured planner nodes refer to these persistent scratch addresses.
            if self.plan_graphs is not None:
                self.plan_graphs.close()
            # Captures bind scratch addresses. Drain and discard all old
            # captures before replacing any workspace allocation.
            self._workspace_views.clear()
            self._block_scratches.clear()
            self.workspace_batch = max(batch_size, self.workspace_batch)
            self.workspace_extent = max(extent, self.workspace_extent)
            shape = (self.workspace_batch, self.workspace_extent)
            device = self.data.device
            self.workspace = {
                name: torch.empty(shape, dtype=dtype, device=device)
                for name, dtype in (
                    ("ranks", torch.int32),
                    ("frequencies", torch.int32),
                    ("logical", torch.int32),
                    ("sources", torch.int64),
                    ("destinations", torch.int32),
                    ("hot_lookup", torch.int32),
                    ("miss_offsets", torch.int32),
                )
            }
            self.workspace["counts"] = torch.empty(
                self.workspace_batch, dtype=torch.int32, device=device
            )
            self.workspace["stats"] = torch.empty(
                (self.workspace_batch, 7), dtype=torch.int32, device=device
            )
            self.workspace["tile_prefix"] = torch.empty(
                (self.workspace_batch, (self.workspace_extent + 255) // 256, 4),
                dtype=torch.int32,
                device=device,
            )
            # A user can copy at most Hot hits and all current query tokens.
            # Flat plans use the same capacity but only their first N entries;
            # no per-user padding reaches the payload-copy kernel.
            copy_shape = (
                self.workspace_batch,
                self.hot + self.workspace_extent - self.max_history,
            )
            for name, dtype in (
                ("hbm_src_ptrs", torch.int64),
                ("hbm_dst_local", torch.int32),
                ("copy_src_ptrs", torch.int64),
                ("copy_dst_rows", torch.int32),
            ):
                self.workspace[name] = torch.empty(
                    copy_shape, dtype=dtype, device=device
                )
        if batch_size not in self._workspace_views:
            self._workspace_views[batch_size] = {
                name: value[:batch_size] for name, value in self.workspace.items()
            }
        return self._workspace_views[batch_size]

    def _plan_layer(self, indices, descriptors, current, work):
        self.module.plan_layer(
            indices,
            descriptors,
            self.tags,
            self.host_locs,
            self.locs,
            self.data,
            current,
            work["ranks"],
            work["frequencies"],
            work["logical"],
            work["sources"],
            work["destinations"],
            work["counts"],
            work["stats"],
            work["hbm_src_ptrs"],
            work["hbm_dst_local"],
            self.pool.pages * self.pool.page_size,
            work["hot_lookup"],
            work["miss_offsets"],
            work["tile_prefix"],
            min(len(indices), self.workspace_extent - self.max_history),
        )

    def fetch_layer(self, fetches, *, block_size=1, borrow=False):
        """Plan/gather once, then retain the most selected history KV this layer.

        Valid IDs within each query are unique and ascending by the Indexer's
        contract; neither property is checked here. Cross-query deduplication
        and frequency counting remain part of the planner. Padding stays -1.

        With block_size=64, IDs name blocks. Each block contributes tokens up
        to cached_length + query_index. The device path returns compact block
        bases and descriptors for block attention, retaining token-level hot
        frequency/eviction semantics without expanding query token matrices.

        One small [batch, 7] summary reaches the CPU for exact output shapes,
        validation and accounting. Token-sized plans remain on the GPU for CXL.
        HBM plans contain only hits/current KV, never CXL misses. One metadata
        kernel packs their pointers; a separate warp-per-token kernel copies
        the dense plan without inspecting logical IDs or hit/miss state.
        RDMA copies only address metadata to CPU; the NIC reads into compact HBM.
        The reference local DRAM path stages KV payload through CPU memory.

        The device_resident local DRAM/CXL path instead returns bounded-capacity
        values and DeferredLayerCounters. Its caller must resolve validation
        and pool accounting at a forward boundary; no layer downloads counts.
        borrow=True exposes reusable device buffers: consume them on the current
        stream before the next fetch. The default returns independently owned
        values, and deferred counters always retain their own storage.
        """
        fetches = list(fetches)
        if block_size not in (1, 64) or (block_size == 64 and not self.device_resident):
            raise ValueError("Block selections require the device-resident CXL cache")
        if not fetches or len({item["row"] for item in fetches}) != len(fetches):
            raise ValueError("Batch cache rows must be nonempty and unique")
        for item in fetches:
            self._validate_layer(**item, block_size=block_size)
        if len({item["indices"].shape[1] for item in fetches}) != 1:
            raise ValueError("Batch Top-K matrices must have the same padded width")
        stream = torch.cuda.current_stream(self.data.device)
        if self.has_last_use:
            stream.wait_event(self.last_use)
        query_begin = 0
        descriptor_rows = []
        for item in fetches:
            queries = len(item["indices"])
            descriptor_rows.append(
                [
                    item["row"],
                    item["cached_length"],
                    query_begin,
                    queries,
                    item["history_length"],
                ]
            )
            query_begin += queries
        if self.device_resident:
            layout = tuple(tuple(row) for row in descriptor_rows)
            key = tuple(row[0] for row in descriptor_rows)
            previous = self._device_descriptors.get(key)
            if previous is None or previous[0] != layout:
                descriptors = torch.tensor(
                    descriptor_rows, dtype=torch.int64, pin_memory=True
                ).to(self.data.device, non_blocking=True)
                self._device_descriptors[key] = (layout, descriptors)
            else:
                descriptors = previous[1]
            self._device_descriptors.move_to_end(key)
            if len(self._device_descriptors) > 64:
                self._device_descriptors.popitem(last=False)
            descriptors.record_stream(stream)
        else:
            descriptors = torch.tensor(
                descriptor_rows, dtype=torch.int64, device=self.data.device
            )
        indices = torch.cat([item["indices"] for item in fetches]).to(torch.int64)
        current = torch.cat([item["current_kv"] for item in fetches])
        work = self._workspace(
            len(fetches), max(len(item["indices"]) for item in fetches)
        )
        # Stream waits order accesses; record_stream also protects allocator
        # lifetimes if a subsequent call grows/replaces the reusable workspace.
        for tensor in (
            *work.values(),
            self.tags,
            self.locs,
            self.host_locs,
            self.data,
            current,
        ):
            tensor.record_stream(stream)
        gathered = None
        if self.device_resident and self.plan_graphs is not None:
            from .device_plan import gather

            max_queries = max(len(item["indices"]) for item in fetches)

            def plan_gather(indices, descriptors, current):
                # Only scratch is mutated. Warmup/capture never change hot tags,
                # backing memory or accounting; those happen once after replay.
                if block_size == 64:
                    from .block_plan import plan

                    plan(self, indices, descriptors, current, work, max_queries)
                else:
                    self._plan_layer(indices, descriptors, current, work)
                return gather(
                    indices,
                    descriptors,
                    work,
                    self.dim,
                    max_queries,
                    self.hot,
                    block_size=block_size,
                )

            gathered = self.plan_graphs.run(
                (block_size, len(fetches), max_queries),
                plan_gather,
                (indices, descriptors, current),
                clone_outputs=not borrow,
            )
            self.planner_calls += 1
        elif block_size == 64:
            from .block_plan import plan

            plan(
                self,
                indices,
                descriptors,
                current,
                work,
                max(len(item["indices"]) for item in fetches),
            )
            self.planner_calls += 1
        else:
            self._plan_layer(indices, descriptors, current, work)
            self.planner_calls += 1
        if self.device_resident:
            return self._fetch_device(
                indices,
                descriptors,
                work,
                fetches,
                stream,
                block_size=block_size,
                gathered=gathered,
                borrow=borrow,
            )
        region_plan = (
            self.pool.route_read_plan(
                work["sources"], work["destinations"], work["counts"]
            )
            if self.pool.backend == "cxl"
            else None
        )
        # Keep the existing single small D2H boundary. Full token plans remain
        # on GPU; append just one count per user/region for accounting/launches.
        summary_tensor = work["stats"]
        if region_plan is not None:
            summary_tensor = torch.cat((summary_tensor, region_plan.counts.T), dim=1)
        summary = summary_tensor.tolist()
        if any(row[6] for row in summary):
            raise ValueError(
                "Top-K must contain causal positions or -1 padding, with bound history slots"
            )
        sequence_indptr = [0]
        copy_count = 0
        offset_rows = []
        metadata_items = 0
        for row, descriptor in zip(summary, descriptor_rows):
            offset_rows.append([sequence_indptr[-1], copy_count])
            sequence_indptr.append(sequence_indptr[-1] + row[0])
            # local_tokens counts references, not unique current tokens. The
            # exact number of HBM jobs is the union size minus unique misses.
            copies = row[0] - row[3]
            copy_count += copies
            metadata_items = max(
                metadata_items, descriptor[3] * indices.shape[1], row[3], copies
            )
        # Upload only O(batch) offsets without introducing another host wait.
        # Keep the pinned source private to this call; never overwrite a shared
        # staging buffer while an earlier stream may still be reading it.
        host_offsets = torch.tensor(
            offset_rows, dtype=torch.int64, device="cpu", pin_memory=True
        )
        offsets = host_offsets.to(self.data.device, non_blocking=True)
        values = torch.empty(
            (sequence_indptr[-1], self.dim),
            dtype=self.data.dtype,
            device=self.data.device,
        )
        mapped = torch.empty_like(indices)
        self.module.gather_layer(
            indices,
            descriptors,
            work["ranks"],
            work["stats"],
            offsets,
            work["hbm_src_ptrs"],
            work["hbm_dst_local"],
            work["copy_src_ptrs"],
            work["copy_dst_rows"],
            values,
            mapped,
            work["destinations"],
            copy_count,
            metadata_items,
        )
        misses = sum(row[3] for row in summary)
        read_launches = 0
        if misses:
            if self.pool.backend == "cxl":
                region_options = {}
                if region_plan is not None:
                    region_options = {
                        "region_plan": region_plan,
                        "region_totals": [
                            sum(row[7 + region] for row in summary)
                            for region in range(self.pool.region_count)
                        ],
                        "value_offsets": offsets[:, 0],
                    }
                read_launches = self.pool.read_cuda(
                    work["sources"],
                    values,
                    work["destinations"],
                    counts=work["counts"],
                    real_requests=torch.full(
                        (1,), len(fetches), dtype=torch.int32, device=self.data.device
                    ),
                    num_tokens=misses,
                    **region_options,
                )
            else:
                # CPU-posted verbs needs slot metadata, not CPU-resident KV.
                sources = torch.cat(
                    [work["sources"][i, : row[3]] for i, row in enumerate(summary)]
                )
                destinations = torch.cat(
                    [work["destinations"][i, : row[3]] for i, row in enumerate(summary)]
                )
                if self.pool.backend == "rdma":
                    self.pool.read_rdma_cuda(
                        sources.tolist(), values, destinations.tolist()
                    )
                else:
                    scratch = torch.empty(
                        (misses, self.dim), dtype=self.data.dtype, pin_memory=True
                    )
                    self.pool.read_many(sources.tolist(), scratch)
                    values[destinations.long()] = scratch.to(self.data.device)
        # Stable GPU sorting ranks counts descending and logical IDs ascending
        # for ties. Counts are reset by plan_layer on every call. This uses
        # precompiled PyTorch CUDA operators, not CPU metadata or runtime JIT.
        selected_hot = torch.argsort(
            work["frequencies"], dim=1, descending=True, stable=True
        )[:, : self.hot].contiguous()
        self.module.update_hot(
            descriptors,
            work["ranks"],
            work["frequencies"],
            selected_hot,
            work["stats"],
            self.tags,
            self.locs,
            values,
            self.data,
        )
        # Returned data owns storage: the next layer may immediately reuse scratch.
        logical = torch.cat(
            [work["logical"][i, : row[0]] for i, row in enumerate(summary)]
        )
        self.last_use.record(stream)
        self.has_last_use = True
        owner = next((i for i, row in enumerate(summary) if row[3]), None)
        counters = [
            {
                "hits": row[2],
                "misses": row[3],
                "old_references": row[4],
                "deduplicated_references": row[4] - row[1],
                "local_tokens": row[5],
                "planner_calls": int(i == 0),
                "h2d_transfers": int(i == owner and self.pool.backend == "dram"),
                "cxl_read_kernel_launches": read_launches if i == owner else 0,
            }
            for i, row in enumerate(summary)
        ]
        logger.debug(
            "hisparse_batch_ready batch_size=%d unique_tokens=%d hits=%d misses=%d backing_read_calls=%d planner_calls=1 policy=layer_frequency",
            len(fetches),
            sequence_indptr[-1],
            sum(row[2] for row in summary),
            misses,
            int(bool(misses)),
        )
        return LayerKVBatch(
            values, mapped, logical, sum(row[1] for row in summary), sequence_indptr
        ), counters

    def _fetch_device(
        self,
        indices,
        descriptors,
        work,
        fetches,
        stream,
        *,
        block_size=1,
        gathered=None,
        borrow=False,
    ):
        from .device_plan import DeferredLayerCounters, DeviceLayerKVBatch, gather

        values, mapped, selected_hot = (
            gathered
            if gathered is not None
            else gather(
                indices,
                descriptors,
                work,
                self.dim,
                max(len(item["indices"]) for item in fetches),
                self.hot,
                block_size=block_size,
            )
        )
        batch_size = len(fetches)
        if batch_size not in self._device_request_counts:
            self._device_request_counts[batch_size] = torch.full(
                (1,), batch_size, dtype=torch.int32, device=values.device
            )
        real_requests = self._device_request_counts[batch_size]
        real_requests.record_stream(stream)
        launches = self.pool.read_cuda(
            work["sources"],
            values,
            work["destinations"],
            counts=work["counts"],
            real_requests=real_requests,
            defer_accounting=True,
        )
        self.module.update_hot(
            descriptors,
            work["ranks"],
            work["frequencies"],
            selected_hot,
            work["stats"],
            self.tags,
            self.locs,
            values,
            self.data,
        )
        summary = work["stats"].clone()
        # Counters always own storage; logical IDs may use the explicit borrow contract.
        batch = DeviceLayerKVBatch(
            values,
            mapped,
            work["logical"] if borrow else work["logical"].clone(),
            summary,
        )
        if block_size == 64:
            batch.blocks = indices
            batch.descriptors = descriptors
        self.last_use.record(stream)
        self.has_last_use = True
        return batch, DeferredLayerCounters(summary, launches, self.pool.backend)
