"""Memory-pool allocator and full-KV backing stores; HBM only holds replicas."""

import ctypes
import logging
import mmap
import os

from .cuda_transfer import RegisteredCudaMapping
from .cxl import CXLStorage
from .kernels import DEFAULT_KERNEL_DIR
from .rdma import RemoteMemory

logger = logging.getLogger(__name__)


class MemoryPool:
    def __init__(
        self,
        pages,
        dim,
        backend="dram",
        *,
        page_size=64,
        dax_device=None,
        dax_offset=None,
        dax_bytes=None,
        pool_target=None,
        device="mlx5_0",
        port=1,
        gid_index=0,
        kernel_dir=DEFAULT_KERNEL_DIR,
    ):
        if pages <= 0 or dim <= 0 or page_size <= 0:
            raise ValueError("pages, dim and page_size must be positive")
        self.pages = pages
        self.dim = dim
        self.backend = backend
        self.kernel_dir = kernel_dir
        self.page_size = page_size
        self.token_bytes = dim * 2
        self.page_bytes = page_size * self.token_bytes
        self.size = pages * self.page_bytes
        self.free = list(range(pages - 1, -1, -1))
        self.allocated = set()
        self.fd = None
        self.mapping: mmap.mmap | None = None
        self.remote = None
        self.cxl: CXLStorage | None = None
        self.cuda_mapping: RegisteredCudaMapping | None = None
        self.read_bytes = 0
        self.write_bytes = 0
        self.read_operations = 0
        self.write_operations = 0
        self.write_kernel_launches = 0
        self.read_kernel_launches = 0
        self._pending_read_counts = []
        self.rdma_gpu_read_bytes = 0
        self.rdma_gpu_write_bytes = 0
        if backend == "dram":
            self.mapping = mmap.mmap(-1, self.size)
        elif backend == "cxl":
            self.cxl = CXLStorage(
                dax_device, dax_offset, dax_bytes, pages, page_size, dim, kernel_dir
            )
            if self.region_count == 1:
                self.mapping = self.cxl.regions[0].mapping
        elif backend == "rdma":
            if not pool_target:
                raise ValueError("RDMA requires --pool-target")
            self.remote = RemoteMemory(
                pool_target, self.size, self.token_bytes, device, port, gid_index
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
        logger.info(
            "pool_initialized backend=%s pages=%d page_bytes=%d capacity_bytes=%d",
            backend,
            pages,
            self.page_bytes,
            self.size,
        )
        if self.cxl is not None:
            logger.info(
                "cxl_regions_initialized layout=page_round_robin regions=%s",
                self.cxl.stats(),
            )

    @property
    def region_count(self):
        return len(self.cxl.regions) if self.cxl is not None else 1

    def _location(self, slot):
        return self.cxl.location(slot) if self.cxl is not None else (0, slot)

    def route_read_plan(self, sources, destinations, counts):
        if self.cxl is not None and self.region_count > 1:
            return self.cxl.route(sources, destinations, counts)
        return None

    def allocate(self, count):
        if count < 0 or count > len(self.free):
            raise ValueError("Backing pool exhausted")
        slots = [self.free.pop() for _ in range(count)]
        self.allocated.update(slots)
        logger.debug(
            "pages_allocated count=%d allocated=%d free=%d",
            count,
            len(self.allocated),
            len(self.free),
        )
        return slots

    def release(self, slots):
        if len(slots) != len(set(slots)) or any(s not in self.allocated for s in slots):
            raise ValueError("Invalid page release")
        for s in slots:
            self.allocated.remove(s)
            self.free.append(s)
        logger.debug(
            "pages_released count=%d allocated=%d free=%d",
            len(slots),
            len(self.allocated),
            len(self.free),
        )

    def token_slot(self, page, offset):
        """Translate an allocated physical page and token offset to a token slot."""
        if page not in self.allocated or not 0 <= offset < self.page_size:
            raise ValueError("Invalid page or token offset")
        return page * self.page_size + offset

    def token_slots(self, pages, start, stop):
        """Expand owned page spans, validating once per page instead of per token."""
        if not 0 <= start <= stop <= len(pages) * self.page_size:
            raise ValueError("Invalid token span")
        slots = []
        cursor = start
        while cursor < stop:
            page, offset = divmod(cursor, self.page_size)
            count = min(stop - cursor, self.page_size - offset)
            base = self.token_slot(pages[page], offset)
            slots.extend(range(base, base + count))
            cursor += count
        return slots

    def _check(self, slot, tensor):
        if (
            slot < 0
            or slot // self.page_size not in self.allocated
            or tensor.device.type != "cpu"
            or not tensor.is_contiguous()
        ):
            raise ValueError("IO requires an allocated slot and contiguous CPU tensor")
        if tensor.numel() * tensor.element_size() != self.token_bytes:
            raise ValueError("KV token size mismatch")

    def write(self, slot, tensor):
        """Single-token compatibility helper; model writes use write_many."""
        self._check(slot, tensor)
        self.write_many([slot], tensor.reshape(1, self.dim))

    def read_into(self, slot, tensor):
        self._check(slot, tensor)
        self.read_many([slot], tensor.reshape(1, self.dim))

    def write_many(self, slots, tensor, *, asynchronous=False):
        """Write caller-owned, unique slots with O(1) tensor validation.

        The allocator/model must supply integer, in-range slots backed by owned
        pages. Writes do not rescan or deduplicate this trusted internal metadata.
        CUDA inputs use registered DRAM or CXL scatter; CPU inputs keep reference IO.
        asynchronous=True allows registered host writes to return before completion.
        The caller must keep GPU consumers on the same ordered stream and wait
        before CPU reads or freeing pages. The public default waits for completion.
        """
        if self.backend == "rdma":
            self._validate_batch(slots, tensor, device_type="cuda")
            self._rdma_cuda(slots, tensor, write=True)
            return
        if self.backend in ("cxl", "dram") and tensor.device.type == "cuda":
            slots = list(slots)
            self._validate_batch(slots, tensor, device_type="cuda")
            if not slots:
                return
            if self.cxl is not None:
                launches = self.cxl.write(slots, tensor)
            else:
                self._cuda_mapping(tensor.device).write(
                    slots, tensor, synchronize=not asynchronous
                )
                launches = 1
            size = len(slots) * self.token_bytes
            self.write_bytes += size
            self.write_operations += launches
            self.write_kernel_launches += launches
            logger.debug(
                "kv_write_batch backend=%s transport=%s tokens=%d operations=%d kernel_launches=%d bytes=%d source_d2h_bytes=0",
                self.backend,
                "region_cuda_kernel"
                if self.cxl is not None
                else "upstream_cuda_kernel",
                len(slots),
                launches,
                launches,
                size,
            )
            return
        self._transfer_many(slots, tensor, write=True)

    def read_many(self, slots, tensor):
        """CPU/reference IO; device-resident local caches use read_cuda instead."""
        if self.backend == "rdma":
            raise ValueError("RDMA requires read_rdma_cuda; CPU staging is disabled")
        self._transfer_many(slots, tensor, write=False)

    def read_rdma_cuda(self, slots, tensor, destination_rows, *, registration=None):
        """Read misses directly into their final compact HBM rows, not a scratch."""
        if self.backend != "rdma":
            raise ValueError("GPUDirect read requires the RDMA backend")
        self._validate_batch(range(len(tensor)), tensor, device_type="cuda")
        return self._rdma_cuda(
            slots, tensor, write=False, rows=destination_rows, registration=registration
        )

    def registered_token_addresses(self, slots, device):
        """Own registered local regions and translate sparse slots to CUDA aliases.

        The caller retains this pool until its consuming stream completes. This
        is deliberately unavailable for RDMA: a remote MR is not a CUDA pointer.
        """
        if self.backend not in ("dram", "cxl"):
            raise ValueError("Registered pointers require a local backing pool")
        slots = list(slots)
        self._validate_read_slots(slots)
        if self.cxl is None:
            registered = self._cuda_mapping(device)
            return [registered.address + s * self.token_bytes for s in slots]
        import torch

        self.cxl._initialize_cuda(device)
        if self.cxl.last_use is not None:
            torch.cuda.current_stream(device).wait_event(self.cxl.last_use)
        if self.region_count == 1:
            registered = self.cxl.regions[0].cuda
            assert registered is not None
            return [registered.address + s * self.token_bytes for s in slots]
        addresses = []
        for slot in slots:
            region, local = self._location(slot)
            registered = self.cxl.regions[region].cuda
            if registered is None:
                raise ValueError("Source region has no registered allocated pages")
            addresses.append(registered.address + local * self.token_bytes)
        return addresses

    def record_fused_read(self, slots, launches=1):
        """Account exact external tokens read by the mapped TileLang producer."""
        slots = list(slots)
        if self.backend not in ("dram", "cxl"):
            raise ValueError("Fused SM reads require a local backing pool")
        self._validate_read_slots(slots)
        if self.cxl is None or self.region_count == 1:
            self._record_cuda_read(len(slots), launches)
            return
        from collections import Counter

        import torch

        self.read_bytes += len(slots) * self.token_bytes
        self.read_operations += launches
        self.read_kernel_launches += launches
        for region, count in Counter(self._location(s)[0] for s in slots).items():
            self.cxl.record(region, count, write=False)
        self.cxl.last_use = torch.cuda.Event()
        self.cxl.last_use.record(torch.cuda.current_stream(self.cxl.device))

    def _rdma_cuda(self, slots, tensor, *, write, rows=None, registration=None):
        if self.remote is None:
            raise RuntimeError("RDMA pool is closed")
        operations = self.remote.region.transfer_cuda(
            slots, tensor, write, rows=rows, registration=registration
        )
        size = len(slots) * self.token_bytes
        if write:
            self.write_bytes += size
            self.write_operations += operations
            self.rdma_gpu_write_bytes += size
        else:
            self.read_bytes += size
            self.read_operations += operations
            self.rdma_gpu_read_bytes += size
        return operations

    def _cuda_mapping(self, device):
        mapping = self._require_mapping()
        if self.cxl is not None:
            # The single-region read path shares registration with fused writes;
            # CXLStorage owns its lifetime, so never register the same mmap twice.
            self.cxl._initialize_cuda(device)
            import torch

            if self.cxl.last_use is not None:
                torch.cuda.current_stream(device).wait_event(self.cxl.last_use)
            registered = self.cxl.regions[0].cuda
            assert registered is not None
            return registered
        if self.cuda_mapping is None:
            self.cuda_mapping = RegisteredCudaMapping(
                mapping, self.dim, self.pages * self.page_size, device, self.kernel_dir
            )
        return self.cuda_mapping

    def read_cuda(
        self,
        slots,
        destination,
        destination_slots=None,
        *,
        counts=None,
        real_requests=None,
        num_tokens=None,
        region_plan=None,
        region_totals=None,
        value_offsets=None,
        defer_accounting=False,
    ):
        """Read local DRAM/CXL misses directly into HBM with upstream kernels.

        HiSparse supplies a validated GPU plan and enqueues a stream-ordered copy.
        The standalone list interface validates CPU metadata and waits for IO.
        BF16 KV bytes travel directly from the registered mmap to compact HBM rows.
        There is deliberately no CPU fallback on registration or kernel errors.
        """
        if self.backend not in ("cxl", "dram"):
            raise ValueError("Direct CUDA reads require a local DRAM or CXL backend")
        if self.cxl is not None and self.region_count > 1:
            if not self.cxl.fds:
                raise RuntimeError("CXL memory pool is closed")
        else:
            self._require_mapping()
        if defer_accounting:
            if counts is None or real_requests is None:
                raise ValueError("Deferred reads require a validated device plan")
            if self.cxl is not None and self.region_count > 1:
                if region_plan is None:
                    region_plan = self.cxl.route(slots, destination_slots, counts)
                launches = self.cxl.read(
                    region_plan, None, destination, real_requests, value_offsets
                )
                totals = region_plan.counts.sum(dim=1)
            else:
                self._cuda_mapping(destination.device).read_plan(
                    slots, destination_slots, counts, real_requests, destination
                )
                totals = counts.sum().reshape(1)
                launches = 1
                if self.cxl is not None:
                    import torch

                    self.cxl.last_use = torch.cuda.Event()
                    self.cxl.last_use.record(
                        torch.cuda.current_stream(destination.device)
                    )
            # Reduction output owns storage; subsequent workspace reuse is safe.
            self._pending_read_counts.append((totals, launches))
            return launches
        if counts is not None:
            # Internal GPU plans originate in HiSparseCache, whose planner checks
            # logical bounds before this call. Do not copy/revalidate each slot
            # on the CPU. Only the small per-layer summary supplies byte counts.
            if num_tokens is None or num_tokens < 0 or real_requests is None:
                raise ValueError("Device read plans require counts and a token total")
            if not num_tokens:
                return 0
            if self.cxl is not None and self.region_count > 1:
                if region_plan is None:
                    # Compatibility for direct callers. HiSparse routes before
                    # its existing summary download, avoiding this extra wait.
                    region_plan = self.cxl.route(slots, destination_slots, counts)
                    region_totals = region_plan.counts.sum(dim=1).tolist()
                if region_totals is None or sum(region_totals) != num_tokens:
                    raise ValueError("Region read totals must match the layer misses")
                launches = self.cxl.read(
                    region_plan,
                    region_totals,
                    destination,
                    real_requests,
                    value_offsets,
                )
            else:
                self._cuda_mapping(destination.device).read_plan(
                    slots, destination_slots, counts, real_requests, destination
                )
                launches = 1
            self._record_cuda_read(num_tokens, launches)
            return launches
        slots = list(slots)
        destination_slots = (
            list(range(len(slots)))
            if destination_slots is None
            else list(destination_slots)
        )
        RegisteredCudaMapping.validate_read(
            slots,
            destination,
            destination_slots,
            self.dim,
            self.pages * self.page_size,
        )
        if any(slot // self.page_size not in self.allocated for slot in slots):
            raise ValueError("Direct CUDA reads require allocated source slots")
        if not slots:
            return 0
        if self.cxl is not None and self.region_count > 1:
            import torch

            sources = torch.tensor(
                [slots], dtype=torch.int64, device=destination.device
            )
            targets = torch.tensor(
                [destination_slots], dtype=torch.int32, device=destination.device
            )
            counts = torch.tensor(
                [len(slots)], dtype=torch.int32, device=destination.device
            )
            real = torch.ones(1, dtype=torch.int32, device=destination.device)
            plan = self.cxl.route(sources, targets, counts)
            totals = plan.counts.sum(dim=1).tolist()
            launches = self.cxl.read(plan, totals, destination, real)
            torch.cuda.current_stream(destination.device).synchronize()
        else:
            self._cuda_mapping(destination.device).read(
                slots, destination, destination_slots
            )
            launches = 1
        self._record_cuda_read(len(slots), launches)
        return launches

    def _record_cuda_read(self, num_tokens, launches=1):
        size = num_tokens * self.token_bytes
        self.read_bytes += size
        self.read_operations += launches
        self.read_kernel_launches += launches
        if self.cxl is not None and self.region_count == 1:
            import torch

            self.cxl.last_use = torch.cuda.Event()
            self.cxl.last_use.record(torch.cuda.current_stream(self.cxl.device))
            self.cxl.record(0, num_tokens, write=False)
        logger.debug(
            "kv_read_batch backend=%s transport=upstream_hisparse_kernel tokens=%d operations=%d kernel_launches=%d bytes=%d dram_staging_bytes=0",
            self.backend,
            num_tokens,
            launches,
            launches,
            size,
        )

    def flush_read_stats(self):
        """Resolve exact deferred byte counters once at a forward boundary."""
        if not self._pending_read_counts:
            return
        import torch

        pending = self._pending_read_counts
        rows = torch.stack([counts for counts, _ in pending]).tolist()
        self._pending_read_counts = []
        for totals, (_, launches) in zip(rows, pending):
            self.read_bytes += sum(totals) * self.token_bytes
            self.read_operations += launches
            self.read_kernel_launches += launches
            if self.cxl is not None:
                for region, count in enumerate(totals):
                    if self.cxl.regions[region].cuda is not None:
                        self.cxl.record(region, count, write=False)

    def _transfer_many(self, slots, tensor, *, write):
        slots = list(slots)
        self._validate_batch(slots, tensor, device_type="cpu")
        if not write:
            # Preserve validation on the CPU/debug read interface. The trusted
            # write path must not perform this per-token scan or set allocation.
            self._validate_read_slots(slots)
        if not slots:
            return
        locations = [self._location(slot) for slot in slots]
        order = sorted(range(len(slots)), key=locations.__getitem__)
        ordered_slots = [slots[index] for index in order]
        reordered = order != list(range(len(slots)))
        if reordered:
            packed = (
                tensor[order].contiguous()
                if write
                else tensor.new_empty(tensor.shape, pin_memory=tensor.is_pinned())
            )
        else:
            packed = tensor
        limit = len(slots)
        if self.remote:
            # A fixed registered staging MR bounds a verbs operation. A long
            # contiguous span is split only at this transport limit.
            limit = self.remote.region.endpoint.size // self.token_bytes
            if limit < 1:
                raise ValueError("RDMA staging region is smaller than one KV token")
        begin = 0
        operations = 0
        while begin < len(slots):
            end = begin + 1
            while (
                end < len(slots)
                and locations[order[end]][0] == locations[order[end - 1]][0]
                and locations[order[end]][1] == locations[order[end - 1]][1] + 1
                and end - begin < limit
            ):
                end += 1
            self._transfer_span(ordered_slots[begin], packed[begin:end], write)
            operations += 1
            begin = end
        if reordered and not write:
            tensor[order] = packed
        logger.debug(
            "kv_%s_batch backend=%s tokens=%d operations=%d bytes=%d",
            "write" if write else "read",
            self.backend,
            len(slots),
            operations,
            len(slots) * self.token_bytes,
        )

    def _validate_batch(self, slots, tensor, *, device_type):
        """Check only O(1) tensor metadata; never inspect individual write slots."""
        if (
            tensor.device.type != device_type
            or not tensor.is_contiguous()
            or tensor.shape != (len(slots), self.dim)
            or tensor.element_size() != 2
        ):
            raise ValueError(
                f"Batch IO requires contiguous {device_type} [tokens, dim] 16-bit data"
            )

    def _validate_read_slots(self, slots):
        """Validate CPU/debug read addresses before any memory access."""
        if any(
            type(slot) is not int
            or slot < 0
            or slot // self.page_size not in self.allocated
            for slot in slots
        ) or len(set(slots)) != len(slots):
            raise ValueError("Batch IO reads require unique allocated integer slots")

    def _transfer_span(self, slot, tensor, write):
        """One contiguous span of caller-owned slots, excluding unrequested gaps."""
        region, local = self._location(slot)
        offset = local * self.token_bytes
        size = tensor.numel() * tensor.element_size()
        if self.remote:
            self.remote.region.transfer(offset, tensor, write)
        else:
            if self.cxl is not None:
                self.cxl.synchronize()
                mapping = self.cxl.regions[region].mapping
                if mapping is None:
                    raise RuntimeError("CXL memory pool is closed")
            else:
                mapping = self._require_mapping()
            if write:
                mapping[offset : offset + size] = ctypes.string_at(
                    tensor.data_ptr(), size
                )
            else:
                ctypes.memmove(tensor.data_ptr(), mapping[offset : offset + size], size)
            if self.cxl is not None:
                self.cxl.record(region, len(tensor), write=write)
        if write:
            self.write_bytes += size
            self.write_operations += 1
        else:
            self.read_bytes += size
            self.read_operations += 1
        logger.debug(
            "kv_%s backend=%s token_start=%d tokens=%d bytes=%d",
            "write" if write else "read",
            self.backend,
            slot,
            len(tensor),
            size,
        )

    def _require_mapping(self) -> mmap.mmap:
        mapping = self.mapping
        if mapping is None:
            raise RuntimeError("Memory pool is closed or has no local mapping")
        return mapping

    def stats(self):
        self.flush_read_stats()
        result = {
            "backend": self.backend,
            "capacity_pages": self.pages,
            "allocated_pages": len(self.allocated),
            "page_bytes": self.page_bytes,
            "page_size": self.page_size,
            "token_bytes": self.token_bytes,
            "capacity_tokens": self.pages * self.page_size,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "read_operations": self.read_operations,
            "write_operations": self.write_operations,
            "write_kernel_launches": self.write_kernel_launches,
            "read_kernel_launches": self.read_kernel_launches,
        }
        if self.cxl is not None:
            result["layout"] = "page_round_robin"
            result["regions"] = self.cxl.stats()
        if self.backend == "rdma":
            result.update(
                transport="gpudirect_dmabuf",
                rdma_gpu_read_bytes=self.rdma_gpu_read_bytes,
                rdma_gpu_write_bytes=self.rdma_gpu_write_bytes,
                cpu_staging_bytes=0,
            )
        return result

    def close(self):
        self.flush_read_stats()
        logger.info(
            "pool_closing backend=%s read_bytes=%d write_bytes=%d read_operations=%d write_operations=%d write_kernel_launches=%d read_kernel_launches=%d",
            self.backend,
            self.read_bytes,
            self.write_bytes,
            self.read_operations,
            self.write_operations,
            self.write_kernel_launches,
            self.read_kernel_launches,
        )
        if self.cuda_mapping is not None:
            self.cuda_mapping.close()
            self.cuda_mapping = None
        if self.remote:
            self.remote.close()
            self.remote = None
        if self.cxl is not None:
            self.cxl.close()
            self.mapping = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
