"""Page-striped devdax regions and direct, stream-ordered multi-region IO."""

import fcntl
import mmap
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common.sizes import dax_regions
from .cuda_transfer import RegisteredCudaMapping


def dax_geometry(device):
    sysfs = Path("/sys/bus/dax/devices") / Path(device).name
    return (
        int((sysfs / "align").read_text().strip(), 0),
        int((sysfs / "size").read_text().strip(), 0),
    )


@dataclass
class CXLRegion:
    device: str
    offset: int
    reserved_bytes: int
    pages: int
    mapped_bytes: int
    mapping: mmap.mmap | None = None
    cuda: RegisteredCudaMapping | None = None
    stream: Any = None
    read_bytes: int = 0
    write_bytes: int = 0
    read_operations: int = 0
    write_operations: int = 0


@dataclass
class RegionReadPlan:
    sources: Any
    destinations: Any
    counts: Any


class CXLStorage:
    def __init__(self, devices, offsets, sizes, pages, page_size, dim, kernel_dir):
        self.page_size = page_size
        self.dim = dim
        self.token_bytes = dim * 2
        self.kernel_dir = kernel_dir
        self.regions: list[CXLRegion] = []
        self.fds: dict[str, int] = {}
        self.module: Any = None
        self.device: Any = None
        self.last_use: Any = None
        self.region_bases: Any = None
        specs = dax_regions(devices, offsets, sizes)
        intervals: dict[str, list[tuple[int, int]]] = {}
        # Resolve/validate every definition before opening or mapping any device.
        for index, (device, offset, reserved) in enumerate(specs):
            device = str(Path(device).resolve())
            alignment, capacity = dax_geometry(device)
            if offset % alignment or reserved % alignment:
                raise ValueError(f"DAX offset/bytes must be aligned to {alignment}")
            if offset + reserved > capacity:
                raise ValueError("DAX reserved region is out of bounds")
            previous = intervals.setdefault(device, [])
            if any(
                offset < end and begin < offset + reserved for begin, end in previous
            ):
                raise ValueError("DAX regions on the same device must not overlap")
            previous.append((offset, offset + reserved))
            local_pages = max(0, (pages + len(specs) - 1 - index) // len(specs))
            needed = local_pages * page_size * self.token_bytes
            mapped = (needed + alignment - 1) // alignment * alignment
            if mapped > reserved:
                raise ValueError(
                    "DAX reserved region is too small for its striped pages"
                )
            self.regions.append(
                CXLRegion(device, offset, reserved, local_pages, mapped)
            )
        try:
            for region in self.regions:
                if region.device not in self.fds:
                    fd = os.open(region.device, os.O_RDWR)
                    self.fds[region.device] = fd
                    if not stat.S_ISCHR(os.fstat(fd).st_mode):
                        raise ValueError(
                            "CXL backend requires a devdax character device"
                        )
                    # One advisory lock per device, shared by its disjoint regions.
                    # This does not coordinate other hosts or noncooperating processes.
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if region.mapped_bytes:
                    region.mapping = mmap.mmap(
                        self.fds[region.device],
                        region.mapped_bytes,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                        offset=region.offset,
                    )
        except Exception:
            self.close()
            raise

    def location(self, global_slot):
        page, token_offset = divmod(global_slot, self.page_size)
        local_page, region = divmod(page, len(self.regions))
        return region, local_page * self.page_size + token_offset

    def _initialize_cuda(self, device):
        import torch

        if not self.fds:
            raise RuntimeError("CXL memory pool is closed")
        device = torch.device(device)
        if self.device is not None:
            if device != self.device:
                raise ValueError("CXL mappings belong to a different CUDA device")
            return
        for region in self.regions:
            if region.mapping is not None:
                region.cuda = RegisteredCudaMapping(
                    region.mapping,
                    self.dim,
                    region.pages * self.page_size,
                    device,
                    self.kernel_dir,
                )
                region.stream = torch.cuda.Stream(device=device)
                self.module = region.cuda.module
        # Registration checks that the host and GPU virtual addresses match.
        # Empty regions have no backing pages and cannot be valid write targets.
        self.region_bases = torch.tensor(
            [
                region.cuda.address if region.cuda is not None else 0
                for region in self.regions
            ],
            dtype=torch.int64,
            device=device,
        )
        self.last_use = torch.cuda.Event()
        self.last_use.record(torch.cuda.current_stream(device))
        self.device = device

    def route(self, sources, destinations, counts):
        """Stable GPU-only partition of valid misses; never inspect KV payload."""
        import torch

        self._initialize_cuda(sources.device)
        shape = (len(self.regions), *sources.shape)
        plan = RegionReadPlan(
            torch.empty(shape, dtype=sources.dtype, device=sources.device),
            torch.empty(shape, dtype=destinations.dtype, device=sources.device),
            torch.empty(
                (len(self.regions), len(counts)),
                dtype=counts.dtype,
                device=counts.device,
            ),
        )
        self.module.route_regions(
            sources,
            destinations,
            counts,
            plan.sources,
            plan.destinations,
            plan.counts,
            self.page_size,
        )
        return plan

    def _begin(self, device):
        import torch

        caller = torch.cuda.current_stream(device)
        if self.last_use is not None:
            caller.wait_event(self.last_use)
        ready = torch.cuda.Event()
        ready.record(caller)
        return caller, ready

    def _finish(self, caller, events):
        import torch

        for event in events:
            caller.wait_event(event)
        self.last_use = torch.cuda.Event()
        self.last_use.record(caller)

    def write(self, slots, source):
        """Write trusted pool slots with one kernel, regardless of region count."""
        import torch

        if source.dtype != torch.bfloat16 or len(slots) >= 2**32:
            raise ValueError("CXL CUDA writes require BF16 and fewer than 2**32 rows")
        if not slots:
            return 0
        self._initialize_cuda(source.device)
        with torch.cuda.device(source.device):
            caller = torch.cuda.current_stream(source.device)
            if self.last_use is not None:
                caller.wait_event(self.last_use)
            # Only one flat index vector crosses H2D; KV stays in the original
            # packed HBM tensor. Routing and local-slot arithmetic run on GPU.
            host = torch.tensor(slots, dtype=torch.int64, pin_memory=True)
            global_slots = host.to(source.device, non_blocking=True)
            self.module.write_regions(
                source, global_slots, self.region_bases, self.page_size
            )
            self.last_use = torch.cuda.Event()
            self.last_use.record(caller)
            # CPU-only accounting, vectorized over existing metadata: no Python
            # per-token tuples, transfer groups, or additional GPU readback.
            if len(self.regions) == 1:
                counts = [len(slots)]
            else:
                regions = (host.numpy() // self.page_size) % len(self.regions)
                counts = np.bincount(regions, minlength=len(self.regions))
            # Keep source/indices alive and retain completed-on-return CPU IO.
            caller.synchronize()
        for index, count in enumerate(counts):
            if count:
                self.record(index, int(count), write=True)
        return 1

    def read(self, plan, totals, destination, real_requests, value_offsets=None):
        """All regions write disjoint final HBM rows, then rejoin the caller."""
        import torch

        self._initialize_cuda(destination.device)
        if value_offsets is not None:
            self.module.offset_region_destinations(
                plan.destinations, plan.counts, value_offsets
            )
        caller, ready = self._begin(destination.device)
        events = []
        unused = torch.empty(0, device=destination.device)
        for index, count in enumerate(
            totals if totals is not None else [None] * len(self.regions)
        ):
            if count == 0:
                continue
            region = self.regions[index]
            if region.cuda is None:
                if totals is None:
                    continue
                raise ValueError("Read plan targets a region without allocated pages")
            with torch.cuda.stream(region.stream):
                region.stream.wait_event(ready)
                for tensor in (
                    plan.sources,
                    plan.destinations,
                    plan.counts,
                    destination,
                    real_requests,
                ):
                    tensor.record_stream(region.stream)
                self.module.read(
                    plan.sources[index],
                    plan.destinations[index],
                    plan.counts[index],
                    real_requests,
                    region.cuda.view,
                    unused,
                    destination,
                    unused,
                    4,
                    self.token_bytes,
                )
                done = torch.cuda.Event()
                done.record(region.stream)
                events.append(done)
            if count is not None:
                self.record(index, count, write=False)
        self._finish(caller, events)
        return len(events)

    def record(self, index, tokens, *, write):
        region = self.regions[index]
        if write:
            region.write_bytes += tokens * self.token_bytes
            region.write_operations += 1
        else:
            region.read_bytes += tokens * self.token_bytes
            region.read_operations += 1

    def stats(self):
        return [
            {
                "region": index,
                "dax_device": region.device,
                "dax_offset": region.offset,
                "dax_bytes": region.reserved_bytes,
                "mapped_bytes": region.mapped_bytes,
                "capacity_pages": region.pages,
                "read_bytes": region.read_bytes,
                "write_bytes": region.write_bytes,
                "read_operations": region.read_operations,
                "write_operations": region.write_operations,
            }
            for index, region in enumerate(self.regions)
        ]

    def synchronize(self):
        if self.last_use is not None:
            self.last_use.synchronize()

    def close(self):
        self.synchronize()
        self.region_bases = None
        for region in self.regions:
            if region.cuda is not None:
                region.cuda.close()
                region.cuda = None
            if region.mapping is not None:
                region.mapping.close()
                region.mapping = None
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
