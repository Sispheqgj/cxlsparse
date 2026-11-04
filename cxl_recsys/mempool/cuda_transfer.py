"""Direct HBM <-> registered host mapping through pinned upstream kernels.

This mapping helper retains HiCacheKernel::run_one_mla for standalone writes
and HiSparse copy_cache_planned for reads. Production devdax writes instead use
CXLStorage's project-owned multi-region kernel, sharing these registrations and
the upstream vectorized load/store helpers without modifying SGLang sources.
No KV payload is staged in ordinary DRAM. List-based calls wait by default;
internal GPU-plan reads and explicitly asynchronous writes are stream-ordered.
"""

import logging
import mmap

import torch

from .kernels import DEFAULT_KERNEL_DIR, UPSTREAM_COMMIT, load_transfer

logger = logging.getLogger(__name__)


def load_transfer_kernel(token_bytes: int, kernel_dir=DEFAULT_KERNEL_DIR):
    """Compatibility entry point: load a prebuilt library without compiling."""
    return load_transfer(token_bytes, kernel_dir)


class RegisteredCudaMapping:
    """Own CUDA registration, not the mmap; unregister before the owner unmaps.

    One serialized caller is supported, as in Worker. Errors propagate to the
    caller and invalidate the whole run; no recovery state is maintained.
    """

    def __init__(
        self,
        mapping: mmap.mmap,
        dim: int,
        capacity: int,
        device,
        kernel_dir=DEFAULT_KERNEL_DIR,
    ):
        if capacity <= 0 or dim <= 0 or capacity * dim * 2 > len(mapping):
            raise ValueError("Invalid CXL tensor view extent")
        self.device = torch.device(device)
        self.dim = dim
        self.capacity = capacity
        self.view: torch.Tensor | None = None
        self.address = 0
        with torch.cuda.device(self.device):
            self.module = load_transfer_kernel(dim * 2, kernel_dir)
            view = torch.frombuffer(
                mapping, dtype=torch.bfloat16, count=capacity * dim
            ).view(capacity, dim)
            address = view.data_ptr()
            self.module.register_mapping(address, len(mapping))
            self.view = view
            self.address = address
        logger.info(
            "host_cuda_registered bytes=%d capacity_tokens=%d device=%s upstream=%s",
            len(mapping),
            capacity,
            self.device,
            UPSTREAM_COMMIT,
        )

    def _validate(self, slots, source):
        """O(1) write metadata checks; the caller owns slot bounds/uniqueness."""
        if self.view is None:
            raise RuntimeError("CXL CUDA mapping is closed")
        if (
            source.device != self.device
            or source.dtype != torch.bfloat16
            or source.shape != (len(slots), self.dim)
            or not source.is_contiguous()
            or len(slots) >= 2**32
        ):
            raise ValueError("Invalid CXL CUDA batch tensor or token count")

    def write(self, slots, source, *, synchronize=True):
        self._validate(slots, source)
        if not slots:
            return
        # Sort destination addresses for locality, without staging/reordering KV
        # on the CPU. Source indices retain the caller's row-to-slot mapping.
        order = sorted(range(len(slots)), key=slots.__getitem__)
        with torch.cuda.device(self.device):
            src_indices = torch.tensor(order, dtype=torch.int64, pin_memory=True).to(
                self.device, non_blocking=True
            )
            dst_indices = torch.tensor(
                [slots[index] for index in order], dtype=torch.int64, pin_memory=True
            ).to(self.device, non_blocking=True)
            stream = torch.cuda.current_stream(self.device)
            self.module.transfer(self.view, dst_indices, source, src_indices)
            # Complete stores before the caller releases source tensors or unmaps.
            if synchronize:
                stream.synchronize()

    @staticmethod
    def validate_read(slots, destination, destination_slots, dim, capacity):
        """Validate the linear MLA layout and the upstream int32 copy plan."""
        destination_limit = min(len(destination), 2**31) if destination.ndim else 0
        if (
            destination.device.type != "cuda"
            or destination.dtype != torch.bfloat16
            or destination.ndim != 2
            or destination.shape[1] != dim
            or not destination.is_contiguous()
            or len(slots) != len(destination_slots)
            or len(slots) >= 2**31
            or len(set(slots)) != len(slots)
            or len(set(destination_slots)) != len(destination_slots)
            or any(type(slot) is not int or not 0 <= slot < capacity for slot in slots)
            or any(
                type(slot) is not int or not 0 <= slot < destination_limit
                for slot in destination_slots
            )
        ):
            raise ValueError("Invalid CXL CUDA read tensor or source/destination slots")

    def read(self, slots, destination, destination_slots):
        """Gather sparse CXL slots straight into caller-owned compact HBM rows."""
        if self.view is None:
            raise RuntimeError("CXL CUDA mapping is closed")
        self.validate_read(
            slots, destination, destination_slots, self.dim, self.capacity
        )
        if destination.device != self.device:
            raise ValueError("CXL read destination must use the registered CUDA device")
        if not slots:
            return
        # Compact destination rows are independent of persistent hot-cache slots.
        # Sorting reorders index metadata without changing query-to-value mapping.
        order = sorted(range(len(slots)), key=slots.__getitem__)
        with torch.cuda.device(self.device):
            source_plan = torch.tensor(
                [[slots[index] for index in order]],
                dtype=torch.int64,
                device=self.device,
            )
            destination_plan = torch.tensor(
                [[destination_slots[index] for index in order]],
                dtype=torch.int32,
                device=self.device,
            )
            counts = torch.tensor([len(slots)], dtype=torch.int32, device=self.device)
            real_requests = torch.ones(1, dtype=torch.int32, device=self.device)
            unused = torch.empty(0, device=self.device)
            stream = torch.cuda.current_stream(self.device)
            # Generic linear MLA, not DSv4 page padding; SkipIO=false performs
            # real GPU loads from the registered CXL address. Match the upstream
            # wrapper's fixed four-block grid; warps loop over arbitrary misses.
            self.module.read(
                source_plan,
                destination_plan,
                counts,
                real_requests,
                self.view,
                unused,
                destination,
                unused,
                4,
                self.dim * 2,
            )
            # Keep mapping/plan/destination alive until all device loads finish.
            stream.synchronize()

    def read_plan(self, sources, destinations, counts, real_requests, destination):
        """Consume a trusted device plan without rebuilding it on the CPU.

        HiSparseCache owns these buffers and orders planning, IO, consumption
        and reuse on one CUDA stream. Unlike the public list-based read helper,
        this internal path is stream-ordered, not synchronizing on return.
        """
        if self.view is None:
            raise RuntimeError("CXL CUDA mapping is closed")
        unused = torch.empty(0, device=self.device)
        with torch.cuda.device(self.device):
            self.module.read(
                sources,
                destinations,
                counts,
                real_requests,
                self.view,
                unused,
                destination,
                unused,
                4,
                self.dim * 2,
            )

    def close(self):
        if self.view is None:
            return
        with torch.cuda.device(self.device):
            # If a GPU context is broken, fail closed: keep the mapping/FD owned
            # until process teardown rather than unmap memory still in use.
            torch.cuda.synchronize(self.device)
            self.module.unregister_mapping(self.address)
        self.view = None
        self.address = 0
        logger.info("host_cuda_unregistered device=%s", self.device)
