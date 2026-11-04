"""Real one-sided RC verbs transport; gRPC exchanges MR/QP metadata only."""

import ctypes as C
import fcntl
import hashlib
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from ..common.rpc import RPC

logger = logging.getLogger(__name__)


def ordered_cuda_transfers(sources, destinations, *, write=False):
    """Keep repeated reads contiguous in each independent destination span.

    Sorting repeated source tokens together would interleave query-private
    destinations and turn block copies into one work request per token.
    Unique sources and writes retain the existing source-ordered path.
    """
    order = np.argsort(sources, kind="stable")
    sorted_sources = sources[order]
    if not write and np.any(sorted_sources[1:] == sorted_sources[:-1]):
        order = np.argsort(destinations, kind="stable")
    return (
        np.ascontiguousarray(sources[order]),
        np.ascontiguousarray(destinations[order]),
    )


class Endpoint(C.Structure):
    _fields_ = [
        ("addr", C.c_uint64),
        ("size", C.c_uint64),
        ("rkey", C.c_uint32),
        ("qpn", C.c_uint32),
        ("psn", C.c_uint32),
        ("lid", C.c_uint32),
        ("mtu", C.c_uint32),
        ("gid", C.c_uint8 * 16),
        ("max_rd_atomic", C.c_uint32),
        ("max_dest_rd_atomic", C.c_uint32),
    ]

    def as_dict(self):
        endpoint = {}
        for field in self._fields_:
            # ctypes field definitions can also include a third bit-width entry.
            name = field[0]
            if name == "gid":
                endpoint[name] = list(self.gid)
            else:
                endpoint[name] = int(getattr(self, name))
        return endpoint

    @classmethod
    def parse(cls, data):
        e = cls()
        for field in cls._fields_:
            name = field[0]
            if name == "gid":
                e.gid[:] = data[name]
            else:
                setattr(e, name, int(data[name]))
        return e


def library():
    source = Path(__file__).with_name("native") / "verbs.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder = Path(tempfile.gettempdir()) / f"cxl-recsys-verbs-{os.getuid()}"
    folder.mkdir(mode=0o700, exist_ok=True)
    target = folder / f"verbs-{digest}.so"
    with (folder / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            subprocess.run(
                [
                    "gcc",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-libverbs",
                    "-o",
                    str(target),
                ],
                check=True,
            )
    lib = C.CDLL(str(target))
    lib.cr_error.restype = C.c_char_p
    lib.cr_open.argtypes = [
        C.c_char_p,
        C.c_int,
        C.c_int,
        C.c_size_t,
        C.POINTER(Endpoint),
    ]
    lib.cr_open.restype = C.c_void_p
    lib.cr_close.argtypes = [C.c_void_p]
    lib.cr_connect.argtypes = [C.c_void_p, C.POINTER(Endpoint)]
    lib.cr_io.argtypes = [
        C.c_void_p,
        C.POINTER(Endpoint),
        C.c_uint64,
        C.c_size_t,
        C.c_int,
        C.c_int,
    ]
    lib.cr_register_gpu.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_int]
    lib.cr_register_gpu.restype = C.c_void_p
    lib.cr_deregister_gpu.argtypes = [C.c_void_p]
    lib.cr_gpu_io.argtypes = [
        C.c_void_p,
        C.POINTER(Endpoint),
        C.c_void_p,
        C.c_uint64,
        C.c_uint64,
        C.c_uint64,
        C.c_void_p,
        C.c_void_p,
        C.c_size_t,
        C.c_size_t,
        C.c_int,
        C.c_int,
        C.POINTER(C.c_uint64),
    ]
    lib.cr_gpu_io_timed.argtypes = [
        *lib.cr_gpu_io.argtypes,
        C.POINTER(C.c_uint64),
    ]
    return lib


class CudaDriver:
    """Export caller-owned HBM as DMA-BUF; never map or copy its payload on CPU."""

    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")
        self.lib.cuGetErrorString.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
        self.lib.cuPointerSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_uint64]
        self.lib.cuMemGetHandleForAddressRange.argtypes = [
            C.c_void_p,
            C.c_uint64,
            C.c_size_t,
            C.c_int,
            C.c_uint64,
        ]
        self.lib.cuFlushGPUDirectRDMAWrites.argtypes = [C.c_int, C.c_int]

    def check(self, result, operation):
        if result:
            message = C.c_char_p()
            self.lib.cuGetErrorString(result, C.byref(message))
            detail = message.value.decode() if message.value else str(result)
            raise RuntimeError(
                f"{operation}: {detail}; GPUDirect RDMA required, no CPU fallback"
            )

    def export(self, tensor):
        # PyTorch suballocations can start inside a host page. Export the page-
        # aligned range, then preserve the byte offset in every NIC work request.
        page = os.sysconf("SC_PAGESIZE")
        pointer = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        base = pointer // page * page
        extent = ((pointer + size + page - 1) // page) * page - base
        flag = C.c_uint(1)
        self.check(
            self.lib.cuPointerSetAttribute(C.byref(flag), 6, pointer),
            "cuPointerSetAttribute(SYNC_MEMOPS)",
        )
        fd = C.c_int(-1)
        self.check(
            self.lib.cuMemGetHandleForAddressRange(C.byref(fd), base, extent, 1, 0),
            "cuMemGetHandleForAddressRange(DMA_BUF)",
        )
        return fd.value, base, extent, pointer - base

    def flush(self):
        # CQ completion precedes this CUDA visibility barrier and all consumers.
        self.check(
            self.lib.cuFlushGPUDirectRDMAWrites(0, 100),
            "cuFlushGPUDirectRDMAWrites(TO_OWNER)",
        )


class RegisteredGPUBuffer:
    """Explicit MR ownership keeps the allocation alive until deregistration.

    This is not a pointer-address cache: the original Tensor and its storage
    are retained, and each transfer must name exactly that allocation/range.
    """

    def __init__(self, region, tensor):
        import torch

        if (
            tensor.device.type != "cuda"
            or tensor.ndim != 2
            or not tensor.is_contiguous()
        ):
            raise ValueError("GPU registration requires a contiguous 2-D CUDA tensor")
        self.region, self.tensor = region, tensor
        self.driver = CudaDriver()
        self.mr = None
        self.fd = None
        self.storage = tensor.untyped_storage()._cdata
        with torch.cuda.device(tensor.device):
            self.fd, self.base, self.extent, self.offset = self.driver.export(tensor)
            self.mr = region.lib.cr_register_gpu(
                region.handle, self.base, self.extent, self.fd
            )
        if not self.mr:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError(region.lib.cr_error().decode())
        region.registrations.add(self)

    def validate(self, region, tensor):
        if (
            not self.mr
            or self.region is not region
            or tensor.untyped_storage()._cdata != self.storage
            or tensor.data_ptr() != self.tensor.data_ptr()
            or tensor.numel() * tensor.element_size()
            != self.tensor.numel() * self.tensor.element_size()
            or tensor.device != self.tensor.device
        ):
            raise ValueError(
                "Persistent MR must match its owned GPU allocation and range"
            )

    def close(self):
        if self.mr:
            import torch

            # Drain all serialized callers, including the dedicated IO stream.
            torch.cuda.synchronize(self.tensor.device)
            if self.region.lib.cr_deregister_gpu(self.mr):
                raise RuntimeError(self.region.lib.cr_error().decode())
            self.mr = None
            os.close(self.fd)
            self.fd = None
            self.region.registrations.discard(self)
            self.tensor = None


class Region:
    def __init__(self, size, device="mlx5_0", port=1, gid_index=0):
        # mlx5 inline CQ payload handling copies through CPU memcpy, which cannot
        # address HBM. Disable it for this process before creating any QP.
        os.environ["MLX5_SCATTER_TO_CQE"] = "0"
        self.lib, self.endpoint = library(), Endpoint()
        self.handle = self.lib.cr_open(
            device.encode(), port, gid_index, size, C.byref(self.endpoint)
        )
        self.failed = False
        self.registrations = set()
        if not self.handle:
            raise RuntimeError(self.lib.cr_error().decode())
        logger.debug("rdma_region_open bytes=%d device=%s port=%d", size, device, port)

    def connect(self, peer):
        self.peer = Endpoint.parse(peer)
        if self.lib.cr_connect(self.handle, C.byref(self.peer)):
            raise RuntimeError(self.lib.cr_error().decode())
        # Native connect queries the actual QP and verifies both maxima.
        logger.info(
            "rdma_connected max_rd_atomic=%d max_dest_rd_atomic=%d",
            min(self.endpoint.max_rd_atomic, self.peer.max_dest_rd_atomic),
            min(self.endpoint.max_dest_rd_atomic, self.peer.max_rd_atomic),
        )

    def transfer(self, offset, tensor, write):
        if self.failed or not self.handle:
            raise RuntimeError("RDMA session unusable; restart Worker")
        n = tensor.numel() * tensor.element_size()
        if (
            tensor.device.type != "cpu"
            or not tensor.is_contiguous()
            or n > self.endpoint.size
        ):
            raise ValueError("RDMA staging tensor must be contiguous CPU and fit MR")
        if write:
            C.memmove(self.endpoint.addr, tensor.data_ptr(), n)
        if self.lib.cr_io(self.handle, C.byref(self.peer), offset, n, write, 5000):
            self.failed = True
            logger.error(
                "rdma_transfer_failed operation=%s offset=%d bytes=%d",
                "write" if write else "read",
                offset,
                n,
            )
            raise RuntimeError(self.lib.cr_error().decode())
        if not write:
            C.memmove(tensor.data_ptr(), self.endpoint.addr, n)
        logger.debug(
            "rdma_transfer_completed operation=%s offset=%d bytes=%d",
            "write" if write else "read",
            offset,
            n,
        )

    def register_cuda(self, tensor):
        if self.failed or not self.handle:
            raise RuntimeError("RDMA session unusable; restart Worker")
        return RegisteredGPUBuffer(self, tensor)

    def transfer_cuda(
        self, slots, tensor, write, *, rows=None, measurements=None, registration=None
    ):
        """NIC <-> actual CUDA tensor, with only slot/row metadata on the CPU.

        Synchronize producer work before posting, wait for every CQ completion,
        and establish CUDA visibility before returning to attention/update_hot.
        Registration is per call unless an explicitly owned persistent MR is
        supplied. Address reuse cannot attach a new allocation to an old MR.
        """
        import torch

        started_ns = time.perf_counter_ns() if measurements is not None else 0
        if self.failed or not self.handle:
            raise RuntimeError("RDMA session unusable; restart Worker")
        if (
            tensor.device.type != "cuda"
            or tensor.ndim != 2
            or not tensor.is_contiguous()
        ):
            raise ValueError("GPUDirect requires a contiguous 2-D CUDA tensor")
        if not len(slots):
            return 0
        sources = np.asarray(slots, dtype=np.uint64)
        destinations = (
            np.arange(len(sources), dtype=np.uint64)
            if rows is None
            else np.asarray(rows, dtype=np.uint64)
        )
        if sources.ndim != 1 or destinations.shape != sources.shape:
            raise ValueError("GPUDirect requires equally sized slot/row vectors")
        sources, destinations = ordered_cuda_transfers(
            sources, destinations, write=write
        )
        token_bytes = tensor.shape[1] * tensor.element_size()
        driver = CudaDriver()
        with torch.cuda.device(tensor.device):
            stream = torch.cuda.current_stream(tensor.device)
            tensor.record_stream(stream)
            stream.synchronize()
            if registration is None:
                fd, base, extent, offset = driver.export(tensor)
            else:
                registration.validate(self, tensor)
                fd, base, extent, offset = (
                    registration.fd,
                    registration.base,
                    registration.extent,
                    registration.offset,
                )
            mr = None
            try:
                mr = (
                    self.lib.cr_register_gpu(self.handle, base, extent, fd)
                    if registration is None
                    else registration.mr
                )
                if not mr:
                    raise RuntimeError(self.lib.cr_error().decode())
                operations = C.c_uint64()
                transport_ns = C.c_uint64()
                io_args = (
                    self.handle,
                    C.byref(self.peer),
                    mr,
                    base,
                    offset,
                    tensor.numel() * tensor.element_size(),
                    sources.ctypes.data,
                    destinations.ctypes.data,
                    len(sources),
                    token_bytes,
                    write,
                    5000,
                    C.byref(operations),
                )
                # Optional native timing excludes registration and Python setup;
                # the underlying WR/CQ loop is shared with the normal API.
                with torch.cuda.nvtx.range(
                    "rdma.transport.write" if write else "rdma.transport.read"
                ):
                    result = (
                        self.lib.cr_gpu_io_timed(*io_args, C.byref(transport_ns))
                        if measurements is not None
                        else self.lib.cr_gpu_io(*io_args)
                    )
                if result:
                    self.failed = True
                    raise RuntimeError(self.lib.cr_error().decode())
                if not write:
                    driver.flush()
            finally:
                try:
                    if registration is None and mr and self.lib.cr_deregister_gpu(mr):
                        raise RuntimeError(self.lib.cr_error().decode())
                finally:
                    if registration is None:
                        os.close(fd)
        if measurements is not None:
            call_ms = (time.perf_counter_ns() - started_ns) / 1e6
            transport_ms = transport_ns.value / 1e6
            measurements.update(
                call_ms=call_ms,
                transport_ms=transport_ms,
                preparation_ms=call_ms - transport_ms,
                wr_count=operations.value,
            )
        logger.debug(
            "rdma_gpu_transfer operation=%s tokens=%d bytes=%d operations=%d "
            "transport=gpudirect_dmabuf cpu_staging_bytes=0",
            "write" if write else "read",
            len(sources),
            len(sources) * token_bytes,
            operations.value,
        )
        return operations.value

    def close(self):
        if self.handle:
            for registration in list(self.registrations):
                registration.close()
            self.lib.cr_close(self.handle)
            self.handle = None
            logger.debug("rdma_region_closed")


class PoolManager:
    """Session-scoped MR ownership; bounded memory, explicit Close, no expiry.

    No unsafe expiry/reuse while a remote QP might still write. Crashed worker
    allocations remain reserved until manager restart (documented MVP limit).
    """

    def __init__(self, max_bytes, device="mlx5_0", port=1, gid_index=0):
        self.max_bytes = max_bytes
        self.device = device
        self.port = port
        self.gid_index = gid_index
        self.sessions = {}
        self.lock = threading.Lock()

    def open(self, request):
        size = request["bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError("bytes must be a positive integer")
        with self.lock:
            used = sum(r.endpoint.size for r in self.sessions.values())
            if used + size > self.max_bytes:
                logger.warning(
                    "rdma_pool_exhausted requested_bytes=%d used_bytes=%d capacity_bytes=%d",
                    size,
                    used,
                    self.max_bytes,
                )
                raise ValueError("RDMA pool capacity exhausted")
            r = Region(size, self.device, self.port, self.gid_index)
            try:
                r.connect(request["endpoint"])
            except Exception:
                r.close()
                raise
            sid = uuid.uuid4().hex
            self.sessions[sid] = r
            logger.info(
                "rdma_session_opened bytes=%d sessions=%d allocated_bytes=%d",
                size,
                len(self.sessions),
                used + size,
            )
            return {"session_id": sid, "endpoint": r.endpoint.as_dict()}

    def release(self, request):
        with self.lock:
            r = self.sessions.pop(request["session_id"], None)
            if r:
                r.close()
                logger.info("rdma_session_released sessions=%d", len(self.sessions))
        return {"closed": True}

    def status(self, _):
        with self.lock:
            return {
                "sessions": len(self.sessions),
                "allocated_bytes": sum(r.endpoint.size for r in self.sessions.values()),
                "capacity_bytes": self.max_bytes,
                "transport": "ibverbs-rc",
            }

    def close(self):
        with self.lock:
            for r in self.sessions.values():
                r.close()
            self.sessions.clear()


class RemoteMemory:
    def __init__(self, target, size, token_bytes, device="mlx5_0", port=1, gid_index=0):
        # Only the server owns DRAM. The client registers caller-owned HBM per IO.
        self.rpc, self.region, self.sid = (
            RPC(target),
            Region(0, device, port, gid_index),
            None,
        )
        try:
            reply = self.rpc.call(
                "Open", {"bytes": size, "endpoint": self.region.endpoint.as_dict()}
            )
            self.sid = reply["session_id"]
            self.region.connect(reply["endpoint"])
            logger.info(
                "rdma_session_connected transport=gpudirect_dmabuf "
                "remote_bytes=%d cpu_staging_bytes=0",
                size,
            )
        except Exception:
            self.close()
            raise

    def close(self):
        self.region.close()  # QP destroyed before remote MR can be released/reused.
        try:
            if self.sid:
                self.rpc.call("Close", {"session_id": self.sid})
        finally:
            self.sid = None
            self.rpc.close()
