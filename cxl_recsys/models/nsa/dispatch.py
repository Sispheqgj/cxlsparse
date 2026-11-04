"""Finite query buckets for NSA's serialized GR executor.

Padding is scratch only: the selected adapter emits invalid references and
positions for every inactive row. Persistent KV writes and hot-set commits must
remain outside functions passed to ``NSABucketExecutor`` because CUDA graph
warmup invokes them more than once. These adapters do not themselves bucket the
history summaries/window: their caller must preserve those causal boundaries.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from math import prod

import torch

from cxl_recsys.models.cuda_graph import CUDAGraphCache

QUERY_BUCKETS = (64, 128, 256, 512, 1024)


class BucketCapacityError(ValueError):
    """The caller must route this request to its explicit safe fallback."""


def query_bucket(active_q, buckets=QUERY_BUCKETS):
    if active_q < 1:
        raise BucketCapacityError("A query bucket requires at least one active row")
    for capacity in buckets:
        if active_q <= capacity:
            return capacity
    raise BucketCapacityError(
        f"Query length {active_q} exceeds bucket limit {buckets[-1]}"
    )


def block_bucket(required, *, maximum, minimum=64):
    """Round an upper bound up without truncating an active physical union."""
    if required < 0 or maximum < 1 or minimum < 1 or required > maximum:
        raise BucketCapacityError("Invalid or overflowing active-block capacity")
    return min(maximum, max(minimum, 1 << max(0, required - 1).bit_length()))


def consumer_groups(capacity, *, max_queries=128):
    """Compute groups only; their transport must share a single physical union."""
    if capacity < 1 or max_queries < 1:
        raise ValueError("Consumer capacities must be positive")
    return tuple(
        (begin, min(begin + max_queries, capacity))
        for begin in range(0, capacity, max_queries)
    )


@dataclass(frozen=True)
class BucketOwner:
    user: object
    layer: int
    backing: object
    epoch: int
    memory_version: int = 0


@dataclass
class _Entry:
    tensors: dict
    bytes: int
    epoch: int = 0
    event: object = None
    leased: bool = False
    padding: dict = field(default_factory=dict)


def _copy_rows(entry, index, source, fill):
    """Refresh real rows; immutable padding only needs clearing after shrink."""
    target = entry.tensors[index]
    active = len(source)
    previous, value = entry.padding.get(index, (len(target), fill))
    target[:active].copy_(source)
    stop = previous if value == fill else len(target)
    if active < stop:
        target[active:stop].fill_(fill)
    entry.padding[index] = (active, fill)


class _BucketStorage:
    def __init__(self, capacity, max_bytes):
        if capacity < 1 or max_bytes < 1:
            raise ValueError("Bucket entry count and byte budget must be positive")
        self.capacity, self.max_bytes = capacity, max_bytes
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.allocations = self.evictions = 0

    def acquire(self, key, templates, shapes):
        entry = self.entries.get(key)
        if entry is not None and entry.leased:
            raise RuntimeError("Bucket inputs are still leased by an earlier consumer")
        if entry is None:
            needed = sum(
                prod(shapes[i]) * value.element_size() for i, value in templates.items()
            )
            if needed > self.max_bytes:
                raise BucketCapacityError(
                    f"Bucket needs {needed} input bytes, budget is {self.max_bytes}"
                )
            while (
                len(self.entries) >= self.capacity
                or self.bytes + needed > self.max_bytes
            ):
                available = next(
                    ((k, e) for k, e in self.entries.items() if not e.leased), None
                )
                if available is None:
                    raise BucketCapacityError("All eviction candidates are leased")
                previous_key, previous = available
                if previous.event is not None:
                    previous.event.synchronize()
                del self.entries[previous_key]
                self.bytes -= previous.bytes
                self.evictions += 1
            entry = _Entry(
                {
                    i: torch.empty(shapes[i], dtype=t.dtype, device=t.device)
                    for i, t in templates.items()
                },
                needed,
            )
            self.entries[key] = entry
            self.bytes += needed
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.allocations += 1
        self.entries.move_to_end(key)
        if entry.event is not None:
            device = next(iter(entry.tensors.values())).device
            torch.cuda.current_stream(device).wait_event(entry.event)
        entry.epoch += 1
        entry.leased = True
        return entry

    def release(self, entry, epoch):
        if not entry.leased or entry.epoch != epoch:
            raise RuntimeError("Stale or already released bucket lease")
        first = next(iter(entry.tensors.values()))
        if first.is_cuda:
            if entry.event is None:
                entry.event = torch.cuda.Event()
            entry.event.record(torch.cuda.current_stream(first.device))
        entry.leased = False

    def close(self):
        if any(entry.leased for entry in self.entries.values()):
            raise RuntimeError("Cannot close bucket storage with live input leases")
        for entry in self.entries.values():
            if entry.event is not None:
                entry.event.synchronize()
        self.entries.clear()
        self.bytes = 0

    def stats(self):
        return dict(
            entries=len(self.entries),
            capacity=self.capacity,
            buffer_bytes=self.bytes,
            peak_buffer_bytes=self.peak_bytes,
            max_buffer_bytes=self.max_bytes,
            allocations=self.allocations,
            evictions=self.evictions,
        )


@dataclass
class BucketLease:
    key: tuple
    active_q: int
    capacity: int
    owner: BucketOwner
    query: torch.Tensor
    blocks: torch.Tensor
    positions: torch.Tensor
    _storage: _BucketStorage
    _entry: _Entry
    _epoch: int
    _released: bool = False

    def assert_valid(self):
        if self._released or self._entry.epoch != self._epoch:
            raise RuntimeError(
                "Borrowed bucket inputs have been released or overwritten"
            )

    def groups(self, max_queries=128):
        self.assert_valid()
        return consumer_groups(self.capacity, max_queries=max_queries)

    def release(self):
        """Call after the last consumer is enqueued on the current CUDA stream."""
        self.assert_valid()
        self._storage.release(self._entry, self._epoch)
        self._released = True

    def __enter__(self):
        self.assert_valid()
        return self

    def __exit__(self, *_):
        self.release()


class NSABucketInputs:
    """Stable read-only inputs; physical IO/KV/MR lifetime belongs to transport.

    Consumers must not mutate these tensors. Inactive rows remain initialized
    between invocations, so a steady effective length requires no padding fill.
    """

    def __init__(self, capacity=5, max_bytes=64 * 2**20):
        self.storage = _BucketStorage(capacity, max_bytes)
        self.calls = 0
        self.last_stats = {}

    @torch.inference_mode()
    def prepare(self, query, blocks, positions, *, owner):
        active_q = len(query)
        capacity = query_bucket(active_q)
        if (
            query.ndim != 4
            or blocks.ndim != 3
            or blocks.shape[:2] != query.shape[:2]
            or positions.shape != (active_q,)
            or blocks.dtype != torch.int32
            or positions.dtype not in (torch.int32, torch.int64)
            or any(t.device != query.device for t in (blocks, positions))
        ):
            raise ValueError("Invalid selected bucket input geometry")
        tensors = (query, blocks, positions)
        shapes = {i: (capacity, *t.shape[1:]) for i, t in enumerate(tensors)}
        key = tuple((shapes[i], t.dtype, t.device) for i, t in enumerate(tensors))
        entry = self.storage.acquire(key, dict(enumerate(tensors)), shapes)
        try:
            for i, source in enumerate(tensors):
                _copy_rows(entry, i, source, 0 if i == 0 else -1)
        except BaseException:
            self.storage.release(entry, entry.epoch)
            raise
        self.calls += 1
        self.last_stats = dict(
            active_q=active_q,
            query_capacity=capacity,
            padded_queries=capacity - active_q,
            owner_epoch=owner.epoch,
            memory_version=owner.memory_version,
        )
        return BucketLease(
            key,
            active_q,
            capacity,
            owner,
            *(entry.tensors[i] for i in range(3)),
            self.storage,
            entry,
            entry.epoch,
        )

    def stats(self):
        return {**self.storage.stats(), "calls": self.calls, **self.last_stats}

    def close(self):
        self.storage.close()


class NSABucketExecutor:
    """NSA-private graph adapter for pure tensor segments.

    ``pad_inputs`` identifies tensors whose first dimension is query length. Other
    inputs retain their shape, so history shape changes still produce new entries.
    ``slice_outputs=()`` retains all capacity rows for a combined prepare segment;
    the caller must slice real rows before any persistent state update. Borrowed
    outputs remain valid only until the next replay of the same entry. Serialized
    same-stream consumers are ordered naturally; call ``mark_consumed`` after the
    last use when handing consumption to another CUDA stream.
    """

    def __init__(
        self,
        capacity=5,
        max_bytes=512 * 2**20,
        *,
        cuda_graphs=True,
        direct_replay=False,
    ):
        self.storage = _BucketStorage(capacity, max_bytes)
        self.graphs = CUDAGraphCache(capacity) if cuda_graphs else None
        self.calls = 0
        self.last_stats = {}
        self.consumer_event = None
        self._graph_bytes = 0
        self._graph_stamp = (0, 0, 0)
        # Opt-in until matched end-to-end measurements justify enabling it.
        self.direct_replay = direct_replay
        self.direct_replays = 0
        self._direct_padding = {}

    def _refresh_graph_bytes(self):
        """Account storage only when a cold capture changes the graph cache."""
        if self.graphs is None:
            return
        stamp = (self.graphs.captures, self.graphs.evictions, len(self.graphs.entries))
        if stamp == self._graph_stamp:
            return
        storages = {}
        for _, inputs, outputs in self.graphs.entries.values():
            for tensor in (*inputs, *outputs):
                storage = tensor.untyped_storage()
                storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
        self._graph_bytes = sum(storages.values())
        self._graph_stamp = stamp
        self._direct_padding = {
            key: state
            for key, state in self._direct_padding.items()
            if key in self.graphs.entries
            and state[0] == id(self.graphs.entries[key][0])
        }

    def _replay_direct(
        self, cache_key, cached, tensors, pad_inputs, fill_values, clone
    ):
        """NSA-private cache hit; the shared graph implementation is unchanged."""
        graph, inputs, outputs = cached
        current = torch.cuda.current_stream(tensors[0].device)
        if self.graphs.last_use is not None:
            current.wait_event(self.graphs.last_use)
        identity, padding = self._direct_padding.get(cache_key, (None, {}))
        if identity != id(graph):
            padding = {}
        entry = _Entry({i: inputs[i] for i in pad_inputs}, 0, padding=padding)
        for i, source in enumerate(tensors):
            if i in pad_inputs:
                _copy_rows(entry, i, source, fill_values.get(i, 0))
            else:
                inputs[i].copy_(source)
        self._direct_padding[cache_key] = (id(graph), padding)
        self.graphs.entries.move_to_end(cache_key)
        graph.replay()
        outputs = tuple(t.clone() for t in outputs) if clone else outputs
        if self.graphs.last_use is None:
            self.graphs.last_use = torch.cuda.Event()
        self.graphs.last_use.record(current)
        self.graphs.replays += 1
        self.direct_replays += 1
        return outputs

    @staticmethod
    def _slice_outputs(outputs, active_q, capacity, indices):
        selected = range(len(outputs)) if indices is None else indices
        if any(outputs[i].ndim < 1 or len(outputs[i]) != capacity for i in selected):
            raise ValueError("Sliced output must preserve the query capacity")
        return tuple(
            t[:active_q] if i in selected else t for i, t in enumerate(outputs)
        )

    @torch.inference_mode()
    def run(
        self,
        key,
        function,
        tensors,
        *,
        active_q,
        pad_inputs=None,
        fill_values=None,
        slice_outputs=None,
        clone_outputs=False,
    ):
        capacity = query_bucket(active_q)
        pad_inputs = tuple(range(len(tensors))) if pad_inputs is None else pad_inputs
        if not pad_inputs or len(set(pad_inputs)) != len(pad_inputs):
            raise ValueError("At least one distinct query input must be padded")
        fill_values = {} if fill_values is None else fill_values
        for i in pad_inputs:
            if tensors[i].ndim < 1 or len(tensors[i]) != active_q:
                raise ValueError("Padded input does not match active query length")
        if any(t.device != tensors[0].device for t in tensors):
            raise ValueError("Bucket inputs must use the same device")
        if self.consumer_event is not None:
            torch.cuda.current_stream(tensors[0].device).wait_event(self.consumer_event)
        shapes = {i: (capacity, *tensors[i].shape[1:]) for i in pad_inputs}
        signature = tuple(
            (shapes.get(i, tuple(t.shape)), t.dtype, t.device)
            for i, t in enumerate(tensors)
        )
        cache_key = (key, signature)
        cached = (
            self.graphs.entries.get(cache_key)
            if self.direct_replay and self.graphs is not None and tensors[0].is_cuda
            else None
        )
        if cached is not None:
            outputs = self._replay_direct(
                cache_key, cached, tensors, pad_inputs, fill_values, clone_outputs
            )
            outputs = self._slice_outputs(outputs, active_q, capacity, slice_outputs)
            self.calls += 1
            self.last_stats = dict(active_q=active_q, query_capacity=capacity)
            return outputs
        # Staging is copied into each graph's own inputs before replay. Its
        # serialized lease therefore permits sharing across layers/phases with
        # identical geometry; graph keys still bind the distinct layer weights.
        storage_key = (tuple(pad_inputs), signature)
        entry = self.storage.acquire(
            storage_key, {i: tensors[i] for i in pad_inputs}, shapes
        )
        inputs = list(tensors)
        try:
            for i in pad_inputs:
                _copy_rows(entry, i, tensors[i], fill_values.get(i, 0))
                inputs[i] = entry.tensors[i]
            if self.graphs is not None and tensors[0].is_cuda:
                outputs = self.graphs.run(
                    key, function, tuple(inputs), clone_outputs=clone_outputs
                )
                self._refresh_graph_bytes()
                if self.direct_replay:
                    self._direct_padding[cache_key] = (
                        id(self.graphs.entries[cache_key][0]),
                        {i: (active_q, fill_values.get(i, 0)) for i in pad_inputs},
                    )
            else:
                outputs = tuple(function(*inputs))
                if clone_outputs:
                    outputs = tuple(t.clone() for t in outputs)
                else:
                    # An eager identity/view result could alias shared staging.
                    # Preserve the phase-independent output ownership provided
                    # by distinct graph entries, including nonzero-offset views.
                    staging = {
                        t.untyped_storage().data_ptr() for t in entry.tensors.values()
                    }
                    outputs = tuple(
                        t.clone() if t.untyped_storage().data_ptr() in staging else t
                        for t in outputs
                    )
            outputs = self._slice_outputs(outputs, active_q, capacity, slice_outputs)
        finally:
            self.storage.release(entry, entry.epoch)
        self.calls += 1
        self.last_stats = dict(active_q=active_q, query_capacity=capacity)
        return outputs

    def mark_consumed(self, device):
        """Record after the last borrowed-output consumer on its actual stream."""
        if torch.device(device).type == "cuda":
            if self.consumer_event is None:
                self.consumer_event = torch.cuda.Event()
            self.consumer_event.record(torch.cuda.current_stream(device))

    def stats(self):
        graph_stats = (
            self.graphs.stats()
            if self.graphs is not None
            else dict(entries=0, captures=0, replays=0, evictions=0)
        )
        # This O(1) accessor also observes an external graphs.close(). Referenced
        # storage excludes graph allocator/driver bookkeeping; report allocator
        # memory separately in experiments.
        if not graph_stats["entries"]:
            self._graph_bytes = 0
            self._graph_stamp = (graph_stats["captures"], graph_stats["evictions"], 0)
        return {
            **self.storage.stats(),
            "calls": self.calls,
            "graphs": graph_stats,
            "graph_tensor_bytes": self._graph_bytes,
            "direct_replay": self.direct_replay,
            "direct_replays": self.direct_replays,
            **self.last_stats,
        }

    def close(self):
        if self.consumer_event is not None:
            self.consumer_event.synchronize()
        if self.graphs is not None:
            self.graphs.close()
        self.storage.close()
        self.consumer_event = None
        self._graph_bytes = 0
        self._direct_padding.clear()
        if self.graphs is not None:
            self._graph_stamp = (self.graphs.captures, self.graphs.evictions, 0)
