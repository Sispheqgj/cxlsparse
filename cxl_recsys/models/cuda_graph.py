"""Bounded CUDA graphs for pure tensor functions in a serialized Worker.

Persistent cache mutation and backing-pool I/O stay outside capture. Planner
scratch may be overwritten provided each invocation fully initializes it.
Every invocation refreshes static inputs; outputs are owned copies by default.
Serialized consumers can borrow outputs until that entry's next replay.
"""

from collections import OrderedDict

import torch


class CUDAGraphCache:
    def __init__(self, capacity=4):
        if capacity <= 0:
            raise ValueError("CUDA graph cache capacity must be positive")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.captures = self.replays = self.evictions = 0
        self.last_use = None

    @torch.inference_mode()
    def run(self, key, function, tensors, *, clone_outputs=True):
        signature = tuple((tuple(t.shape), t.dtype, t.device) for t in tensors)
        key = (key, signature)
        current = torch.cuda.current_stream(tensors[0].device)
        if self.last_use is not None:
            current.wait_event(self.last_use)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) == self.capacity:
                current.synchronize()
                _, evicted = self.entries.popitem(last=False)
                evicted[0].reset()
                del evicted
                self.evictions += 1
            stream = torch.cuda.Stream(device=tensors[0].device)
            stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                inputs = tuple(
                    t.clone(memory_format=torch.contiguous_format) for t in tensors
                )
                # JIT compilation, library initialization and allocator setup
                # must finish before capture. Only disposable scratch may mutate.
                for _ in range(2):
                    function(*inputs)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = function(*inputs)
            current.wait_stream(stream)
            entry = (graph, inputs, outputs)
            self.entries[key] = entry
            self.captures += 1
        self.entries.move_to_end(key)
        graph, inputs, outputs = entry
        for destination, source in zip(inputs, tensors):
            destination.copy_(source)
        graph.replay()
        # Borrowing is for serialized consumers that finish before this entry's
        # next replay. The default preserves independently owned results.
        result = tuple(t.clone() for t in outputs) if clone_outputs else outputs
        if self.last_use is None:
            self.last_use = torch.cuda.Event()
        self.last_use.record(current)
        self.replays += 1
        return result

    def stats(self):
        return dict(
            entries=len(self.entries),
            captures=self.captures,
            replays=self.replays,
            evictions=self.evictions,
        )

    def close(self):
        if self.last_use is not None:
            self.last_use.synchronize()
        for graph, _, _ in self.entries.values():
            graph.reset()
        self.entries.clear()
        self.last_use = None
