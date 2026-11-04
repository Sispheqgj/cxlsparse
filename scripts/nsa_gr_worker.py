"""CXL-safe benchmark Worker; preserve graphs across mixed revisit lengths."""

import os
import sys
from pathlib import Path

import torch

if os.environ.get("NSA_RUNTIME_SNAPSHOT"):
    sys.path.insert(0, os.environ["NSA_RUNTIME_SNAPSHOT"])

from cxl_recsys.models.nsa import NSAModel
from scripts.nsa_benchmark_worker import main

original_forward = NSAModel.forward
original_execution_stats = NSAModel._execution_stats


def execution_stats(self):
    result = original_execution_stats(self)
    attention = getattr(self.selected_transport, "attention_graphs", None)
    result["selected_attention_graphs"] = attention.stats() if attention else None
    # Per-request timing excludes allocator telemetry for both variants.
    result["cuda_memory"] = None
    return result


def forward(self, input_ids, *args, **kwargs):
    contexts = kwargs.get("caches", ())
    append = any(
        c.cacheable_prefix_length
        > len(self.users.get(c.cache_key, {}).get("tokens", ()))
        for c in contexts
    )
    # Prefill grows summaries every chunk. Release its old graphs between setup
    # requests, and once at the transition to suffix replay. Never clear graphs
    # because effective Q changes during the measured mixed-length workload.
    if append or getattr(self, "_gr_benchmark_append", False):
        for graph in (self.compute_graphs, self.graphs, self.cache.plan_graphs):
            if graph is not None:
                graph.close()
    self._gr_benchmark_append = append
    output = original_forward(self, input_ids, *args, **kwargs)
    destination = os.environ.get("NSA_GR_CAPTURE_OUTPUTS")
    if destination and not append:
        # Diagnostic-only snapshots. Normal performance runs never enable this;
        # the first request of each length then includes recording overhead.
        captured = getattr(self, "_gr_captured_lengths", set())
        q = len(output)
        if q not in captured:
            path = Path(destination)
            path.mkdir(exist_ok=True)
            torch.save(output.detach().cpu(), path / f"q-{q}.pt")
            captured.add(q)
            self._gr_captured_lengths = captured
    return output


def install_shutdown_memory_stats(destination):
    original_close = NSAModel.close

    def close(self):
        import json
        import time

        begin = time.perf_counter_ns()
        stats = torch.cuda.memory_stats()
        record = dict(
            allocated_bytes=stats.get("allocated_bytes.all.current", 0),
            reserved_bytes=stats.get("reserved_bytes.all.current", 0),
            peak_allocated_bytes=stats.get("allocated_bytes.all.peak", 0),
            timing="Worker shutdown after all requests; outside E2E and execution_ms",
            collection_ms=(time.perf_counter_ns() - begin) / 1e6,
        )
        Path(destination, "shutdown-memory.json").write_text(
            json.dumps(record, indent=2)
        )
        transport = self.selected_transport
        if (
            transport is not None
            and getattr(transport, "consumer_backend", None) == "tma"
        ):
            evidence = []
            for index, work in enumerate(transport.workspaces.values()):
                kinds = (
                    torch.bincount(work["tma_source_kinds"], minlength=3).cpu().tolist()
                )
                name = f"worker-tma-{index}.cu"
                Path(destination, name).write_text(work["kernel"].get_kernel_source())
                evidence.append(
                    dict(
                        source=name,
                        rows=work["rows"],
                        blocks=work["physical"],
                        consumer_ctas=work["consumer_ctas"],
                        io_ctas=work["io_ctas"],
                        ready_window=work.get("ready_window", 1),
                        last_plan_source_kinds=dict(
                            resident_or_inactive=kinds[0],
                            full_block_tma=kinds[1],
                            mixed_or_tail=kinds[2],
                        ),
                    )
                )
            Path(destination, "shutdown-tma.json").write_text(
                json.dumps(evidence, indent=2)
            )
        return original_close(self)

    NSAModel.close = close


if __name__ == "__main__":
    NSAModel.forward = forward
    NSAModel._execution_stats = execution_stats
    install_shutdown_memory_stats(Path(sys.argv[1]))
    main()
