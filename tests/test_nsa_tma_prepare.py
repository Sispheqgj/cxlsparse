"""Exact source preparation writes, skipped blocks and replayed readiness."""

import pytest
import torch


@pytest.mark.gpu
def test_prepare_sources_dirty_destination_and_changed_graph_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from cxl_recsys.models.nsa.tma.transport import _prepare_sources

    external = (torch.arange(32 * 128).reshape(32, 128) % 97).bfloat16().cuda()
    resident = -(torch.arange(16 * 128).reshape(16, 128) % 71).bfloat16().cuda()
    address = torch.zeros((7, 16), dtype=torch.int64, device="cuda")
    valid = torch.empty(7, dtype=torch.int32, device="cuda")
    missing = torch.empty_like(valid)
    kv = torch.empty((7, 16, 128), dtype=torch.bfloat16, device="cuda")
    sources = torch.empty(7, dtype=torch.int64, device="cuda")
    kinds, ready = torch.empty_like(valid), torch.empty_like(valid)
    external_ptrs = external.data_ptr() + torch.arange(32, dtype=torch.int64) * 256
    resident_ptrs = resident.data_ptr() + torch.arange(16, dtype=torch.int64) * 256

    # Cover all source classes in one launch. Device addresses suffice for the
    # gather contract; test_nsa_tma_transport covers actual mapped-host reads.
    cases = ["inactive", "direct", "resident", "mixed", "tail", "hot_tail", "gather"]

    def configure(order):
        a = torch.zeros((7, 16), dtype=torch.int64)
        v, m, k = [], [], []
        expected = torch.full((7, 16, 128), torch.nan, dtype=torch.bfloat16)
        ext, hot = external.cpu(), resident.cpu()
        for block, kind in enumerate(order):
            count = {"inactive": 0, "tail": 5, "hot_tail": 7}.get(kind, 16)
            miss = kind in {"direct", "mixed", "tail", "gather"}
            v.append(count)
            m.append(int(miss))
            k.append(1 if kind == "direct" else 2 if miss else 0)
            if kind == "direct":
                a[block] = external_ptrs[:16]
            elif kind in {"resident", "hot_tail"}:
                a[block, :count] = resident_ptrs[:count]
                expected[block].zero_()
                expected[block, :count] = hot[:count]
            elif kind == "mixed":
                a[block, ::2] = external_ptrs[:16:2]
                a[block, 1::2] = resident_ptrs[1::2]
                expected[block, ::2] = ext[:16:2]
                expected[block, 1::2] = hot[1::2]
            elif kind == "tail":
                a[block, :count] = external_ptrs[16 : 16 + count]
                expected[block].zero_()
                expected[block, :count] = ext[16 : 16 + count]
            elif kind == "gather":
                a[block] = external_ptrs[:16].flip(0)
                expected[block] = ext[:16].flip(0)
        for destination, value in (
            (address, a),
            (valid, torch.tensor(v)),
            (missing, torch.tensor(m)),
        ):
            destination.copy_(value)
        expected_sources = torch.tensor(
            [
                external.data_ptr() if kind == "direct" else kv.data_ptr() + i * 4096
                for i, kind in enumerate(order)
            ],
            dtype=torch.int64,
        )
        return expected, expected_sources, torch.tensor(k), 1 - torch.tensor(m)

    def prepare():
        # These writes are captured too: every replay starts from invalid data
        # and stale positive ready flags, not the preceding invocation's values.
        kv.fill_(torch.nan)
        sources.fill_(-1)
        kinds.fill_(-1)
        ready.fill_(73)
        _prepare_sources[(7,)](
            address,
            valid,
            missing,
            kv,
            sources,
            kinds,
            ready,
            num_warps=4,
        )

    configure(cases)
    prepare()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare()
    for order in (cases, list(reversed(cases)), cases):
        external.add_(1)
        resident.sub_(1)
        expected, expected_sources, expected_kinds, expected_ready = configure(order)
        graph.replay()
        torch.testing.assert_close(kv.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(sources.cpu(), expected_sources)
        assert torch.equal(kinds.cpu(), expected_kinds)
        assert torch.equal(ready.cpu(), expected_ready)
