"""FLA serving adaptation: causal suffixes, complete blocks, ties and graphs."""

import pytest
import torch
import torch.nn.functional as F

from cxl_recsys.models.nsa.fla import compressed_attention_select


@pytest.mark.gpu
@pytest.mark.parametrize("length", [1, 15, 16, 17, 255, 513])
def test_fla_compression_and_selection_match_resident_with_suffix(length):
    from third_party.nsa_reference import forward_only as fla

    torch.manual_seed(length)
    q = torch.randn(length, 2, 16, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(length // 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(length, device="cuda")
    padded = F.pad(kv, (0, 0, 0, 0, 0, (length + 15) // 16 - len(kv)))
    k, v = (x[None].contiguous() for x in padded.split(64, -1))
    resident_q = q.reshape(1, length, 32, 64)
    expected, lse = fla.parallel_nsa_compression_fwd(resident_q, k, v, 16, 64**-0.5)
    selected = fla.parallel_nsa_topk(resident_q, k, lse, 16, 16, 64**-0.5)[0]
    width = min(16, (length + 15) // 16)
    for begin in sorted({0, length // 2, length - 1}):
        actual, blocks = compressed_attention_select(
            q[begin:], kv, positions[begin:], length, 16, 16
        )
        torch.testing.assert_close(
            actual, expected[0, begin:].reshape_as(actual), atol=0.015, rtol=0.015
        )
        # Kernel ordering can differ only on exact score ties, so compare sets.
        assert torch.equal(
            blocks.sort(-1).values, selected[begin:, :, :width].sort(-1).values
        )
        assert ((blocks * 16 <= positions[begin:, None, None]) | (blocks == -1)).all()


@pytest.mark.gpu
def test_fla_topk_deterministic_ties_and_future_invisibility():
    q = torch.zeros(3, 2, 16, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(40, 2, 128, device="cuda", dtype=torch.bfloat16)
    pos = torch.tensor([17, 95, 401], device="cuda")
    first, blocks = compressed_attention_select(q, kv, pos, 640, 16, 16)
    for row, p in enumerate((17, 95, 401)):
        current = p // 16
        expected = [current] + list(range(min(current, 15)))
        expected += [-1] * (16 - len(expected))
        assert blocks[row, 0].tolist() == expected
        assert torch.equal(blocks[row, 0], blocks[row, 1])
    kv[26:] = 1e4
    second, other = compressed_attention_select(q, kv, pos, 640, 16, 16)
    assert torch.equal(first, second)
    assert torch.equal(blocks, other)


@pytest.mark.gpu
def test_cuda_graph_refreshes_inputs_owns_outputs_and_evicts():
    from cxl_recsys.models.cuda_graph import CUDAGraphCache

    cache = CUDAGraphCache(capacity=1)
    try:

        def fn(a, b):
            return a @ b, a.sum(-1)

        x = torch.randn(16, 32, device="cuda")
        y = torch.randn(32, 8, device="cuda")
        first = cache.run("matmul", fn, (x, y))
        saved = tuple(t.clone() for t in first)
        x.add_(1)
        for actual, expected in zip(cache.run("matmul", fn, (x, y)), fn(x, y)):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(first, saved):
            assert torch.equal(actual, expected)
        assert cache.stats() == dict(entries=1, captures=1, replays=2, evictions=0)
        cache.run("matmul", fn, (x[:3], y))
        assert cache.stats() == dict(entries=1, captures=2, replays=3, evictions=1)
    finally:
        cache.close()
    assert cache.stats()["entries"] == 0
