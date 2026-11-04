"""NSA causality, incremental summaries and independent upstream GPU oracles."""

import pytest
import torch
import torch.nn.functional as F

from cxl_recsys.models.nsa import ops


def test_compression_append_boundaries_and_ephemeral_suffix():
    torch.manual_seed(6)
    values = torch.randn(199, 2, 128).bfloat16()
    compressed, tail, start = values[:0], values[:0], 0
    for stop in (1, 15, 16, 17, 31, 32, 65, 129, 199):
        compressed, tail = ops.append_compressed(
            compressed, tail, values[start:stop], 16
        )
        expected = (
            values[: stop // 16 * 16].reshape(-1, 16, 2, 128).float().mean(1).bfloat16()
        )
        assert torch.equal(compressed, expected)
        assert torch.equal(tail, values[stop // 16 * 16 : stop])
        before = tail.clone()
        ops.append_compressed(compressed, tail, values[:21], 16)
        assert torch.equal(before, tail)
        start = stop


@pytest.mark.gpu
@pytest.mark.parametrize("length", [1, 15, 16, 17, 255, 513])
def test_three_branches_against_upstream_fla_kernels(length):
    from third_party.nsa_reference import forward_only as fla

    torch.manual_seed(length)
    shape = (1, length, 2, 64)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn((1, length, 32, 64), device="cuda", dtype=torch.bfloat16)
    grouped = query[0].reshape(length, 2, 16, 64)
    padded_k = F.pad(key, (0, 0, 0, 0, 0, (-length) % 16))
    padded_v = F.pad(value, (0, 0, 0, 0, 0, (-length) % 16))
    ck = padded_k.reshape(1, -1, 16, 2, 64).float().mean(2).bfloat16()
    cv = padded_v.reshape(1, -1, 16, 2, 64).float().mean(2).bfloat16()
    positions = torch.arange(length, device="cuda")
    cmp, selected = ops.compressed_attention_select(
        grouped, ck[0, : length // 16], cv[0, : length // 16], positions, length, 16, 16
    )
    expected_cmp, lse = fla.parallel_nsa_compression_fwd(query, ck, cv, 16, 64**-0.5)
    expected_blocks = fla.parallel_nsa_topk(query, ck, lse, 16, 16, 64**-0.5)
    # Regression for the upstream merge-width bug at block_size == block_counts:
    # crossing 16 blocks must still retain all 16 unique causal candidates.
    counts = (positions // 16 + 1).clamp_max(16)[:, None].expand(-1, 2)
    assert torch.equal((selected >= 0).sum(-1), counts)
    assert torch.equal((expected_blocks[0] >= 0).sum(-1), counts)
    expected_slc, _ = fla.parallel_nsa_fwd(
        query, key, value, expected_blocks, 16, 16, 64**-0.5
    )
    records = torch.cat((key[0], value[0]), -1)
    logical = ops.expand_blocks(selected, positions, 16)
    assert ((logical <= positions[:, None, None]) | (logical == -1)).all()
    # Fetch from a resident compact buffer to isolate attention from transport.
    ids = logical * 2 + torch.arange(2, device="cuda")[None, :, None]
    ids.masked_fill_(logical < 0, -1)
    slc = ops.indexed_attention(
        grouped.flatten(0, 1), records.flatten(0, 1), ids.flatten(0, 1), 32
    ).reshape_as(query)
    # Different BF16 rounding in Triton and FP32 reference GEMMs is expected.
    for actual, expected in (
        (cmp.reshape_as(query), expected_cmp),
        (slc, expected_slc),
    ):
        error = (actual.float() - expected.float()).square().mean().sqrt()
        base = expected.float().square().mean().sqrt().clamp_min(1e-6)
        assert error / base < 0.015
    window = ops.window_attention(grouped, records, positions, 0, 128, 32).reshape_as(
        query
    )
    full_k = key.repeat_interleave(16, dim=2).transpose(1, 2).float()
    full_v = value.repeat_interleave(16, dim=2).transpose(1, 2).float()
    mask = (positions[None, :] <= positions[:, None]) & (
        positions[None, :] > positions[:, None] - 128
    )
    reference = (
        F.scaled_dot_product_attention(
            query.transpose(1, 2).float(), full_k, full_v, attn_mask=mask
        )
        .transpose(1, 2)
        .bfloat16()
    )
    # BF16 SDPA vs the independent FP32 full-matrix window oracle. Relative
    # RMS avoids interpreting a single BF16 rounding boundary as a causal bug.
    relative = (
        window.float() - reference.float()
    ).square().mean() / reference.float().square().mean()
    assert relative.sqrt() < 0.004


def test_suffix_selection_is_causal_and_independent_of_future_summaries():
    torch.manual_seed(70)
    key = torch.randn(45, 2, 64)
    value = torch.randn_like(key)
    query = torch.randn(3, 2, 16, 64)
    pos = torch.tensor([17, 95, 401])
    first, blocks = ops.compressed_attention_select(query, key, value, pos, 720, 16, 16)
    key[26:] = 1e4
    value[26:] = 1e4
    second, other = ops.compressed_attention_select(query, key, value, pos, 720, 16, 16)
    assert torch.equal(first, second) and torch.equal(blocks, other)
    for i in range(3):
        out, selected = ops.compressed_attention_select(
            query[i : i + 1], key, value, pos[i : i + 1], 720, 16, 16
        )
        torch.testing.assert_close(out[0], first[i])
        assert torch.equal(selected[0], blocks[i])


@pytest.mark.gpu
@pytest.mark.parametrize("prefix,length", [(0, 2179), (17, 131), (8215, 105)])
def test_batched_window_suffix_and_tile_boundaries(prefix, length):
    torch.manual_seed(22)
    retained = min(prefix, 127)
    start = prefix - retained
    q = torch.randn(length, 2, 16, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(retained + length, 2, 128, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(prefix, prefix + length, device="cuda")
    actual = ops.window_attention(q, kv, pos, start, 128, 32)
    # Independent unbatched SDPA oracle, including both batch and tile seams.
    key, value = kv.split(64, -1)
    expected = torch.empty_like(q)
    for begin in range(0, length, 128):
        end = min(begin + 128, length)
        first, last = max(0, retained + begin - 127), retained + end
        kp = torch.arange(first, last, device="cuda") + start
        mask = (kp <= pos[begin:end, None]) & (kp > pos[begin:end, None] - 128)
        result = F.scaled_dot_product_attention(
            q[begin:end].flatten(1, 2).transpose(0, 1)[None],
            key[first:last].transpose(0, 1)[None],
            value[first:last].transpose(0, 1)[None],
            attn_mask=mask,
            enable_gqa=True,
        )
        expected[begin:end] = result[0].transpose(0, 1).reshape_as(q[begin:end])
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
