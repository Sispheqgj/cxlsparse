"""Indexed NSA attention with FLA's BF16 probability/online-softmax arithmetic.

One program consumes a query and its GQA group from a compact HiSparse buffer.
Explicit indices support arbitrary cached-prefix lengths; no full history is
reloaded. Arithmetic follows the pinned FLA forward kernel (MIT), while storage
and masking use the project's indexed KV representation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention(
    Q,
    KV,
    IDS,
    OUT,
    QR: tl.constexpr,
    QG: tl.constexpr,
    KR: tl.constexpr,
    IR: tl.constexpr,
    GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BG: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    g, d, n = tl.arange(0, BG), tl.arange(0, DIM), tl.arange(0, BN)
    q = tl.load(
        Q + row * QR + g[:, None] * QG + d[None, :], g[:, None] < GROUPS, other=0
    )
    q = (q * (DIM**-0.5)).to(q.dtype)
    maximum = tl.full((BG,), -float("inf"), tl.float32)
    denominator = tl.zeros((BG,), tl.float32)
    accumulator = tl.zeros((BG, DIM), tl.float32)
    for start in range(tl.cdiv(WIDTH, BN)):
        column = start * BN + n
        ids = tl.load(IDS + row * IR + column, column < WIDTH, other=-1)
        valid = (column < WIDTH) & (ids >= 0)
        offsets = tl.maximum(ids, 0).to(tl.int64) * KR
        k = tl.load(KV + offsets[None, :] + d[:, None], valid[None, :], other=0)
        v = tl.load(KV + offsets[:, None] + DIM + d[None, :], valid[:, None], other=0)
        score = tl.where(valid[None, :], tl.dot(q, k), -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, 1))
        safe = tl.where(updated == -float("inf"), 0.0, updated)
        correction = tl.exp(maximum - safe)
        probability = tl.exp(score - safe[:, None])
        accumulator = accumulator * correction[:, None] + tl.dot(
            probability.to(q.dtype), v
        )
        denominator = denominator * correction + tl.sum(probability, 1)
        maximum = updated
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        OUT + row * GROUPS * DIM + g[:, None] * DIM + d[None, :],
        result,
        g[:, None] < GROUPS,
    )


def indexed_attention(query, records, indices, block_size=16):
    rows, groups, dim = query.shape
    if (
        dim != 64
        or groups < 16
        or groups & (groups - 1)
        or query.dtype != torch.bfloat16
        or records.dtype != query.dtype
        or records.ndim != 2
        or records.shape[1] != 2 * dim
        or indices.ndim != 2
        or indices.shape[0] != rows
        or indices.dtype not in (torch.int32, torch.int64)
        or any(
            not x.is_cuda or x.device != query.device or x.stride(-1) != 1
            for x in (query, records, indices)
        )
    ):
        raise ValueError("Unsupported NSA indexed attention geometry")
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    if indices.shape[1] == 0:
        return output.zero_()
    _attention.run(
        query,
        records,
        indices,
        output,
        query.stride(0),
        query.stride(1),
        records.stride(0),
        indices.stride(0),
        groups,
        dim,
        indices.shape[1],
        groups,
        block_size,
        grid=(rows,),
        warmup=False,
        num_warps=1,
        num_stages=3,
    )
    return output
