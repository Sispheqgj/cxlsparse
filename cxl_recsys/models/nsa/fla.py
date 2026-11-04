"""Serving wrappers for pinned FLA compression and online top-k kernels."""

import torch


def compressed_attention_select(query, compressed, positions, total, block_size, count):
    from nsa_runtime.kernels import (
        parallel_nsa_compression_fwd_kernel,
        parallel_nsa_kernel_topk,
    )

    queries, heads, groups, dim = query.shape
    query = query.contiguous()
    key, value = (x.contiguous() for x in compressed.split(dim, -1))
    output = torch.empty_like(query)
    lse = torch.empty(
        (queries, heads * groups), device=query.device, dtype=torch.float32
    )
    indices = torch.empty(
        (queries, heads, count), device=query.device, dtype=torch.int32
    )
    shared = dict(
        q=query,
        k=key,
        scale=dim**-0.5,
        positions=positions,
        TC=len(compressed),
        T=queries,
        H=heads,
        HQ=heads * groups,
        G=groups,
        K=dim,
        BS=block_size,
        BK=dim,
        offsets=None,
        token_indices=None,
        chunk_offsets=None,
        USE_OFFSETS=False,
        num_warps=1,
        num_stages=3,
        warmup=False,
    )
    parallel_nsa_compression_fwd_kernel.run(
        **shared,
        v=value,
        o=output,
        lse=lse,
        V=dim,
        BV=dim,
        BC=block_size,
        grid=(queries, 1, heads),
    )
    parallel_nsa_kernel_topk.run(
        **shared,
        lse=lse,
        block_indices=indices,
        S=count,
        BC=max(block_size, 2 * count),
        grid=(queries, heads),
    )
    width = min(count, (total + block_size - 1) // block_size)
    return output, indices[..., :width]
