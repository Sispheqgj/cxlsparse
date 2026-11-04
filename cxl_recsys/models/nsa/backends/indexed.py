"""TileLang selected arithmetic on the serving HiSparse GPU token map.

Derived from the handed-off N2b cp.async / mma.sync attention. IO stays in the
existing backend; no CPU block planner/packing or cooperative grid is needed.
Per-row token IDs already encode the model's causal selection, including
duplicate occurrences. FP32 online softmax spans every original BS tile.
"""

# pyright: reportInvalidTypeForm=false, reportAttributeAccessIssue=false

from functools import lru_cache
from typing import Any, cast

import torch

from cxl_recsys.models.nsa.backends.tilelang import pass_configs


@lru_cache(maxsize=64)
def indexed_kernel(rows, groups, capacity, width, block_size=16):
    import tilelang
    from tilelang import language as T

    if min(rows, capacity, width) < 1 or width % block_size:
        raise ValueError("Positive block-aligned indexed NSA geometry required")
    if groups not in (16, 32) or block_size not in (16, 32, 64):
        raise ValueError("Indexed TileLang NSA supports G16/32, BS16/32/64")

    @T.macro
    def prefetch(row, slot, KV, IDs, k, v):
        for token, vector in T.Parallel(block_size, 8):
            physical = T.min(
                T.max(IDs[row, slot * block_size + token], 0), capacity - 1
            )
            valid = IDs[row, slot * block_size + token] >= 0
            T.call_extern(
                "handle",
                "tl::cp_async_gs_conditional<16>",
                T.address_of(k[slot % 2, token, vector * 8]),
                T.address_of(KV[physical, vector * 8]),
                valid,
            )
            T.call_extern(
                "handle",
                "tl::cp_async_gs_conditional<16>",
                T.address_of(v[slot % 2, token, vector * 8]),
                T.address_of(KV[physical, 64 + vector * 8]),
                valid,
            )
        T.ptx_commit_group()

    @T.prim_func
    def main(
        Q: T.Tensor((rows, groups, 64), "bfloat16"),
        KV: T.Tensor((capacity, 128), "bfloat16"),
        IDs: T.Tensor((rows, width), "int64"),
        Out: T.Tensor((rows, groups, 64), "bfloat16"),
    ):
        with T.Kernel(rows, threads=32) as row:
            q = T.alloc_shared((groups, 64), "bfloat16")
            k = T.alloc_shared((2, block_size, 64), "bfloat16")
            v = T.alloc_shared((2, block_size, 64), "bfloat16")
            probability = T.alloc_shared((groups, block_size), "bfloat16")
            score = T.alloc_fragment((groups, block_size), "float32")
            acc = T.alloc_fragment((groups, 64), "float32")
            maximum = T.alloc_fragment((groups,), "float32")
            previous = T.alloc_fragment((groups,), "float32")
            denominator = T.alloc_fragment((groups,), "float32")
            factor = T.alloc_fragment((groups,), "float32")
            block_sum = T.alloc_fragment((groups,), "float32")
            for h, d in T.Parallel(groups, 64):
                q[h, d] = Q[row, h, d].astype("float32") * 0.125
            T.fill(maximum, -T.infinity("float32"))
            T.clear(denominator)
            T.clear(acc)
            prefetch(row, 0, KV, IDs, k, v)
            T.ptx_wait_group(0)
            T.sync_threads()
            for slot in T.serial(width // block_size):
                if slot + 1 < width // block_size:
                    prefetch(row, slot + 1, KV, IDs, k, v)
                # Explicit zeroing prevents masked NaN values entering PV.
                for token, d in T.Parallel(block_size, 64):
                    if IDs[row, slot * block_size + token] < 0:
                        k[slot % 2, token, d] = 0
                        v[slot % 2, token, d] = 0
                T.sync_threads()
                T.clear(score)
                T.gemm(q, k[slot % 2, :, :], score, transpose_B=True)
                for h, token in T.Parallel(groups, block_size):
                    score[h, token] = T.if_then_else(
                        IDs[row, slot * block_size + token] >= 0,
                        score[h, token],
                        -T.infinity("float32"),
                    )
                T.copy(maximum, previous)
                T.reduce_max(score, maximum, dim=1, clear=False)
                for h in T.Parallel(groups):
                    factor[h] = T.if_then_else(
                        previous[h] == -T.infinity("float32"),
                        0.0,
                        T.exp2((previous[h] - maximum[h]) * 1.4426950408889634),
                    )
                for h, token in T.Parallel(groups, block_size):
                    score[h, token] = T.if_then_else(
                        score[h, token] == -T.infinity("float32"),
                        0.0,
                        T.exp2((score[h, token] - maximum[h]) * 1.4426950408889634),
                    )
                T.reduce_sum(score, block_sum, dim=1)
                for h in T.Parallel(groups):
                    denominator[h] = denominator[h] * factor[h] + block_sum[h]
                for h, d in T.Parallel(groups, 64):
                    acc[h, d] *= factor[h]
                T.copy(score, probability)
                T.sync_threads()
                T.gemm(probability, v[slot % 2, :, :], acc)
                T.ptx_wait_group(0)
                T.sync_threads()
            for h, d in T.Parallel(groups, 64):
                Out[row, h, d] = acc[h, d] / T.if_then_else(
                    denominator[h] > 0, denominator[h], 1.0
                )

    config = pass_configs()
    config[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True
    return tilelang.compile(
        cast(Any, main).with_attr("global_symbol", "nsa_tilelang_indexed"),
        target={"kind": "cuda", "arch": "sm_90a"},
        pass_configs=config,
    )


def indexed_attention(query, records, indices, block_size=16):
    if (
        query.dtype != torch.bfloat16
        or query.ndim != 3
        or query.shape[-1] != 64
        or records.dtype != query.dtype
        or records.ndim != 2
        or records.shape[1] != 128
        or indices.dtype != torch.int64
        or indices.shape[0] != len(query)
        or any(
            not x.is_cuda or not x.is_contiguous() for x in (query, records, indices)
        )
        or any(x.device != query.device for x in (records, indices))
    ):
        raise ValueError("Expected contiguous BF16 GPU Q/KV and int64 token IDs")
    output = torch.empty_like(query)
    if indices.shape[1] == 0:
        return output.zero_()
    indexed_kernel(
        len(query), query.shape[1], len(records), indices.shape[1], block_size
    )(query, records, indices, output)
    return output
