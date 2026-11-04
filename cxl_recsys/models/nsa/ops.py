"""Bounded-memory inference for the FLA mean-pooling NSA variant.

Complete, non-overlapping blocks feed compressed attention. GQA probabilities
are summed to select blocks, always including the current block. Each of the
three branches normalizes independently and the checkpoint's sigmoid gates
combine their outputs. Explicit absolute positions support suffix-only queries.
"""

import torch


def local_branches(
    query,
    compressed,
    window,
    positions,
    *,
    total,
    window_start,
    block_size,
    block_counts,
    window_size,
    tile,
    backend,
    return_blocks=False,
):
    """Pure GPU segment; persistent cache mutation and host planning stay outside."""
    if backend == "fla":
        from cxl_recsys.models.nsa.fla import compressed_attention_select as fused

        output, blocks = fused(
            query, compressed, positions, total, block_size, block_counts
        )
        selected = (
            blocks if return_blocks else expand_blocks(blocks, positions, block_size)
        )
    else:
        key, value = compressed.split(query.shape[-1], -1)
        output = torch.empty_like(query)
        selections = []
        for start in range(0, len(query), tile):
            stop = min(start + tile, len(query))
            output[start:stop], blocks = compressed_attention_select(
                query[start:stop],
                key,
                value,
                positions[start:stop],
                total,
                block_size,
                block_counts,
            )
            selections.append(
                blocks
                if return_blocks
                else expand_blocks(blocks, positions[start:stop], block_size)
            )
        selected = torch.cat(selections)
    local = window_attention(query, window, positions, window_start, window_size, tile)
    return output, local, selected


def append_compressed(compressed, tail, current, block_size):
    values = torch.cat((tail, current))
    count = len(values) // block_size
    complete = values[: count * block_size]
    added = complete.reshape(count, block_size, *values.shape[1:]).float().mean(1)
    return (
        torch.cat((compressed, added.to(values.dtype))),
        values[count * block_size :].clone(),
    )


def compressed_attention_select(
    query, keys, values, positions, total, block_size, count
):
    """Q=[queries, KV heads, group, dim]; summaries=[blocks, KV heads, dim]."""
    queries, heads, groups, dim = query.shape
    blocks = (total + block_size - 1) // block_size
    width = min(count, blocks)
    if len(keys):
        logits = torch.einsum("qhgd,chd->qhgc", query.float(), keys.float()) * dim**-0.5
        valid = (torch.arange(len(keys), device=query.device) + 1) * block_size
        mask = valid[None, :] <= positions[:, None] + 1
        logits.masked_fill_(~mask[:, None, None, :], -torch.inf)
        # No completed block exists for the first block's early queries.
        probability = logits.softmax(-1).nan_to_num()
        if query.is_cuda and query.dtype == torch.bfloat16:
            from cxl_recsys.models.nsa.attention import indexed_attention as fused

            ids = (
                torch.arange(len(keys), device=query.device)[None, None, :] * heads
                + torch.arange(heads, device=query.device)[None, :, None]
            )
            ids = (
                ids.expand(queries, -1, -1).clone().masked_fill_(~mask[:, None, :], -1)
            )
            output = fused(
                query.flatten(0, 1),
                torch.cat((keys, values), -1).flatten(0, 1),
                ids.flatten(0, 1),
                block_size,
            ).reshape_as(query)
        else:
            output = torch.einsum("qhgc,chd->qhgd", probability, values.float()).to(
                query.dtype
            )
        scores = probability.sum(2)
        scores = torch.nn.functional.pad(scores, (0, blocks - len(keys)))
    else:
        scores = torch.zeros((queries, heads, blocks), device=query.device)
        output = torch.zeros_like(query)
    current = positions // block_size
    causal = (
        torch.arange(blocks, device=query.device)[None, None, :]
        <= current[:, None, None]
    )
    scores.masked_fill_(~causal, -torch.inf)
    # FLA assigns the current block probability 1 per Q head. Prioritize it
    # explicitly on saturated-score ties; lower older block IDs break other ties.
    scores.scatter_(2, current[:, None, None].expand(-1, heads, 1), torch.inf)
    selected = scores.argsort(dim=-1, descending=True, stable=True)[..., :width]
    selected.masked_fill_(selected > current[:, None, None], -1)
    return output, selected


def expand_blocks(blocks, positions, block_size):
    offsets = torch.arange(block_size, device=blocks.device)
    tokens = (blocks[..., None] * block_size + offsets).flatten(-2)
    return tokens.masked_fill((tokens < 0) | (tokens > positions[:, None, None]), -1)


def indexed_attention(query, records, indices, tile):
    """Compact KV working set; Q=[rows, group, dim], indices=[rows, tokens]."""
    if query.is_cuda and query.dtype == torch.bfloat16:
        from cxl_recsys.models.nsa.attention import indexed_attention as fused

        return fused(query, records, indices)
    output = torch.empty_like(query)
    dim = query.shape[-1]
    for begin in range(0, len(query), tile):
        end = min(begin + tile, len(query))
        ids = indices[begin:end]
        key, value = records[ids.clamp_min(0)].split(dim, -1)
        scores = (
            torch.bmm(query[begin:end].float(), key.float().transpose(1, 2)) * dim**-0.5
        )
        scores.masked_fill_(ids[:, None, :] < 0, -torch.inf)
        output[begin:end] = torch.bmm(
            scores.softmax(-1).nan_to_num(), value.float()
        ).to(query.dtype)
    return output


def window_attention(query, records, positions, start, window_size, tile):
    """Only the retained window tail and this call's KV are needed on HBM."""
    if query.is_cuda and query.dtype == torch.bfloat16:
        # Batch independent 128-query windows into bounded SDPA calls. Padding
        # is masked, including the initial partial window and final query tile.
        import torch.nn.functional as F

        qtile = 128
        heads, groups, dim = query.shape[1:]
        width = qtile + window_size - 1
        output = torch.empty_like(query)
        history = len(records) - len(query)
        offsets = torch.arange(width, device=query.device)
        for begin in range(0, len(query), qtile * 16):
            end = min(begin + qtile * 16, len(query))
            tiles = (end - begin + qtile - 1) // qtile
            bases = torch.arange(tiles, device=query.device) * qtile + begin
            ids = bases[:, None] + history - window_size + 1 + offsets
            kv = records[ids.clamp(0, len(records) - 1)]
            key, value = kv.split(dim, -1)
            pos = F.pad(positions[begin:end], (0, tiles * qtile - (end - begin)))
            pos = pos.view(tiles, qtile, 1)
            absolute = ids[:, None, :] + start
            mask = (ids[:, None, :] >= 0) & (ids[:, None, :] < len(records))
            mask = mask & (absolute <= pos) & (absolute > pos - window_size)
            q = F.pad(
                query[begin:end], (0, 0, 0, 0, 0, 0, 0, tiles * qtile - (end - begin))
            )
            q = q.reshape(tiles, qtile, heads * groups, dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                q,
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=mask[:, None],
                enable_gqa=True,
            )
            output[begin:end] = attended.transpose(1, 2).reshape(
                -1, heads, groups, dim
            )[: end - begin]
        return output
    heads = query.shape[1]
    offsets = torch.arange(1 - window_size, 1, device=query.device)
    logical = positions[:, None] + offsets
    local = logical - start
    indices = (
        local[:, None, :] * heads
        + torch.arange(heads, device=query.device)[None, :, None]
    )
    indices.masked_fill_((logical < 0)[:, None, :], -1)
    return indexed_attention(
        query.flatten(0, 1), records.flatten(0, 1), indices.flatten(0, 1), tile
    ).reshape_as(query)
