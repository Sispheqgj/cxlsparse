"""Independent stable-sort oracle for the bounded fast hot-cache selection."""

import pytest
import torch
import triton

from cxl_recsys.models.nsa.planning.active import (
    _hot_merge_top,
    _hot_tile_top,
    _sort_active_keys,
)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "n,hot",
    [
        (17, 1),
        (256, 7),
        (513, 7),
        (513, 256),
        (1024, 1024),
        (4096, 1024),
        (8448, 256),
        (9216, 256),
        (16384, 256),
    ],
)
@pytest.mark.parametrize("distribution", ["random", "ties", "empty"])
def test_hot_priority_matches_stable_sort(n, hot, distribution):
    torch.manual_seed(n + hot)
    frequencies = torch.randint(0, 129, (2, n), device="cuda", dtype=torch.int32)
    if distribution == "ties":
        frequencies.fill_(5)
    if distribution == "empty":
        frequencies.zero_()
    k = triton.next_power_of_2(hot)
    tile = max(1024, k)
    tiles = triton.cdiv(n, tile)
    partial = torch.empty((2, tiles, k), device="cuda", dtype=torch.int64)
    chosen = torch.empty((2, hot), device="cuda", dtype=torch.int64)
    _hot_tile_top[(2, tiles)](frequencies, partial, n, k, tiles, tile, num_warps=4)
    _hot_merge_top[(2, tiles)](
        partial,
        chosen,
        tiles,
        k,
        hot,
        triton.next_power_of_2(tiles * k),
        k.bit_length(),
        num_warps=8,
    )
    expected = torch.argsort(frequencies, dim=1, descending=True, stable=True)[:, :hot]
    assert torch.equal(chosen, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("capacity", [5, 576, 4096])
def test_active_key_sort_matches_torch(capacity):
    torch.manual_seed(capacity)
    keys = torch.randint(0, 10000, (2, capacity), device="cuda", dtype=torch.int32)
    keys[:, capacity // 2 :] = 2147483647
    ordered = torch.empty_like(keys)
    _sort_active_keys[(2,)](
        keys, ordered, capacity, triton.next_power_of_2(capacity), num_warps=4
    )
    assert torch.equal(ordered, torch.sort(keys, dim=1).values)
