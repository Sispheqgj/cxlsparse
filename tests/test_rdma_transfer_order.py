"""Repeated reads keep separate targets and contiguous RDMA spans."""

from collections import Counter

import numpy as np

from cxl_recsys.mempool.rdma import ordered_cuda_transfers


def test_query_private_repeated_blocks_remain_two_contiguous_reads():
    sources = np.tile(np.arange(80, 96, dtype=np.uint64), 2)
    destinations = np.arange(32, dtype=np.uint64)
    # The IO submission order need not already follow target allocation order.
    permutation = np.arange(31, -1, -1)
    src, dst = ordered_cuda_transfers(sources[permutation], destinations[permutation])
    assert Counter(zip(src, dst)) == Counter(zip(sources, destinations))
    boundaries = (src[1:] != src[:-1] + 1) | (dst[1:] != dst[:-1] + 1)
    assert np.count_nonzero(boundaries) + 1 == 2
    assert len(src) == 32  # Both copies must still read the remote payload.


def test_unique_sources_and_writes_keep_source_sorting():
    for sources, write in (([9, 2, 3, 8], False), ([9, 2, 2, 8], True)):
        src = np.array(sources, dtype=np.uint64)
        dst = np.array([5, 9, 8, 6], dtype=np.uint64)
        actual_src, actual_dst = ordered_cuda_transfers(src, dst, write=write)
        expected = np.argsort(src, kind="stable")
        np.testing.assert_array_equal(actual_src, src[expected])
        np.testing.assert_array_equal(actual_dst, dst[expected])
