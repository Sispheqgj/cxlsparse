#pragma once

// One warp owns a row. Bits identify occurrences, not deduplicated KV blocks.
// The window spans original positions [first_pending, first_pending + Window).
template <int Window>
__device__ __forceinline__ int nsa_pick_ready(
    const int* ids, const int* flags, unsigned remaining) {
  const int first = __ffs(remaining) - 1;
  if (!remaining) return -1;
  // Most blocks are ready: avoid scanning the window in that common case.
  if (nsa_ready_warp(flags + ids[first])) return first;
  const int lane = threadIdx.x & 31;
  const int position = first + lane;
  bool ready = false;
  if (lane < Window && position < 32 &&
      (remaining & (1u << position))) {
    ready = nsa_acquire(flags + ids[position]) != 0;
  }
  const unsigned candidates = __ballot_sync(0xffffffff, ready);
  if (!candidates) return -1;
  const int picked = first + __ffs(candidates) - 1;
  // Each lane that will load payload must acquire the producer's publication.
  nsa_wait_ready(flags + ids[picked]);
  return picked;
}
