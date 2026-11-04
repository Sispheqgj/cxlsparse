#pragma once
#include <cooperative_groups.h>
#include <stdint.h>

// The owner must keep every registered source alive until the kernel completes.
// All payloads are aligned, contiguous 4096-byte NSA blocks (16 x K64/V64 BF16).
__device__ __forceinline__ unsigned nsa_shared(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void nsa_bar_init(uint64_t* p, int arrivals = 1) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(nsa_shared(p)), "r"(arrivals) : "memory");
}
__device__ __forceinline__ void nsa_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
// IO-only linear bytes; the attention CTA uses its own swizzled allocation.
// Full completion makes destination writes visible to this lane before its
// ready release. wait_group.read would allow premature publication to readers.
__device__ __forceinline__ void nsa_tma_store(void* dst, const void* src) {
  nsa_proxy_fence();
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 4096;"
               :: "l"(dst), "r"(nsa_shared(src)) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void nsa_expect(uint64_t* p) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4096;"
               :: "r"(nsa_shared(p)) : "memory");
}
__device__ __forceinline__ void nsa_tma(void* dst, int64_t src, uint64_t* bar) {
  nsa_proxy_fence();
  nsa_expect(bar);
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
               "[%0], [%1], 4096, [%2];"
               :: "r"(nsa_shared(dst)), "l"(src), "r"(nsa_shared(bar)) : "memory");
}
__device__ __forceinline__ void nsa_tensor_tma(
    void* dst, const uint8_t* maps, int physical, int external, uint64_t* bar) {
  nsa_proxy_fence();
  nsa_expect(bar);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
               "[%0], [%1, {0, %2}], [%3];"
               :: "r"(nsa_shared(dst)), "l"(maps + external * 128),
                  "r"(physical * 16), "r"(nsa_shared(bar)) : "memory");
}
__device__ __forceinline__ void nsa_cp_async_block(void* dst, int64_t src) {
  for (int v = threadIdx.x; v < 256; v += 32) {
    unsigned target = nsa_shared(static_cast<char*>(dst) + v * 16);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(target), "l"(src + v * 16) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int Stages>
__device__ __forceinline__ void nsa_cp_async_wait(int newer) {
  // All threads commit one group per block. Match the TMA ring's number of
  // outstanding blocks, draining the shorter tail without waiting on an
  // unissued group. Waiting for every block at issue would handicap cp.async.
  if (newer == 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
  if constexpr (Stages > 1) if (newer == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  if constexpr (Stages > 2) if (newer == 2) asm volatile("cp.async.wait_group 2;" ::: "memory");
  if constexpr (Stages > 3) if (newer == 3) asm volatile("cp.async.wait_group 3;" ::: "memory");
  if constexpr (Stages > 4) if (newer == 4) asm volatile("cp.async.wait_group 4;" ::: "memory");
  if constexpr (Stages > 5) if (newer == 5) asm volatile("cp.async.wait_group 5;" ::: "memory");
  if constexpr (Stages > 6) if (newer == 6) asm volatile("cp.async.wait_group 6;" ::: "memory");
  if constexpr (Stages > 7) if (newer == 7) asm volatile("cp.async.wait_group 7;" ::: "memory");
}
__device__ __forceinline__ void nsa_wait(uint64_t* bar, int phase) {
  unsigned ready;
  unsigned long long start = clock64();
  do {
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 "
                 "p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready) : "r"(nsa_shared(bar)), "r"(phase) : "memory");
    if (clock64() - start > 8000000000ULL) asm volatile("trap;");
  } while (!ready);
}
__device__ __forceinline__ int nsa_acquire(const int* p) {
  int value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
  return value;
}
__device__ __forceinline__ void nsa_release(int* p, int value) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(value) : "memory");
}
// Every lane acquires, then the collective decision is uniform even if the
// ready release races with polling. This kernel uses exactly one full warp/CTA.
__device__ __forceinline__ int nsa_ready_warp(const int* p) {
  return __all_sync(0xffffffff, nsa_acquire(p) != 0);
}
__device__ __forceinline__ void nsa_wait_ready(const int* p) {
  unsigned long long start = clock64();
  while (!nsa_acquire(p)) {
    if (clock64() - start > 8000000000ULL) asm volatile("trap;");
  }
}
__device__ __forceinline__ void nsa_cluster_sync() {
  cooperative_groups::this_cluster().sync();
}
__device__ __forceinline__ void nsa_cluster_barrier_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void nsa_arrive_remote(uint64_t* bar, int rank = 0) {
  unsigned remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote) : "r"(nsa_shared(bar)), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
               :: "r"(remote) : "memory");
}
__device__ __forceinline__ void nsa_request(int* p, int value, uint64_t* ready) {
  unsigned remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
               : "=r"(remote) : "r"(nsa_shared(p)));
  asm volatile("st.shared::cluster.u32 [%0], %1;"
               :: "r"(remote), "r"(value) : "memory");
  nsa_arrive_remote(ready);
}
__device__ __forceinline__ bool nsa_request_ready(uint64_t* bar, int phase) {
  unsigned ready;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 "
               "p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ready) : "r"(nsa_shared(bar)), "r"(phase) : "memory");
  return ready != 0;
}
__device__ __forceinline__ void nsa_wait_cluster(uint64_t* bar, int phase) {
  unsigned long long start = clock64();
  while (!nsa_request_ready(bar, phase)) {
    if (clock64() - start > 8000000000ULL) asm volatile("trap;");
  }
}

// Explicit per-slot request barriers carry the consumer -> producer ordering,
// including retirement of all previous reads from that destination. A separate
// transaction barrier carries producer -> consumer data readiness. Request
// phases advance only on service, so ready requests remain visible until used.
template<int Consumers, int Stages, bool Diagnostic, bool Tensor>
__device__ __forceinline__ void nsa_cluster_io(
    void* tiles, uint64_t* bars, const int64_t* addresses, const int* external,
    const uint8_t* maps, int* requests, uint64_t* request_bars, int active, int64_t* counters) {
  auto* bytes = static_cast<char*>(tiles);
  constexpr unsigned full_warp = 0xffffffffu;
  constexpr int entries = Consumers * Stages;
  static_assert(entries <= 32);
  int lane = threadIdx.x;
  int cursor = 0;
  int request_phase = 0;
  unsigned long long start = clock64();
  while (true) {
    bool finished = lane < active && nsa_request_ready(request_bars + entries + lane, 0);
    if (__popc(__ballot_sync(full_warp, finished)) == active) break;
    bool available = lane < entries && nsa_request_ready(request_bars + lane, request_phase);
    int request = available ? requests[lane] : 0;
    unsigned ready = __ballot_sync(full_warp, request != 0);
    if (ready) {
      // Rotate priority so a fast consumer cannot starve a slower neighbor.
      unsigned rotated = __funnelshift_r(ready, ready, cursor);
      int chosen = (cursor + __ffs(rotated) - 1) & 31;
      int selected = __shfl_sync(full_warp, request, chosen);
      int slot = chosen % Stages;
      bool match = lane < entries && request == selected && lane % Stages == slot;
      unsigned targets = match ? (1u << (lane / Stages + 1)) : 0;
      uint16_t mask = static_cast<uint16_t>(__reduce_or_sync(full_warp, targets));
      if (match) request_phase ^= 1;
      __syncwarp(full_warp);
      if (lane == 0) {
        int physical = selected - 1;
        if constexpr (Tensor) {
          asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes."
                       "multicast::cluster [%0], [%1, {0, %2}], [%3], %4;"
                       :: "r"(nsa_shared(bytes + slot * 4096)),
                          "l"(maps + external[physical] * 128), "r"(physical * 16),
                          "r"(nsa_shared(bars + slot)), "h"(mask) : "memory");
        } else {
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
                       "multicast::cluster [%0], [%1], 4096, [%2], %3;"
                       :: "r"(nsa_shared(bytes + slot * 4096)), "l"(addresses[physical]),
                          "r"(nsa_shared(bars + slot)), "h"(mask) : "memory");
        }
        if constexpr (Diagnostic) {
          counters[0] += 1;
          counters[1] += external[physical];
          counters[2] += __popc(mask);
        }
      }
      // The data phase requires both consumer arming and producer acceptance,
      // as well as complete_tx. This explicit producer arrival orders the
      // issuing warp before consumers, including in racecheck's raw-bulk model.
      __syncwarp(full_warp);
      if (match) nsa_arrive_remote(bars + slot, lane / Stages + 1);
      cursor = (chosen + 1) & 31;
      start = clock64();
    } else if (clock64() - start > 8000000000ULL) asm volatile("trap;");
  }
}
__device__ __forceinline__ float nsa_tma_exp(float x) {
  float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float nsa_tma_div(float a, float b) {
  float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
