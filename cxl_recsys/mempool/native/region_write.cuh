#pragma once

#include <kvcacheio/hicache.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cxl_recsys {

// Reuse HiCache's vectorized IO, but route each token to its own registered
// region. Region selection is metadata work, not a predicate around each dim.
template <int64_t TokenBytes, uint32_t Unroll, int Regions>
__global__ __launch_bounds__(1024, 1) void write_regions_kernel(
    const char *source,
    const int64_t *global_slots,
    const int64_t *region_bases,
    int64_t length,
    int64_t page_size,
    int64_t runtime_regions
) {
    constexpr uint32_t threads_per_token = 32 / Unroll;
    constexpr uint32_t tokens_per_block = 1024 / threads_per_token;
    const int64_t region_count = Regions ? Regions : runtime_regions;
    const int64_t first =
        blockIdx.x * tokens_per_block + threadIdx.x / threads_per_token;
    const int64_t stride = gridDim.x * tokens_per_block;

    for (int64_t row = first; row < length; row += stride) {
        // One lane calculates the address; all lanes of this token group remain
        // active, including the final partial warp. No source index is needed:
        // source[row] corresponds to global_slots[row] in the packed layer KV.
        unsigned long long address = 0;
        const unsigned mask = __activemask();
        if (threadIdx.x % threads_per_token == 0) {
            const int64_t slot = global_slots[row];
            const int64_t page = slot / page_size;
            const int64_t region = page % region_count;
            const int64_t local_slot =
                (page / region_count) * page_size + slot % page_size;
            address = region_bases[region] + local_slot * TokenBytes;
        }
        address = __shfl_sync(mask, address, 0, threads_per_token);
        const auto value = sglang::device::load_vec<TokenBytes, threads_per_token>(
            source + row * TokenBytes
        );
        sglang::device::store_vec<TokenBytes, threads_per_token>(
            reinterpret_cast<void *>(address), value
        );
    }
}

template <int64_t TokenBytes, uint32_t Unroll>
void write_regions(
    tvm::ffi::TensorView source,
    tvm::ffi::TensorView global_slots,
    tvm::ffi::TensorView region_bases,
    int64_t page_size
) {
    using namespace sglang::host;
    auto length = SymbolicSize{ "token count" };
    auto regions = SymbolicSize{ "region count" };
    auto device = SymbolicDevice{};
    TensorMatcher({ length, TokenBytes / 2 })
        .with_dtype<sglang::bf16_t>()
        .with_device<sglang::kDLGPU>(device)
        .verify(source);
    TensorMatcher({ length })
        .with_dtype<int64_t>()
        .with_device<sglang::kDLGPU>(device)
        .verify(global_slots);
    TensorMatcher({ regions })
        .with_dtype<int64_t>()
        .with_device<sglang::kDLGPU>(device)
        .verify(region_bases);
    RuntimeCheck(page_size > 0 && regions.unwrap() > 0, "Invalid CXL geometry");
    RuntimeCheck(
        length.unwrap() <= std::numeric_limits<uint32_t>::max(),
        "CXL writes require fewer than 2**32 tokens"
    );
    if (length.unwrap() == 0) {
        return;
    }

    // Preserve the previous aggregate IO concurrency budget (two CTAs per
    // region), now inside one grid/stream. Threads are not pinned to a region.
    constexpr int64_t tokens_per_block = 1024 / (32 / Unroll);
    const int64_t blocks = std::min(
        (length.unwrap() + tokens_per_block - 1) / tokens_per_block,
        2 * regions.unwrap()
    );
    // All common region counts are compiled offline into the same library.
    // Constant divisors also make the three-region layout cheap to address.
    const auto kernel =
        regions.unwrap() == 1   ? write_regions_kernel<TokenBytes, Unroll, 1>
        : regions.unwrap() == 2 ? write_regions_kernel<TokenBytes, Unroll, 2>
        : regions.unwrap() == 3 ? write_regions_kernel<TokenBytes, Unroll, 3>
        : regions.unwrap() == 4 ? write_regions_kernel<TokenBytes, Unroll, 4>
                                : write_regions_kernel<TokenBytes, Unroll, 0>;
    LaunchKernel(blocks, 1024, device.unwrap())(
        kernel,
        static_cast<const char *>(source.data_ptr()),
        static_cast<const int64_t *>(global_slots.data_ptr()),
        static_cast<const int64_t *>(region_bases.data_ptr()),
        length.unwrap(),
        page_size,
        regions.unwrap()
    );
}

} // namespace cxl_recsys
