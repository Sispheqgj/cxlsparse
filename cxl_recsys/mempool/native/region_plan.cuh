#pragma once

#include <cub/block/block_scan.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <stdexcept>

namespace cxl_recsys {

// Only this metadata kernel classifies regions. The upstream copy kernels
// receive dense local-slot plans and never walk another region's KV jobs.
__global__ void route_regions_kernel(
    const int64_t *sources,
    const int32_t *destinations,
    const int32_t *counts,
    int64_t *region_sources,
    int32_t *region_destinations,
    int32_t *region_counts,
    int batch_size,
    int stride,
    int region_count,
    int page_size
) {
    constexpr int threads = 256;
    using Scan = cub::BlockScan<int, threads>;
    __shared__ typename Scan::TempStorage temporary;
    __shared__ int written;
    const int row = blockIdx.x;
    const int region = blockIdx.y;
    const int lane = threadIdx.x;
    const int64_t input_base = static_cast<int64_t>(row) * stride;
    const int64_t output_base =
        (static_cast<int64_t>(region) * batch_size + row) * stride;
    const int count = counts[row];
    if (lane == 0) {
        written = 0;
    }
    __syncthreads();
    // Stable compaction preserves source-plan order without per-token atomics.
    // The metadata is read at most N times (N is normally 1..4), never the KV.
    for (int begin = 0; begin < count; begin += threads) {
        const int item = begin + lane;
        const int64_t source = item < count ? sources[input_base + item] : 0;
        const int64_t page = source / page_size;
        const bool selected = item < count && page % region_count == region;
        int offset = 0;
        int total = 0;
        Scan(temporary).ExclusiveSum(static_cast<int>(selected), offset, total);
        if (selected) {
            const int64_t output = output_base + written + offset;
            region_sources[output] =
                (page / region_count) * page_size + source % page_size;
            region_destinations[output] = destinations[input_base + item];
        }
        __syncthreads();
        if (lane == 0) {
            written += total;
        }
        __syncthreads();
    }
    if (lane == 0) {
        region_counts[region * batch_size + row] = written;
    }
}

__global__ void offset_region_destinations_kernel(
    int32_t *destinations,
    const int32_t *counts,
    const int64_t *offsets,
    int batch_size,
    int stride,
    int offset_stride
) {
    const int row = blockIdx.y;
    const int region = blockIdx.z;
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < counts[region * batch_size + row]) {
        const int64_t index =
            (static_cast<int64_t>(region) * batch_size + row) * stride + item;
        destinations[index] += offsets[row * offset_stride];
    }
}

using RegionView = tvm::ffi::TensorView;

inline void route_regions(
    RegionView sources,
    RegionView destinations,
    RegionView counts,
    RegionView region_sources,
    RegionView region_destinations,
    RegionView region_counts,
    int64_t page_size
) {
    const auto batch = sources.shape()[0];
    const auto stride = sources.shape()[1];
    const auto regions = region_sources.shape()[0];
    if (page_size <= 0 || regions <= 0 || destinations.shape()[0] != batch ||
        destinations.shape()[1] != stride || counts.shape()[0] != batch ||
        region_sources.shape()[1] != batch || region_sources.shape()[2] != stride ||
        region_destinations.shape()[0] != regions ||
        region_destinations.shape()[1] != batch ||
        region_destinations.shape()[2] != stride ||
        region_counts.shape()[0] != regions || region_counts.shape()[1] != batch) {
        throw std::runtime_error("Invalid CXL region plan dimensions");
    }
    const auto device = sglang::host::LaunchKernel::resolve_device(sources.device());
    sglang::host::LaunchKernel(dim3(batch, regions), 256, device)(
        route_regions_kernel,
        static_cast<const int64_t *>(sources.data_ptr()),
        static_cast<const int32_t *>(destinations.data_ptr()),
        static_cast<const int32_t *>(counts.data_ptr()),
        static_cast<int64_t *>(region_sources.data_ptr()),
        static_cast<int32_t *>(region_destinations.data_ptr()),
        static_cast<int32_t *>(region_counts.data_ptr()),
        batch,
        stride,
        regions,
        page_size
    );
}

inline void offset_region_destinations(
    RegionView destinations, RegionView counts, RegionView offsets
) {
    const auto regions = destinations.shape()[0];
    const auto batch = destinations.shape()[1];
    const auto stride = destinations.shape()[2];
    const auto device =
        sglang::host::LaunchKernel::resolve_device(destinations.device());
    sglang::host::LaunchKernel(dim3((stride + 255) / 256, batch, regions), 256, device)(
        offset_region_destinations_kernel,
        static_cast<int32_t *>(destinations.data_ptr()),
        static_cast<const int32_t *>(counts.data_ptr()),
        static_cast<const int64_t *>(offsets.data_ptr()),
        batch,
        stride,
        offsets.strides()[0]
    );
}

} // namespace cxl_recsys
