#pragma once

#include <cub/block/block_scan.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace cxl_recsys {

// Descriptor: row, cached prefix, packed query start, query count, history length.
// Statistics: unique, unique old, hits, misses, old refs, local refs, error.
constexpr int kStats = 7;
constexpr int kThreads = 256;
constexpr int kDescriptorSize = 5;

// Four counters are scanned together so each token is classified only once.
struct PlanCounts {
    int unique, misses, old, hits;
};
struct AddCounts {
    __device__ PlanCounts operator()(PlanCounts a, PlanCounts b) const {
        return {
            a.unique + b.unique, a.misses + b.misses, a.old + b.old, a.hits + b.hits
        };
    }
};

__global__ void initialize_plan_kernel(
    int32_t *ranks, int32_t *frequencies, int32_t *lookup, int32_t *stats, int extent
) {
    const int batch = blockIdx.y;
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token < extent) {
        const int64_t index = static_cast<int64_t>(batch) * extent + token;
        ranks[index] = 0;
        frequencies[index] = 0;
        lookup[index] = -1;
    }
    if (blockIdx.x == 0 && threadIdx.x < kStats) {
        stats[batch * kStats + threadIdx.x] = 0;
    }
}

template <int Hot>
__global__ void build_hot_lookup_kernel(
    const int64_t *descriptors, const int32_t *tags, int32_t *lookup, int extent
) {
    const int batch = blockIdx.y;
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < Hot) {
        const int row = descriptors[batch * kDescriptorSize];
        const int token = tags[row * (Hot + 1) + slot];
        if (token >= 0 && token < extent) {
            lookup[static_cast<int64_t>(batch) * extent + token] = slot;
        }
    }
}

__global__ void count_references_kernel(
    const int64_t *indices,
    const int64_t *descriptors,
    int32_t *ranks,
    int32_t *frequencies,
    int32_t *stats,
    int width,
    int extent
) {
    const int batch = blockIdx.y;
    const int query = blockIdx.x;
    const int64_t *descriptor = descriptors + batch * kDescriptorSize;
    if (query >= descriptor[3]) {
        return;
    }
    const int cached = descriptor[1];
    const int query_begin = descriptor[2];
    const int history = descriptor[4];
    const int64_t base = static_cast<int64_t>(batch) * extent;
    // unique/misses carry old/local reference counts; old carries errors.
    PlanCounts local{ 0, 0, 0, 0 };
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const int64_t token =
            indices[static_cast<int64_t>(query_begin + query) * width + column];
        if (token == -1) {
            continue;
        }
        if (token < 0 || token > cached + query || token >= extent) {
            local.old = 1;
            continue;
        }
        atomicExch(ranks + base + token, 1);
        if (token < history) {
            atomicAdd(frequencies + base + token, 1);
        }
        if (token < cached) {
            ++local.unique;
        } else {
            ++local.misses;
        }
    }
    using Scan = cub::BlockScan<PlanCounts, kThreads>;
    __shared__ typename Scan::TempStorage storage;
    PlanCounts unused, total;
    Scan(storage).ExclusiveScan(
        local, unused, PlanCounts{ 0, 0, 0, 0 }, AddCounts{}, total
    );
    if (threadIdx.x == 0) {
        int32_t *summary = stats + batch * kStats;
        atomicAdd(summary + 4, total.unique);
        atomicAdd(summary + 5, total.misses);
        if (total.old || total.unique + total.misses == 0) {
            atomicExch(summary + 6, 1);
        }
    }
}

__global__ void scan_token_tiles_kernel(
    const int64_t *descriptors,
    const int32_t *lookup,
    int32_t *ranks,
    int32_t *miss_offsets,
    PlanCounts *tiles,
    int extent,
    int tile_count
) {
    const int batch = blockIdx.y;
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t base = static_cast<int64_t>(batch) * extent;
    const int cached = descriptors[batch * kDescriptorSize + 1];
    const int length = cached + descriptors[batch * kDescriptorSize + 3];
    const bool present = token < length && ranks[base + token] != 0;
    const bool old = present && token < cached;
    const bool hit = old && lookup[base + token] >= 0;
    PlanCounts local{ int(present), int(old && !hit), int(old), int(hit) };
    using Scan = cub::BlockScan<PlanCounts, kThreads>;
    __shared__ typename Scan::TempStorage storage;
    PlanCounts prefix, total;
    Scan(storage).ExclusiveScan(
        local, prefix, PlanCounts{ 0, 0, 0, 0 }, AddCounts{}, total
    );
    if (token < extent) {
        ranks[base + token] = present ? prefix.unique : -1;
        miss_offsets[base + token] = prefix.misses;
    }
    if (threadIdx.x == 0) {
        tiles[batch * tile_count + blockIdx.x] = total;
    }
}

__global__ void scan_tile_totals_kernel(
    PlanCounts *tiles, int32_t *stats, int32_t *miss_counts, int tile_count
) {
    const int batch = blockIdx.x;
    using Scan = cub::BlockScan<PlanCounts, kThreads>;
    __shared__ typename Scan::TempStorage storage;
    __shared__ PlanCounts base;
    if (threadIdx.x == 0) {
        base = { 0, 0, 0, 0 };
    }
    __syncthreads();
    for (int begin = 0; begin < tile_count; begin += kThreads) {
        const int tile = begin + threadIdx.x;
        PlanCounts local = tile < tile_count ? tiles[batch * tile_count + tile]
                                             : PlanCounts{ 0, 0, 0, 0 };
        PlanCounts prefix, total;
        Scan(storage).ExclusiveScan(
            local, prefix, PlanCounts{ 0, 0, 0, 0 }, AddCounts{}, total
        );
        if (tile < tile_count) {
            tiles[batch * tile_count + tile] = AddCounts{}(base, prefix);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            base = AddCounts{}(base, total);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int32_t *summary = stats + batch * kStats;
        summary[0] = base.unique;
        summary[1] = base.old;
        summary[2] = base.hits;
        summary[3] = base.misses;
        miss_counts[batch] = base.misses;
    }
}

template <int Hot>
__global__ void materialize_plan_kernel(
    const int64_t *descriptors,
    const int32_t *lookup,
    const int64_t *host_locations,
    const int32_t *locations,
    const uint16_t *cache,
    const uint16_t *current,
    int32_t *ranks,
    const int32_t *miss_offsets,
    const PlanCounts *tiles,
    int32_t *logical,
    int64_t *miss_sources,
    int32_t *miss_destinations,
    int32_t *stats,
    uint64_t *hbm_src_ptrs,
    int32_t *hbm_dst_local,
    int extent,
    int tile_count,
    int max_history,
    int hbm_capacity,
    int dim,
    int64_t pool_capacity
) {
    const int batch = blockIdx.y;
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t base = static_cast<int64_t>(batch) * extent;
    if (token >= extent || ranks[base + token] < 0) {
        return;
    }
    const int64_t *descriptor = descriptors + batch * kDescriptorSize;
    const int row = descriptor[0];
    const int cached = descriptor[1];
    const PlanCounts prefix = tiles[batch * tile_count + blockIdx.x];
    const int compact = ranks[base + token] + prefix.unique;
    const int miss_index = miss_offsets[base + token] + prefix.misses;
    ranks[base + token] = compact;
    logical[base + compact] = token;
    const int slot = lookup[base + token];
    if (token < cached && slot < 0) {
        const int64_t source =
            host_locations[static_cast<int64_t>(row) * max_history + token];
        if (source < 0 || source >= pool_capacity) {
            atomicExch(stats + batch * kStats + 6, 1);
        }
        miss_sources[base + miss_index] = source;
        miss_destinations[base + miss_index] = compact;
    } else {
        const uint16_t *source;
        if (token < cached) {
            const int location = locations[row * (Hot + 1) + slot];
            source = cache + static_cast<int64_t>(location) * dim;
        } else {
            source = current + (descriptor[2] + token - cached) * dim;
        }
        const int64_t hbm =
            static_cast<int64_t>(batch) * hbm_capacity + compact - miss_index;
        hbm_src_ptrs[hbm] = reinterpret_cast<uint64_t>(source);
        hbm_dst_local[hbm] = compact;
    }
}

__global__ void prepare_gather_kernel(
    const int64_t *indices,
    const int64_t *descriptors,
    const int32_t *ranks,
    const int32_t *stats,
    const int64_t *offsets,
    const uint64_t *hbm_src_ptrs,
    const int32_t *hbm_dst_local,
    uint64_t *copy_src_ptrs,
    int32_t *copy_dst_rows,
    int64_t *mapped,
    int32_t *miss_destinations,
    int width,
    int extent,
    int hbm_capacity
) {
    const int batch = blockIdx.y;
    const int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t *descriptor = descriptors + batch * kDescriptorSize;
    const int query_begin = descriptor[2];
    const int queries = descriptor[3];
    const int64_t base = static_cast<int64_t>(batch) * extent;
    const int64_t value_base = offsets[batch * 2];
    const int64_t copy_base = offsets[batch * 2 + 1];
    const int misses = stats[batch * kStats + 3];
    const int copies = stats[batch * kStats] - misses;

    if (item < static_cast<int64_t>(queries) * width) {
        const int64_t index = static_cast<int64_t>(query_begin) * width + item;
        const int64_t token = indices[index];
        mapped[index] = token < 0 ? -1 : value_base + ranks[base + token];
    }
    if (item < misses) {
        // Each destination is visited exactly once, independent of grid size.
        miss_destinations[base + item] += value_base;
    }
    if (item < copies) {
        const int64_t local = static_cast<int64_t>(batch) * hbm_capacity + item;
        copy_src_ptrs[copy_base + item] = hbm_src_ptrs[local];
        copy_dst_rows[copy_base + item] = value_base + hbm_dst_local[local];
    }
}

template <bool Vector128>
__global__ void copy_planned_hbm_kernel(
    const uint64_t *copy_src_ptrs,
    const int32_t *copy_dst_rows,
    uint16_t *values,
    int64_t num_copies,
    int dim
) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int64_t job = static_cast<int64_t>(blockIdx.x) * (blockDim.x / warp_size) +
                        threadIdx.x / warp_size;
    // Only the final block may contain spare warps. No miss or source-type
    // classification belongs here: every valid job must copy one complete KV.
    if (job >= num_copies) {
        return;
    }
    unsigned long long source_address = 0;
    int destination_row = 0;
    if (lane == 0) {
        source_address = copy_src_ptrs[job];
        destination_row = copy_dst_rows[job];
    }
    source_address = __shfl_sync(0xffffffff, source_address, 0);
    destination_row = __shfl_sync(0xffffffff, destination_row, 0);
    const auto *source = reinterpret_cast<const uint16_t *>(source_address);
    auto *destination = values + static_cast<int64_t>(destination_row) * dim;
    if constexpr (Vector128) {
        // Contiguous 128-BF16 rows are 256 bytes and preserve uint2 alignment.
        reinterpret_cast<uint2 *>(destination)[lane] =
            reinterpret_cast<const uint2 *>(source)[lane];
    } else {
        for (int column = lane; column < dim; column += warp_size) {
            destination[column] = source[column];
        }
    }
}

template <int Hot>
__global__ void update_hot_kernel(
    const int64_t *descriptors,
    const int32_t *ranks,
    const int32_t *frequencies,
    const int64_t *selected,
    const int32_t *stats,
    int32_t *tags,
    const int32_t *locations,
    const uint16_t *values,
    uint16_t *cache,
    int extent,
    int selected_count,
    int dim
) {
    const int batch = blockIdx.y;
    const int slot = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (slot >= Hot) {
        return;
    }
    const int lane = threadIdx.x % 32;
    const int row = descriptors[batch * kDescriptorSize];
    const int cached = descriptors[batch * kDescriptorSize + 1];
    const int64_t base = static_cast<int64_t>(batch) * extent;
    int token = -1, previous = -1;
    if (lane == 0) {
        previous = tags[row * (Hot + 1) + slot];
        if (slot < selected_count) {
            const int candidate = selected[batch * selected_count + slot];
            if (frequencies[base + candidate] > 0) {
                token = candidate;
            }
        }
    }
    token = __shfl_sync(0xffffffff, token, 0);
    previous = __shfl_sync(0xffffffff, previous, 0);
    if (lane == 0) {
        tags[row * (Hot + 1) + slot] = token;
    }
    // Only immutable cached history can skip a copy. Newly written history
    // must refresh even if the caller reused the same logical token number.
    if (token < 0 || (token == previous && token < cached)) {
        return;
    }
    int value_base = 0;
    for (int prior = 0; prior < batch; ++prior) {
        value_base += stats[prior * kStats];
    }
    const int location = locations[row * (Hot + 1) + slot];
    const int compact = value_base + ranks[base + token];
    // Each warp copies a complete KV from the independent values snapshot.
    for (int column = lane; column < dim; column += 32) {
        cache[static_cast<int64_t>(location) * dim + column] =
            values[static_cast<int64_t>(compact) * dim + column];
    }
}

using View = tvm::ffi::TensorView;

template <int Hot>
void plan_layer(
    View indices,
    View descriptors,
    View tags,
    View host_locations,
    View locations,
    View cache,
    View current,
    View ranks,
    View frequencies,
    View logical,
    View miss_sources,
    View miss_destinations,
    View counts,
    View stats,
    View hbm_src_ptrs,
    View hbm_dst_local,
    int64_t capacity,
    View hot_lookup,
    View miss_offsets,
    View tile_prefix,
    int64_t max_queries
) {
    const auto device = sglang::host::LaunchKernel::resolve_device(indices.device());
    const int batch = descriptors.shape()[0];
    const int extent = ranks.shape()[1];
    const int tiles = (extent + kThreads - 1) / kThreads;
    const dim3 grid(tiles, batch);
    auto *rank = static_cast<int32_t *>(ranks.data_ptr());
    auto *freq = static_cast<int32_t *>(frequencies.data_ptr());
    auto *lookup = static_cast<int32_t *>(hot_lookup.data_ptr());
    auto *miss = static_cast<int32_t *>(miss_offsets.data_ptr());
    auto *prefix = static_cast<PlanCounts *>(tile_prefix.data_ptr());
    auto *summary = static_cast<int32_t *>(stats.data_ptr());
    auto *descriptor = static_cast<const int64_t *>(descriptors.data_ptr());
    sglang::host::LaunchKernel(grid, kThreads, device)(
        initialize_plan_kernel, rank, freq, lookup, summary, extent
    );
    sglang::host::
        LaunchKernel(dim3((Hot + kThreads - 1) / kThreads, batch), kThreads, device)(
            build_hot_lookup_kernel<Hot>,
            descriptor,
            static_cast<const int32_t *>(tags.data_ptr()),
            lookup,
            extent
        );
    sglang::host::
        LaunchKernel(dim3(max_queries > 0 ? max_queries : 1, batch), kThreads, device)(
            count_references_kernel,
            static_cast<const int64_t *>(indices.data_ptr()),
            descriptor,
            rank,
            freq,
            summary,
            static_cast<int>(indices.shape()[1]),
            extent
        );
    sglang::host::LaunchKernel(grid, kThreads, device)(
        scan_token_tiles_kernel, descriptor, lookup, rank, miss, prefix, extent, tiles
    );
    sglang::host::LaunchKernel(batch, kThreads, device)(
        scan_tile_totals_kernel,
        prefix,
        summary,
        static_cast<int32_t *>(counts.data_ptr()),
        tiles
    );
    sglang::host::LaunchKernel(grid, kThreads, device)(
        materialize_plan_kernel<Hot>,
        descriptor,
        lookup,
        static_cast<const int64_t *>(host_locations.data_ptr()),
        static_cast<const int32_t *>(locations.data_ptr()),
        static_cast<const uint16_t *>(cache.data_ptr()),
        static_cast<const uint16_t *>(current.data_ptr()),
        rank,
        miss,
        prefix,
        static_cast<int32_t *>(logical.data_ptr()),
        static_cast<int64_t *>(miss_sources.data_ptr()),
        static_cast<int32_t *>(miss_destinations.data_ptr()),
        summary,
        static_cast<uint64_t *>(hbm_src_ptrs.data_ptr()),
        static_cast<int32_t *>(hbm_dst_local.data_ptr()),
        extent,
        tiles,
        static_cast<int>(host_locations.shape()[1]),
        static_cast<int>(hbm_src_ptrs.shape()[1]),
        static_cast<int>(cache.shape()[1]),
        capacity
    );
}

template <int Hot>
void gather_layer(
    View indices,
    View descriptors,
    View ranks,
    View stats,
    View offsets,
    View hbm_src_ptrs,
    View hbm_dst_local,
    View copy_src_ptrs,
    View copy_dst_rows,
    View values,
    View mapped,
    View miss_destinations,
    int64_t num_copies,
    int64_t metadata_items
) {
    const auto device = sglang::host::LaunchKernel::resolve_device(indices.device());
    if (metadata_items > 0) {
        const dim3 grid(
            (metadata_items + kThreads - 1) / kThreads, descriptors.shape()[0]
        );
        sglang::host::LaunchKernel(grid, kThreads, device)(
            prepare_gather_kernel,
            static_cast<const int64_t *>(indices.data_ptr()),
            static_cast<const int64_t *>(descriptors.data_ptr()),
            static_cast<const int32_t *>(ranks.data_ptr()),
            static_cast<const int32_t *>(stats.data_ptr()),
            static_cast<const int64_t *>(offsets.data_ptr()),
            static_cast<const uint64_t *>(hbm_src_ptrs.data_ptr()),
            static_cast<const int32_t *>(hbm_dst_local.data_ptr()),
            static_cast<uint64_t *>(copy_src_ptrs.data_ptr()),
            static_cast<int32_t *>(copy_dst_rows.data_ptr()),
            static_cast<int64_t *>(mapped.data_ptr()),
            static_cast<int32_t *>(miss_destinations.data_ptr()),
            static_cast<int>(indices.shape()[1]),
            static_cast<int>(ranks.shape()[1]),
            static_cast<int>(hbm_src_ptrs.shape()[1])
        );
    }
    // Stream order publishes the complete flat plan before any warp reads it.
    // All-miss batches have no HBM payload and must not launch this kernel.
    if (num_copies > 0) {
        constexpr int warps_per_block = kThreads / 32;
        const int64_t blocks = (num_copies + warps_per_block - 1) / warps_per_block;
        const int dim = values.shape()[1];
        const auto kernel =
            dim == 128 ? copy_planned_hbm_kernel<true> : copy_planned_hbm_kernel<false>;
        sglang::host::LaunchKernel(blocks, kThreads, device)(
            kernel,
            static_cast<const uint64_t *>(copy_src_ptrs.data_ptr()),
            static_cast<const int32_t *>(copy_dst_rows.data_ptr()),
            static_cast<uint16_t *>(values.data_ptr()),
            num_copies,
            dim
        );
    }
}

template <int Hot>
void update_hot(
    View descriptors,
    View ranks,
    View frequencies,
    View selected,
    View stats,
    View tags,
    View locations,
    View values,
    View cache
) {
    const auto device =
        sglang::host::LaunchKernel::resolve_device(descriptors.device());
    const dim3 grid(
        (Hot + kThreads / 32 - 1) / (kThreads / 32), descriptors.shape()[0]
    );
    sglang::host::LaunchKernel(grid, kThreads, device)(
        update_hot_kernel<Hot>,
        static_cast<const int64_t *>(descriptors.data_ptr()),
        static_cast<const int32_t *>(ranks.data_ptr()),
        static_cast<const int32_t *>(frequencies.data_ptr()),
        static_cast<const int64_t *>(selected.data_ptr()),
        static_cast<const int32_t *>(stats.data_ptr()),
        static_cast<int32_t *>(tags.data_ptr()),
        static_cast<const int32_t *>(locations.data_ptr()),
        static_cast<const uint16_t *>(values.data_ptr()),
        static_cast<uint16_t *>(cache.data_ptr()),
        static_cast<int>(ranks.shape()[1]),
        static_cast<int>(selected.shape()[1]),
        static_cast<int>(cache.shape()[1])
    );
}

} // namespace cxl_recsys
