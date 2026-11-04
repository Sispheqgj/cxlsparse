#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace cxl_recsys {

inline void check_cuda(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result)
        );
    }
}

inline void register_mapping(int64_t address, int64_t bytes) {
    auto *host = reinterpret_cast<void *>(address);
    check_cuda(
        cudaHostRegister(host, bytes, cudaHostRegisterMapped),
        "cudaHostRegister(CXL mapping)"
    );
    try {
        void *device = nullptr;
        check_cuda(
            cudaHostGetDevicePointer(&device, host, 0),
            "cudaHostGetDevicePointer(CXL mapping)"
        );
        // The unmodified upstream TensorView launcher uses the CPU data_ptr.
        // Do not silently pass an inaccessible host VA on a different platform.
        if (device != host) {
            throw std::runtime_error("CXL mapping requires identical host/device VA");
        }
    } catch (...) {
        check_cuda(cudaHostUnregister(host), "cudaHostUnregister(failed registration)");
        throw;
    }
}

inline void unregister_mapping(int64_t address) {
    check_cuda(
        cudaHostUnregister(reinterpret_cast<void *>(address)),
        "cudaHostUnregister(CXL mapping)"
    );
}

} // namespace cxl_recsys
