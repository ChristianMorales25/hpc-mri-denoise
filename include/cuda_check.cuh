#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), \
                         __FILE__, __LINE__, cudaGetErrorString(err_));               \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

__device__ __forceinline__ int clampi(int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); }

inline dim3 grid_for(int nx, int ny, int nz, const dim3& block) {
    return dim3((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (nz + block.z - 1) / block.z);
}

// Owns a device allocation of n floats.
struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float))); }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Wraps a pair of CUDA events to time a region on the default stream.
struct EventTimer {
    cudaEvent_t start_, stop_;
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};
