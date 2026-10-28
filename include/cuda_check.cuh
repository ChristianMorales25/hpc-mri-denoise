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
