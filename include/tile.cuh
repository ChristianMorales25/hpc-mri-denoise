#pragma once

#include "cuda_check.cuh"

// Cooperatively copies a tx*ty*tz region of the volume whose lowest corner is
// (ox, oy, oz) into shared memory, x fastest. Coordinates are clamped on load,
// which gives the same replicate-border behavior as the CPU versions. Every
// thread of the block must call it; the caller adds the __syncthreads().
__device__ __forceinline__ void load_tile(const float* __restrict__ in, float* tile, int nx, int ny, int nz,
                                          int ox, int oy, int oz, int tx, int ty, int tz) {
    const int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;
    const int n = tx * ty * tz;
    for (int i = tid; i < n; i += nthreads) {
        const int lx = i % tx, ly = (i / tx) % ty, lz = i / (tx * ty);
        tile[i] = in[(size_t(clampi(oz + lz, nz)) * ny + clampi(oy + ly, ny)) * nx + clampi(ox + lx, nx)];
    }
}

// Opts a kernel in to more than the default 48 KB of dynamic shared memory
// when needed, and fails with a clear message if the device cannot provide it.
template <typename Kernel>
inline void require_shared_memory(Kernel kernel, size_t bytes, const char* what) {
    int dev = 0, optin = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (bytes > size_t(optin)) {
        std::fprintf(stderr, "%s needs %zu bytes of shared memory per block, device allows %d; use a smaller --block\n",
                     what, bytes, optin);
        std::exit(1);
    }
    if (bytes > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
}
