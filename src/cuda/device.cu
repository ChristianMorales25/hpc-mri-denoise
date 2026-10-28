#include "cuda_check.cuh"
#include "filters.hpp"

void print_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::printf("GPU: %s (sm_%d%d), %d SMs, %.1f GB, %zu KB smem/block, runtime %d, driver %d\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / 1073741824.0, prop.sharedMemPerBlock / 1024, rt, drv);
}
