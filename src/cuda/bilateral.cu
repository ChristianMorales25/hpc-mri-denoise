#include "cuda_check.cuh"
#include "filters.hpp"

namespace {

// Naive version: one thread per voxel, every neighbor read straight from
// global memory. This is the reference point for the shared-memory version.
__global__ void bilateral_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int nx,
                                       int ny, int nz, int r, float inv_2ss, float inv_2sr) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const float c = in[(size_t(z) * ny + y) * nx + x];
    float sum = 0.0f, wsum = 0.0f;
    for (int dz = -r; dz <= r; ++dz) {
        const int zz = clampi(z + dz, nz);
        for (int dy = -r; dy <= r; ++dy) {
            const int yy = clampi(y + dy, ny);
            for (int dx = -r; dx <= r; ++dx) {
                const int xx = clampi(x + dx, nx);
                const float v = in[(size_t(zz) * ny + yy) * nx + xx];
                const float d = v - c;
                const float wt = __expf(-float(dx * dx + dy * dy + dz * dz) * inv_2ss - d * d * inv_2sr);
                sum += wt * v;
                wsum += wt;
            }
        }
    }
    out[(size_t(z) * ny + y) * nx + x] = sum / wsum;
}

}  // namespace

GpuTiming bilateral_cuda_naive(const VolumeF& in, VolumeF& out, const BilateralParams& p,
                               const LaunchConfig& cfg) {
    out = VolumeF(in.nx, in.ny, in.nz);
    const size_t bytes = in.size() * sizeof(float);
    DeviceBuffer d_in(in.size()), d_out(in.size());
    GpuTiming t;
    EventTimer timer;

    timer.start();
    CUDA_CHECK(cudaMemcpy(d_in.ptr, in.data.data(), bytes, cudaMemcpyHostToDevice));
    t.h2d_ms = timer.stop_ms();

    const dim3 block(cfg.bx, cfg.by, cfg.bz);
    timer.start();
    bilateral_naive_kernel<<<grid_for(in.nx, in.ny, in.nz, block), block>>>(
        d_in.ptr, d_out.ptr, in.nx, in.ny, in.nz, p.radius,
        1.0f / (2.0f * p.sigma_spatial * p.sigma_spatial), 1.0f / (2.0f * p.sigma_range * p.sigma_range));
    CUDA_CHECK(cudaGetLastError());
    t.kernel_ms = timer.stop_ms();

    timer.start();
    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out.ptr, bytes, cudaMemcpyDeviceToHost));
    t.d2h_ms = timer.stop_ms();
    return t;
}
