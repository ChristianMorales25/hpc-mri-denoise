#include <utility>

#include "cuda_check.cuh"
#include "filters.hpp"

namespace {

__device__ __forceinline__ float conduct(float d, float inv_k2) { return __expf(-d * d * inv_k2); }

// Naive version: one launch per iteration, ping-ponging between two device
// buffers, all 7 stencil reads from global memory. The volume stays on the
// GPU across iterations, so transfer cost is paid once regardless of the
// iteration count.
__global__ void diffusion_step_kernel(const float* __restrict__ cur, float* __restrict__ next, int nx, int ny,
                                      int nz, float inv_k2, float dt) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    auto at = [&](int xx, int yy, int zz) {
        return cur[(size_t(clampi(zz, nz)) * ny + clampi(yy, ny)) * nx + clampi(xx, nx)];
    };
    const float c = at(x, y, z);
    const float dxp = at(x + 1, y, z) - c, dxm = at(x - 1, y, z) - c;
    const float dyp = at(x, y + 1, z) - c, dym = at(x, y - 1, z) - c;
    const float dzp = at(x, y, z + 1) - c, dzm = at(x, y, z - 1) - c;
    const float flux = conduct(dxp, inv_k2) * dxp + conduct(dxm, inv_k2) * dxm + conduct(dyp, inv_k2) * dyp +
                       conduct(dym, inv_k2) * dym + conduct(dzp, inv_k2) * dzp + conduct(dzm, inv_k2) * dzm;
    next[(size_t(z) * ny + y) * nx + x] = c + dt * flux;
}

}  // namespace

GpuTiming diffusion_cuda_naive(const VolumeF& in, VolumeF& out, const DiffusionParams& p,
                               const LaunchConfig& cfg) {
    out = VolumeF(in.nx, in.ny, in.nz);
    const size_t bytes = in.size() * sizeof(float);
    DeviceBuffer a(in.size()), b(in.size());
    GpuTiming t;
    EventTimer timer;

    timer.start();
    CUDA_CHECK(cudaMemcpy(a.ptr, in.data.data(), bytes, cudaMemcpyHostToDevice));
    t.h2d_ms = timer.stop_ms();

    const dim3 block(cfg.bx, cfg.by, cfg.bz);
    const dim3 grid = grid_for(in.nx, in.ny, in.nz, block);
    const float inv_k2 = 1.0f / (p.kappa * p.kappa);
    float* cur = a.ptr;
    float* next = b.ptr;
    timer.start();
    for (int it = 0; it < p.iterations; ++it) {
        diffusion_step_kernel<<<grid, block>>>(cur, next, in.nx, in.ny, in.nz, inv_k2, p.dt);
        CUDA_CHECK(cudaGetLastError());
        std::swap(cur, next);
    }
    t.kernel_ms = timer.stop_ms();

    timer.start();
    CUDA_CHECK(cudaMemcpy(out.data.data(), cur, bytes, cudaMemcpyDeviceToHost));
    t.d2h_ms = timer.stop_ms();
    return t;
}
