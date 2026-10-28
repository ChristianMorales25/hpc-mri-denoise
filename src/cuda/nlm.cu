#include "cuda_check.cuh"
#include "filters.hpp"

namespace {

// Naive version: one thread per voxel, mirrors nlm_cpu exactly. Every patch
// comparison re-reads (2P+1)^3 voxels from global memory, so the same voxels
// are loaded many times by neighboring threads; that redundancy is what the
// optimized version removes.
__global__ void nlm_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int nx, int ny,
                                 int nz, int z0, int z1, int S, int P, float inv_patch_n, float inv_h2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = z0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= z1) return;

    auto at = [&](int xx, int yy, int zz) {
        return in[(size_t(clampi(zz, nz)) * ny + clampi(yy, ny)) * nx + clampi(xx, nx)];
    };

    float sum = 0.0f, wsum = 0.0f, wmax = 0.0f;
    for (int sz = -S; sz <= S; ++sz)
        for (int sy = -S; sy <= S; ++sy)
            for (int sx = -S; sx <= S; ++sx) {
                if (sx == 0 && sy == 0 && sz == 0) continue;
                float d2 = 0.0f;
                for (int pz = -P; pz <= P; ++pz)
                    for (int py = -P; py <= P; ++py)
                        for (int px = -P; px <= P; ++px) {
                            const float d = at(x + px, y + py, z + pz) - at(x + sx + px, y + sy + py, z + sz + pz);
                            d2 += d * d;
                        }
                const float wt = __expf(-(d2 * inv_patch_n) * inv_h2);
                sum += wt * at(x + sx, y + sy, z + sz);
                wsum += wt;
                wmax = fmaxf(wmax, wt);
            }
    if (wmax == 0.0f) wmax = 1.0f;
    const float c = in[(size_t(z) * ny + y) * nx + x];
    out[(size_t(z) * ny + y) * nx + x] = (sum + wmax * c) / (wsum + wmax);
}

}  // namespace

GpuTiming nlm_cuda_naive(const VolumeF& in, VolumeF& out, const NlmParams& p, const LaunchConfig& cfg) {
    out = VolumeF(in.nx, in.ny, in.nz);
    const size_t bytes = in.size() * sizeof(float);
    DeviceBuffer d_in(in.size()), d_out(in.size());
    GpuTiming t;
    EventTimer timer;

    timer.start();
    CUDA_CHECK(cudaMemcpy(d_in.ptr, in.data.data(), bytes, cudaMemcpyHostToDevice));
    t.h2d_ms = timer.stop_ms();

    const int pw = 2 * p.patch_radius + 1;
    const float inv_patch_n = 1.0f / float(pw * pw * pw);
    const float inv_h2 = 1.0f / (p.h * p.h);
    const dim3 block(cfg.bx, cfg.by, cfg.bz);
    // Process the volume in z-slabs: a single launch over a large volume can
    // exceed the ~2 s Windows/WSL display-driver watchdog (TDR) and be killed.
    const int slab = cfg.chunk_z > 0 ? cfg.chunk_z : in.nz;
    timer.start();
    for (int z0 = 0; z0 < in.nz; z0 += slab) {
        const int z1 = z0 + slab < in.nz ? z0 + slab : in.nz;
        nlm_naive_kernel<<<grid_for(in.nx, in.ny, z1 - z0, block), block>>>(
            d_in.ptr, d_out.ptr, in.nx, in.ny, in.nz, z0, z1, p.search_radius, p.patch_radius, inv_patch_n,
            inv_h2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    t.kernel_ms = timer.stop_ms();

    timer.start();
    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out.ptr, bytes, cudaMemcpyDeviceToHost));
    t.d2h_ms = timer.stop_ms();
    return t;
}
