#include <cmath>
#include <vector>

#include "cuda_check.cuh"
#include "filters.hpp"
#include "tile.cuh"

namespace {

constexpr int kMaxRadius = 5;
constexpr int kMaxWindow = (2 * kMaxRadius + 1) * (2 * kMaxRadius + 1) * (2 * kMaxRadius + 1);

// Spatial weights exp(-|offset|^2 / 2 sigma_s^2), one per window offset. All
// threads of a warp read the same entry at the same time, which is the access
// pattern constant memory serves with a single broadcast.
__constant__ float c_spatial[kMaxWindow];

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

// Shared-memory version: the block first copies its region plus an r-voxel
// halo into shared memory (one global read per tile voxel instead of
// (2r+1)^3 per output voxel), then filters entirely from shared memory, with
// the spatial weights taken from constant memory.
__global__ void bilateral_smem_kernel(const float* __restrict__ in, float* __restrict__ out, int nx, int ny,
                                      int nz, int r, float inv_2sr) {
    extern __shared__ float tile[];
    const int tx = blockDim.x + 2 * r, ty = blockDim.y + 2 * r, tz = blockDim.z + 2 * r;
    const int bx0 = blockIdx.x * blockDim.x, by0 = blockIdx.y * blockDim.y, bz0 = blockIdx.z * blockDim.z;
    load_tile(in, tile, nx, ny, nz, bx0 - r, by0 - r, bz0 - r, tx, ty, tz);
    __syncthreads();

    const int x = bx0 + threadIdx.x, y = by0 + threadIdx.y, z = bz0 + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    // Tile index of the window's lowest corner; the center is r voxels further on each axis.
    const int base = (threadIdx.z * ty + threadIdx.y) * tx + threadIdx.x;
    const float c = tile[base + (r * ty + r) * tx + r];
    const int w = 2 * r + 1;
    float sum = 0.0f, wsum = 0.0f;
    int k = 0;
    for (int dz = 0; dz < w; ++dz)
        for (int dy = 0; dy < w; ++dy) {
            const float* row = tile + base + (dz * ty + dy) * tx;
            for (int dx = 0; dx < w; ++dx, ++k) {
                const float v = row[dx];
                const float d = v - c;
                const float wt = c_spatial[k] * __expf(-d * d * inv_2sr);
                sum += wt * v;
                wsum += wt;
            }
        }
    out[(size_t(z) * ny + y) * nx + x] = sum / wsum;
}

}  // namespace

GpuTiming bilateral_cuda(const VolumeF& in, VolumeF& out, const BilateralParams& p, const LaunchConfig& cfg,
                         GpuVariant variant) {
    if (variant != GpuVariant::Naive && variant != GpuVariant::Smem) {
        std::fprintf(stderr, "bilateral has only the naive and smem GPU variants\n");
        std::exit(1);
    }
    const int r = p.radius;
    if (variant == GpuVariant::Smem && r > kMaxRadius) {
        std::fprintf(stderr, "bilateral cuda-smem supports radius <= %d\n", kMaxRadius);
        std::exit(1);
    }
    out = VolumeF(in.nx, in.ny, in.nz);
    const size_t bytes = in.size() * sizeof(float);
    const float inv_2ss = 1.0f / (2.0f * p.sigma_spatial * p.sigma_spatial);
    const float inv_2sr = 1.0f / (2.0f * p.sigma_range * p.sigma_range);
    DeviceBuffer d_in(in.size()), d_out(in.size());
    GpuTiming t;
    EventTimer timer;

    const dim3 block(cfg.bx, cfg.by, cfg.bz);
    const dim3 grid = grid_for(in.nx, in.ny, in.nz, block);
    size_t shmem = 0;
    if (variant == GpuVariant::Smem) {
        // The weight table is setup (the CPU version builds one too), so it is not timed.
        const int w = 2 * r + 1;
        std::vector<float> spatial(size_t(w) * w * w);
        for (int dz = -r, k = 0; dz <= r; ++dz)
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx, ++k)
                    spatial[k] = std::exp(-float(dx * dx + dy * dy + dz * dz) * inv_2ss);
        CUDA_CHECK(cudaMemcpyToSymbol(c_spatial, spatial.data(), spatial.size() * sizeof(float)));
        shmem = size_t(cfg.bx + 2 * r) * (cfg.by + 2 * r) * (cfg.bz + 2 * r) * sizeof(float);
        require_shared_memory(bilateral_smem_kernel, shmem, "bilateral cuda-smem");
    }

    timer.start();
    CUDA_CHECK(cudaMemcpy(d_in.ptr, in.data.data(), bytes, cudaMemcpyHostToDevice));
    t.h2d_ms = timer.stop_ms();

    timer.start();
    if (variant == GpuVariant::Smem)
        bilateral_smem_kernel<<<grid, block, shmem>>>(d_in.ptr, d_out.ptr, in.nx, in.ny, in.nz, r, inv_2sr);
    else
        bilateral_naive_kernel<<<grid, block>>>(d_in.ptr, d_out.ptr, in.nx, in.ny, in.nz, r, inv_2ss, inv_2sr);
    CUDA_CHECK(cudaGetLastError());
    t.kernel_ms = timer.stop_ms();

    timer.start();
    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out.ptr, bytes, cudaMemcpyDeviceToHost));
    t.d2h_ms = timer.stop_ms();
    return t;
}
