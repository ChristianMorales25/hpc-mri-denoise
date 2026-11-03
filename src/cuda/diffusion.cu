#include <algorithm>
#include <utility>

#include "cuda_check.cuh"
#include "filters.hpp"
#include "tile.cuh"

namespace {

__device__ __forceinline__ float conduct(float d, float inv_k2) { return __expf(-d * d * inv_k2); }

// One explicit Perona-Malik update from the center value and its 6 neighbors,
// in the same order as diffusion_cpu so results agree to rounding.
__device__ __forceinline__ float pm_update(float c, float xp, float xm, float yp, float ym, float zp, float zm,
                                           float inv_k2, float dt) {
    const float dxp = xp - c, dxm = xm - c, dyp = yp - c, dym = ym - c, dzp = zp - c, dzm = zm - c;
    const float flux = conduct(dxp, inv_k2) * dxp + conduct(dxm, inv_k2) * dxm + conduct(dyp, inv_k2) * dyp +
                       conduct(dym, inv_k2) * dym + conduct(dzp, inv_k2) * dzp + conduct(dzm, inv_k2) * dzm;
    return c + dt * flux;
}

// ---------------------------------------------------------------------------
// Naive: one launch per iteration, ping-ponging between two device buffers,
// all 7 stencil reads from global memory. The volume stays on the GPU across
// iterations, so transfer cost is paid once regardless of the iteration count.
// ---------------------------------------------------------------------------
__global__ void diffusion_naive_kernel(const float* __restrict__ cur, float* __restrict__ next, int nx, int ny,
                                       int nz, float inv_k2, float dt) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    auto at = [&](int xx, int yy, int zz) {
        return cur[(size_t(clampi(zz, nz)) * ny + clampi(yy, ny)) * nx + clampi(xx, nx)];
    };
    next[(size_t(z) * ny + y) * nx + x] = pm_update(at(x, y, z), at(x + 1, y, z), at(x - 1, y, z), at(x, y + 1, z),
                                                    at(x, y - 1, z), at(x, y, z + 1), at(x, y, z - 1), inv_k2, dt);
}

// ---------------------------------------------------------------------------
// Smem: still one launch per iteration, but each block loads its region plus
// a 1-voxel halo once, so each input voxel is read from global memory about
// once instead of up to 7 times (minus whatever the L1/L2 caches already
// absorbed for the naive kernel).
// ---------------------------------------------------------------------------
__global__ void diffusion_smem_kernel(const float* __restrict__ cur, float* __restrict__ next, int nx, int ny,
                                      int nz, float inv_k2, float dt) {
    extern __shared__ float tile[];
    const int tx = blockDim.x + 2, ty = blockDim.y + 2, tz = blockDim.z + 2;
    const int bx0 = blockIdx.x * blockDim.x, by0 = blockIdx.y * blockDim.y, bz0 = blockIdx.z * blockDim.z;
    load_tile(cur, tile, nx, ny, nz, bx0 - 1, by0 - 1, bz0 - 1, tx, ty, tz);
    __syncthreads();

    const int x = bx0 + threadIdx.x, y = by0 + threadIdx.y, z = bz0 + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    // Clamped loads already replicate the border into the halo, so no clamping here.
    const int i = ((threadIdx.z + 1) * ty + threadIdx.y + 1) * tx + threadIdx.x + 1;
    const int sy = tx, sz = tx * ty;
    next[(size_t(z) * ny + y) * nx + x] = pm_update(tile[i], tile[i + 1], tile[i - 1], tile[i + sy], tile[i - sy],
                                                    tile[i + sz], tile[i - sz], inv_k2, dt);
}

// ---------------------------------------------------------------------------
// Fused: 2.5D blocking with temporal blocking, K iterations per launch.
//
// A block of PX x PY threads owns one x-y column each (the block's output
// tile plus a K-voxel x-y halo) and marches through a chunk of z planes. Each
// iteration level s = 0..K-1 keeps only its last 3 planes in shared memory
// (a ring buffer). When input plane t arrives, level s can produce plane
// t - s, so the K updates form a pipeline along z and every input voxel is
// read from global memory once per K iterations. There is no z halo except
// K planes at each chunk boundary, which is what made the 3D-tile version
// slower than the naive kernel.
//
// Validity: after s updates only columns at least s voxels inside the block
// are correct, so level s is computed on a region that shrinks by one column
// per side, and level K is exactly the output tile.
//
// Borders: neighbor coordinates are clamped to the volume before they are
// mapped to a column or plane, so a voxel on the volume face uses itself as
// its outside neighbor at every level, like the replicate border of the CPU
// version.
// ---------------------------------------------------------------------------
__global__ void diffusion_fused_kernel(const float* __restrict__ cur, float* __restrict__ next, int nx, int ny,
                                       int nz, int K, int zchunk, float inv_k2, float dt) {
    extern __shared__ float ring[];  // [level 0..K-1][3 planes][PY][PX]
    const int PX = blockDim.x, PY = blockDim.y, plane = PX * PY;
    const int ox = blockIdx.x * (PX - 2 * K) - K, oy = blockIdx.y * (PY - 2 * K) - K;
    const int lx = threadIdx.x, ly = threadIdx.y;
    const int gx = ox + lx, gy = oy + ly;
    const bool in_volume = gx >= 0 && gy >= 0 && gx < nx && gy < ny;

    // Shared-memory offsets of this column and its 4 in-plane neighbors. They
    // only matter for columns that compute, which are never on the tile edge.
    const int c = ly * PX + lx;
    const int xp = ly * PX + (clampi(gx + 1, nx) - ox), xm = ly * PX + (clampi(gx - 1, nx) - ox);
    const int yp = (clampi(gy + 1, ny) - oy) * PX + lx, ym = (clampi(gy - 1, ny) - oy) * PX + lx;
    auto level = [&](int s, int z) { return ring + (s * 3 + z % 3) * plane; };  // z >= 0

    const int z0 = blockIdx.z * zchunk, z1 = min(nz, z0 + zchunk);
    const int load_lo = max(0, z0 - K), load_hi = min(nz, z1 + K);
    const float* column = cur + size_t(clampi(gy, ny)) * nx + clampi(gx, nx);
    const size_t zstride = size_t(nx) * ny;
    // Software pipelining: plane t + 1 is requested from global memory before
    // the K levels of plane t are computed, so its latency overlaps that work
    // instead of stalling the whole block at the next barrier.
    float incoming = column[load_lo * zstride];
    for (int t = load_lo; t < z1 + K; ++t) {
        if (t < load_hi) level(0, t)[c] = incoming;
        if (t + 1 < load_hi) incoming = column[(t + 1) * zstride];
        __syncthreads();
        for (int s = 1; s <= K; ++s) {
            const int zs = t - s;
            // Level s is needed on planes [z0 - (K - s), z1 + (K - s)) within the volume.
            const bool active = zs >= max(0, z0 - (K - s)) && zs < min(nz, z1 + (K - s)) && in_volume &&
                                lx >= s && lx < PX - s && ly >= s && ly < PY - s;
            if (active) {
                const float* P = level(s - 1, zs);
                const float* Pm = level(s - 1, max(zs - 1, 0));
                const float* Pp = level(s - 1, min(zs + 1, nz - 1));
                const float v = pm_update(P[c], P[xp], P[xm], P[yp], P[ym], Pp[c], Pm[c], inv_k2, dt);
                if (s == K) next[(size_t(zs) * ny + gy) * nx + gx] = v;
                else level(s, zs)[c] = v;
            }
            __syncthreads();
        }
    }
}

}  // namespace

GpuTiming diffusion_cuda(const VolumeF& in, VolumeF& out, const DiffusionParams& p, const LaunchConfig& cfg,
                         GpuVariant variant) {
    if (variant == GpuVariant::Sep) {
        std::fprintf(stderr, "diffusion has no sep variant\n");
        std::exit(1);
    }
    const int K = std::max(1, cfg.fuse);
    const int zchunk = cfg.zchunk > 0 ? cfg.zchunk : in.nz;
    if (variant == GpuVariant::Fused && (cfg.bx <= 2 * K || cfg.by <= 2 * K)) {
        std::fprintf(stderr, "cuda-fused: block X and Y must exceed 2*fuse (the x-y halo); got %d,%d with fuse %d\n",
                     cfg.bx, cfg.by, K);
        std::exit(1);
    }
    out = VolumeF(in.nx, in.ny, in.nz);
    const size_t bytes = in.size() * sizeof(float);
    DeviceBuffer a(in.size()), b(in.size());
    GpuTiming t;
    EventTimer timer;

    const dim3 block(cfg.bx, cfg.by, cfg.bz);
    const dim3 grid = grid_for(in.nx, in.ny, in.nz, block);
    const dim3 block2d(cfg.bx, cfg.by, 1);  // fused: one thread per x-y column
    const float inv_k2 = 1.0f / (p.kappa * p.kappa);
    const size_t smem1 = size_t(cfg.bx + 2) * (cfg.by + 2) * (cfg.bz + 2) * sizeof(float);
    auto smem_fused = [&](int k) { return size_t(k) * 3 * cfg.bx * cfg.by * sizeof(float); };
    if (variant == GpuVariant::Smem) require_shared_memory(diffusion_smem_kernel, smem1, "diffusion cuda-smem");
    if (variant == GpuVariant::Fused) require_shared_memory(diffusion_fused_kernel, smem_fused(K), "diffusion cuda-fused");

    timer.start();
    CUDA_CHECK(cudaMemcpy(a.ptr, in.data.data(), bytes, cudaMemcpyHostToDevice));
    t.h2d_ms = timer.stop_ms();

    float* cur = a.ptr;
    float* next = b.ptr;
    timer.start();
    for (int done = 0; done < p.iterations;) {
        int steps = 1;
        if (variant == GpuVariant::Naive)
            diffusion_naive_kernel<<<grid, block>>>(cur, next, in.nx, in.ny, in.nz, inv_k2, p.dt);
        else if (variant == GpuVariant::Smem)
            diffusion_smem_kernel<<<grid, block, smem1>>>(cur, next, in.nx, in.ny, in.nz, inv_k2, p.dt);
        else {
            // The last launch takes whatever is left when iterations is not a multiple of K.
            steps = std::min(K, p.iterations - done);
            const int ox = cfg.bx - 2 * steps, oy = cfg.by - 2 * steps;  // output tile per block
            const dim3 g((in.nx + ox - 1) / ox, (in.ny + oy - 1) / oy, (in.nz + zchunk - 1) / zchunk);
            diffusion_fused_kernel<<<g, block2d, smem_fused(steps)>>>(cur, next, in.nx, in.ny, in.nz, steps, zchunk,
                                                                      inv_k2, p.dt);
        }
        CUDA_CHECK(cudaGetLastError());
        std::swap(cur, next);
        done += steps;
    }
    t.kernel_ms = timer.stop_ms();

    timer.start();
    CUDA_CHECK(cudaMemcpy(out.data.data(), cur, bytes, cudaMemcpyDeviceToHost));
    t.d2h_ms = timer.stop_ms();
    return t;
}
