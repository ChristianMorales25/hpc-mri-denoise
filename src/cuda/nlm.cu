#include <stdexcept>
#include <string>

#include "cuda_check.cuh"
#include "filters.hpp"

namespace {

// ---------------------------------------------------------------------------
// v1 naive: one thread per voxel, mirrors nlm_cpu exactly. Every patch
// comparison re-reads (2P+1)^3 voxels from global memory, so the same voxels
// are loaded many times by neighboring threads.
// ---------------------------------------------------------------------------
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

// Cooperatively copies the block's input region plus a halo of H voxels on
// every side into shared memory. Coordinates are clamped on load, which gives
// the same replicate-border behavior as the CPU version.
__device__ void load_tile(const float* __restrict__ in, float* tile, int nx, int ny, int nz, int ox, int oy,
                          int oz, int tx, int ty, int tz) {
    const int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;
    const int n = tx * ty * tz;
    for (int i = tid; i < n; i += nthreads) {
        const int lx = i % tx, ly = (i / tx) % ty, lz = i / (tx * ty);
        tile[i] = in[(size_t(clampi(oz + lz, nz)) * ny + clampi(oy + ly, ny)) * nx + clampi(ox + lx, nx)];
    }
}

// ---------------------------------------------------------------------------
// v2 shared-memory tile: identical arithmetic to v1, but all reads after the
// initial load come from shared memory. The tile holds the block plus a halo
// of H = S + P voxels, enough for every patch at every search offset.
// ---------------------------------------------------------------------------
__global__ void nlm_smem_kernel(const float* __restrict__ in, float* __restrict__ out, int nx, int ny, int nz,
                                int z0, int z1, int S, int P, float inv_patch_n, float inv_h2) {
    extern __shared__ float tile[];
    const int H = S + P;
    const int tx = blockDim.x + 2 * H, ty = blockDim.y + 2 * H, tz = blockDim.z + 2 * H;
    const int bx0 = blockIdx.x * blockDim.x, by0 = blockIdx.y * blockDim.y, bz0 = z0 + blockIdx.z * blockDim.z;
    load_tile(in, tile, nx, ny, nz, bx0 - H, by0 - H, bz0 - H, tx, ty, tz);
    __syncthreads();

    const int x = bx0 + threadIdx.x, y = by0 + threadIdx.y, z = bz0 + threadIdx.z;
    if (x >= nx || y >= ny || z >= z1) return;

    auto T = [&](int lx, int ly, int lz) { return tile[(lz * ty + ly) * tx + lx]; };
    const int cx = threadIdx.x + H, cy = threadIdx.y + H, cz = threadIdx.z + H;

    float sum = 0.0f, wsum = 0.0f, wmax = 0.0f;
    for (int sz = -S; sz <= S; ++sz)
        for (int sy = -S; sy <= S; ++sy)
            for (int sx = -S; sx <= S; ++sx) {
                if (sx == 0 && sy == 0 && sz == 0) continue;
                float d2 = 0.0f;
                for (int pz = -P; pz <= P; ++pz)
                    for (int py = -P; py <= P; ++py)
                        for (int px = -P; px <= P; ++px) {
                            const float d = T(cx + px, cy + py, cz + pz) - T(cx + sx + px, cy + sy + py, cz + sz + pz);
                            d2 += d * d;
                        }
                const float wt = __expf(-(d2 * inv_patch_n) * inv_h2);
                sum += wt * T(cx + sx, cy + sy, cz + sz);
                wsum += wt;
                wmax = fmaxf(wmax, wt);
            }
    if (wmax == 0.0f) wmax = 1.0f;
    out[(size_t(z) * ny + y) * nx + x] = (sum + wmax * T(cx, cy, cz)) / (wsum + wmax);
}

// ---------------------------------------------------------------------------
// v3 shared memory + separable patch distance. For a fixed search offset s,
// the patch distance of every voxel is a (2P+1)^3 box sum of the squared
// difference image D(v) = (I(v) - I(v+s))^2. The block computes D over its
// region plus a P-halo once, then box-sums it with three 1D passes, so each
// voxel costs O(P) adds per offset instead of O(P^3) reads and FMAs. The
// accumulators stay in registers across all offsets.
// Each thread handles the same shared-memory elements for every offset, so
// their indices are computed once up front (integer div/mod is expensive and
// would otherwise run 3x per element per offset).
// Summation order differs from v1/v2, so results match to float rounding.
// ---------------------------------------------------------------------------
constexpr int kSepMaxPerThread = 8;  // max elements per thread in any stage

__global__ void nlm_sep_kernel(const float* __restrict__ in, float* __restrict__ out, int nx, int ny, int nz,
                               int z0, int z1, int S, int P, float inv_patch_n, float inv_h2) {
    extern __shared__ float smem[];
    const int H = S + P;
    const int bdx = blockDim.x, bdy = blockDim.y, bdz = blockDim.z;
    const int tx = bdx + 2 * H, ty = bdy + 2 * H, tz = bdz + 2 * H;
    const int rx = bdx + 2 * P, ry = bdy + 2 * P, rz = bdz + 2 * P;  // D region
    float* tile = smem;
    float* D = tile + tx * ty * tz;    // rx * ry * rz
    float* B1 = D + rx * ry * rz;      // bdx * ry * rz  (summed along x)
    float* B2 = B1 + bdx * ry * rz;    // bdx * bdy * rz (summed along x, y)

    const int bx0 = blockIdx.x * bdx, by0 = blockIdx.y * bdy, bz0 = z0 + blockIdx.z * bdz;
    load_tile(in, tile, nx, ny, nz, bx0 - H, by0 - H, bz0 - H, tx, ty, tz);
    __syncthreads();

    const int tid = (threadIdx.z * bdy + threadIdx.y) * bdx + threadIdx.x;
    const int nthreads = bdx * bdy * bdz;
    const int x = bx0 + threadIdx.x, y = by0 + threadIdx.y, z = bz0 + threadIdx.z;
    // Out-of-range threads still take part in the cooperative passes and
    // barriers; they just do not write a result.
    const bool active = x < nx && y < ny && z < z1;

    auto T = [&](int lx, int ly, int lz) { return tile[(lz * ty + ly) * tx + lx]; };
    const int off = H - P;  // tile coordinate of D's origin
    const int cx = threadIdx.x + H, cy = threadIdx.y + H, cz = threadIdx.z + H;
    const int w = 2 * P + 1;
    const int nD = rx * ry * rz, nB1 = bdx * ry * rz, nB2 = bdx * bdy * rz;

    // Per-thread source indices for the three stages; -1 marks "no element".
    int dSrc[kSepMaxPerThread], b1Src[kSepMaxPerThread], b2Src[kSepMaxPerThread];
#pragma unroll
    for (int k = 0; k < kSepMaxPerThread; ++k) {
        const int i = tid + k * nthreads;
        dSrc[k] = i < nD ? ((i / (rx * ry) + off) * ty + (i / rx) % ry + off) * tx + i % rx + off : -1;
        b1Src[k] = i < nB1 ? (i / bdx) * rx + i % bdx : -1;
        b2Src[k] = i < nB2 ? ((i / (bdx * bdy)) * ry + (i / bdx) % bdy) * bdx + i % bdx : -1;
    }

    float sum = 0.0f, wsum = 0.0f, wmax = 0.0f;
    for (int sz = -S; sz <= S; ++sz)
        for (int sy = -S; sy <= S; ++sy)
            for (int sx = -S; sx <= S; ++sx) {
                if (sx == 0 && sy == 0 && sz == 0) continue;
                const int shift = (sz * ty + sy) * tx + sx;

#pragma unroll
                for (int k = 0; k < kSepMaxPerThread; ++k)
                    if (dSrc[k] >= 0) {
                        const float d = tile[dSrc[k]] - tile[dSrc[k] + shift];
                        D[tid + k * nthreads] = d * d;
                    }
                __syncthreads();
#pragma unroll
                for (int k = 0; k < kSepMaxPerThread; ++k)
                    if (b1Src[k] >= 0) {
                        float acc = 0.0f;
                        for (int j = 0; j < w; ++j) acc += D[b1Src[k] + j];
                        B1[tid + k * nthreads] = acc;
                    }
                __syncthreads();
#pragma unroll
                for (int k = 0; k < kSepMaxPerThread; ++k)
                    if (b2Src[k] >= 0) {
                        float acc = 0.0f;
                        for (int j = 0; j < w; ++j) acc += B1[b2Src[k] + j * bdx];
                        B2[tid + k * nthreads] = acc;
                    }
                __syncthreads();

                if (active) {
                    float d2 = 0.0f;
                    for (int k = 0; k < w; ++k)
                        d2 += B2[((threadIdx.z + k) * bdy + threadIdx.y) * bdx + threadIdx.x];
                    const float wt = __expf(-(d2 * inv_patch_n) * inv_h2);
                    sum += wt * T(cx + sx, cy + sy, cz + sz);
                    wsum += wt;
                    wmax = fmaxf(wmax, wt);
                }
                // No barrier needed here: the next offset's first write is to
                // D, which nobody reads until after the next barrier, and B2
                // is not overwritten until two barriers later.
            }
    if (!active) return;
    if (wmax == 0.0f) wmax = 1.0f;
    out[(size_t(z) * ny + y) * nx + x] = (sum + wmax * T(cx, cy, cz)) / (wsum + wmax);
}

size_t smem_bytes(NlmVariant v, const dim3& b, int S, int P) {
    const size_t H = S + P;
    const size_t tile = (b.x + 2 * H) * (b.y + 2 * H) * (b.z + 2 * H);
    if (v == NlmVariant::Smem) return tile * sizeof(float);
    if (v == NlmVariant::Sep) {
        const size_t rx = b.x + 2 * P, ry = b.y + 2 * P, rz = b.z + 2 * P;
        return (tile + rx * ry * rz + b.x * ry * rz + b.x * b.y * rz) * sizeof(float);
    }
    return 0;
}

}  // namespace

GpuTiming nlm_cuda(const VolumeF& in, VolumeF& out, const NlmParams& p, const LaunchConfig& cfg,
                   NlmVariant variant) {
    const dim3 block(cfg.bx, cfg.by, cfg.bz);
    const size_t shmem = smem_bytes(variant, block, p.search_radius, p.patch_radius);
    auto kernel = variant == NlmVariant::Naive ? nlm_naive_kernel
                  : variant == NlmVariant::Smem ? nlm_smem_kernel
                                                : nlm_sep_kernel;
    if (variant == NlmVariant::Sep) {
        const int P = p.patch_radius;
        const long nD = long(block.x + 2 * P) * (block.y + 2 * P) * (block.z + 2 * P);
        if (nD > long(kSepMaxPerThread) * block.x * block.y * block.z)
            throw std::runtime_error("cuda-sep: block too small for this patch radius; use a larger --block");
    }
    if (shmem > 0) {
        // Above 48 KB per block the kernel must opt in explicitly (sm_89 allows ~99 KB).
        int dev = 0, limit = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        if (shmem > size_t(limit))
            throw std::runtime_error("NLM tile needs " + std::to_string(shmem) + " B shared memory, device allows " +
                                     std::to_string(limit) + "; use a smaller --block or search radius");
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shmem)));
    }

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
    // Process the volume in z-slabs: a single launch over a large volume can
    // exceed the ~2 s Windows/WSL display-driver watchdog (TDR) and be killed.
    const int slab = cfg.chunk_z > 0 ? cfg.chunk_z : in.nz;
    timer.start();
    for (int z0 = 0; z0 < in.nz; z0 += slab) {
        const int z1 = z0 + slab < in.nz ? z0 + slab : in.nz;
        kernel<<<grid_for(in.nx, in.ny, z1 - z0, block), block, shmem>>>(
            d_in.ptr, d_out.ptr, in.nx, in.ny, in.nz, z0, z1, p.search_radius, p.patch_radius, inv_patch_n, inv_h2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    t.kernel_ms = timer.stop_ms();

    timer.start();
    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out.ptr, bytes, cudaMemcpyDeviceToHost));
    t.d2h_ms = timer.stop_ms();
    return t;
}
