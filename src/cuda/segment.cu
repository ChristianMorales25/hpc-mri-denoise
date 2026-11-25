#include <cstdint>
#include <vector>

#include "cuda_check.cuh"
#include "segment.hpp"

// GPU version of the segmentation pipeline in src/cpu/segment.cpp. Every
// stage computes exactly what the CPU version computes, so outputs are
// bit-identical (checked by scripts/check_segment.sh).

namespace {

template <typename T>
struct DevArray {
    T* ptr = nullptr;
    explicit DevArray(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
    ~DevArray() { cudaFree(ptr); }
    DevArray(const DevArray&) = delete;
    DevArray& operator=(const DevArray&) = delete;
};

struct Dims {
    int nx, ny, nz;
    __device__ size_t idx(int x, int y, int z) const { return (size_t(z) * ny + y) * nx + x; }
    __device__ bool inside(int x, int y, int z) const {
        return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz;
    }
};

// One thread per voxel on a 3D grid; x is the fastest index so warps read
// consecutive voxels.
#define VOXEL_XYZ(d)                                       \
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   \
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   \
    const int z = blockIdx.z * blockDim.z + threadIdx.z;   \
    if (x >= (d).nx || y >= (d).ny || z >= (d).nz) return; \
    const size_t i = (d).idx(x, y, z)

// --- 1. classify -------------------------------------------------------------

// Histogram privatized in shared memory: each block counts into its own copy
// with fast shared-memory atomics, then adds it to the global histogram once.
__global__ void histogram_kernel(const float* __restrict__ img, const uint8_t* __restrict__ mask, size_t n,
                                 int bins, float scale, unsigned int* __restrict__ hist) {
    extern __shared__ unsigned int local[];
    for (int b = threadIdx.x; b < bins; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        if (mask[i]) atomicAdd(&local[hist_bin(img[i], bins, scale)], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
        if (local[b]) atomicAdd(&hist[b], local[b]);
}

__global__ void classify_kernel(const float* __restrict__ img, const uint8_t* __restrict__ mask,
                                uint8_t* __restrict__ seg, size_t n, float t0, float t1) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) seg[i] = mask[i] ? classify_voxel(img[i], t0, t1) : 0;
}

// --- 2. opening and refill ---------------------------------------------------

__constant__ int c_six[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

__global__ void erode_kernel(const uint8_t* __restrict__ seg, uint8_t* __restrict__ E, Dims d) {
    VOXEL_XYZ(d);
    const uint8_t c = seg[i];
    bool e = c != 0;
    for (int k = 0; k < 6 && e; ++k) {
        const int xx = x + c_six[k][0], yy = y + c_six[k][1], zz = z + c_six[k][2];
        if (d.inside(xx, yy, zz) && seg[d.idx(xx, yy, zz)] != c) e = false;
    }
    E[i] = e;
}

__global__ void dilate_keep_kernel(const uint8_t* __restrict__ seg, const uint8_t* __restrict__ E,
                                   uint8_t* __restrict__ keep, Dims d) {
    VOXEL_XYZ(d);
    const uint8_t c = seg[i];
    bool k = !c || E[i];
    for (int j = 0; j < 6 && !k; ++j) {
        const int xx = x + c_six[j][0], yy = y + c_six[j][1], zz = z + c_six[j][2];
        if (d.inside(xx, yy, zz)) {
            const size_t u = d.idx(xx, yy, zz);
            k = E[u] && seg[u] == c;
        }
    }
    keep[i] = k;
}

__global__ void fill_kernel(const uint8_t* __restrict__ seg, const uint8_t* __restrict__ keep,
                            uint8_t* __restrict__ out, Dims d) {
    VOXEL_XYZ(d);
    const uint8_t c = seg[i];
    out[i] = c;
    if (!c || keep[i]) return;
    int cnt[4] = {0, 0, 0, 0};
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int xx = x + dx, yy = y + dy, zz = z + dz;
                if (!d.inside(xx, yy, zz)) continue;
                const size_t j = d.idx(xx, yy, zz);
                if (seg[j] && keep[j]) ++cnt[seg[j]];
            }
    int best = 1;
    if (cnt[2] > cnt[best]) best = 2;
    if (cnt[3] > cnt[best]) best = 3;
    if (cnt[best] > 0) out[i] = uint8_t(best);
}

// --- 3. connected components: union-find (Playne and Hawick) -----------------

__global__ void ccl_init_kernel(const uint8_t* __restrict__ seg, int32_t* __restrict__ L, size_t n) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) L[i] = seg[i] ? int32_t(i) : -1;
}

// Reads go through a volatile pointer so a thread always sees links made by
// other threads instead of a stale L1-cached value.
__device__ int32_t find_root(const volatile int32_t* L, int32_t i) {
    int32_t p;
    while ((p = L[i]) != i) i = p;
    return i;
}

__device__ void unite(int32_t* L, int32_t a, int32_t b) {
    for (;;) {
        a = find_root(L, a);
        b = find_root(L, b);
        if (a == b) return;
        if (a < b) {
            const int32_t t = a;
            a = b;
            b = t;
        }
        // b < a: hang root a under b. If a stopped being a root in the
        // meantime, atomicMin returns its new parent and we retry from there.
        const int32_t old = atomicMin(&L[a], b);
        if (old == a) return;
        a = old;
    }
}

__global__ void ccl_merge_kernel(const uint8_t* __restrict__ seg, int32_t* L, Dims d) {
    VOXEL_XYZ(d);
    const uint8_t c = seg[i];
    if (!c) return;
    for (int k = 0; k < 13; ++k) {
        int dx, dy, dz;
        backward_offset(k, dx, dy, dz);
        const int xx = x + dx, yy = y + dy, zz = z + dz;
        if (!d.inside(xx, yy, zz)) continue;
        const size_t j = d.idx(xx, yy, zz);
        if (seg[j] == c) unite(L, int32_t(i), int32_t(j));
    }
}

__global__ void ccl_compress_kernel(int32_t* L, size_t n) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n && L[i] >= 0) L[i] = find_root(L, int32_t(i));
}

__global__ void count_roots_kernel(const uint8_t* __restrict__ seg, const int32_t* __restrict__ L, size_t n,
                                   unsigned long long* counts) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n && L[i] == int32_t(i)) atomicAdd(&counts[seg[i]], 1ull);
}

// --- 4. small-component removal ----------------------------------------------

__global__ void component_size_kernel(const int32_t* __restrict__ L, size_t n, int32_t* size) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n && L[i] >= 0) atomicAdd(&size[L[i]], 1);
}

__global__ void keep_large_kernel(const int32_t* __restrict__ L, const int32_t* __restrict__ size, size_t n,
                                  int min_size, uint8_t* __restrict__ keep) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) keep[i] = L[i] < 0 || size[L[i]] >= min_size;
}

// --- host side ---------------------------------------------------------------

constexpr int kThreads1D = 256;
inline unsigned blocks1d(size_t n) { return unsigned((n + kThreads1D - 1) / kThreads1D); }

struct Launch3D {
    dim3 block{32, 4, 2};
    dim3 grid;
    explicit Launch3D(const Dims& d) : grid(grid_for(d.nx, d.ny, d.nz, block)) {}
};

void ccl_device(const uint8_t* seg, int32_t* L, const Dims& d, size_t n) {
    const Launch3D l(d);
    ccl_init_kernel<<<blocks1d(n), kThreads1D>>>(seg, L, n);
    ccl_merge_kernel<<<l.grid, l.block>>>(seg, L, d);
    ccl_compress_kernel<<<blocks1d(n), kThreads1D>>>(L, n);
    CUDA_CHECK(cudaGetLastError());
}

void count_device(const uint8_t* seg, const int32_t* L, size_t n, unsigned long long* d_counts, int64_t comps[4]) {
    CUDA_CHECK(cudaMemset(d_counts, 0, 4 * sizeof(unsigned long long)));
    count_roots_kernel<<<blocks1d(n), kThreads1D>>>(seg, L, n, d_counts);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long h[4];
    CUDA_CHECK(cudaMemcpy(h, d_counts, sizeof(h), cudaMemcpyDeviceToHost));
    comps[1] = int64_t(h[1]);
    comps[2] = int64_t(h[2]);
    comps[3] = int64_t(h[3]);
    comps[0] = comps[1] + comps[2] + comps[3];
}

}  // namespace

double ccl_cuda(const VolumeU8& seg, std::vector<int32_t>& labels) {
    const size_t n = seg.size();
    const Dims d{seg.nx, seg.ny, seg.nz};
    DevArray<uint8_t> d_seg(n);
    DevArray<int32_t> d_L(n);
    CUDA_CHECK(cudaMemcpy(d_seg.ptr, seg.data.data(), n, cudaMemcpyHostToDevice));
    EventTimer timer;
    timer.start();
    ccl_device(d_seg.ptr, d_L.ptr, d, n);
    const double ms = timer.stop_ms();
    labels.resize(n);
    CUDA_CHECK(cudaMemcpy(labels.data(), d_L.ptr, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    return ms;
}

void segment_cuda(const VolumeF& img, const VolumeU8& mask, VolumeU8& seg, const SegmentParams& p,
                  SegmentStats& st) {
    const size_t n = img.size();
    const Dims d{img.nx, img.ny, img.nz};
    const Launch3D l3(d);
    DevArray<float> d_img(n);
    DevArray<uint8_t> d_mask(n), d_seg(n), d_tmp(n), d_keep(n);
    DevArray<int32_t> d_L(n), d_size(n);
    DevArray<unsigned int> d_hist(p.bins);
    DevArray<unsigned long long> d_counts(4);
    EventTimer timer;

    timer.start();
    CUDA_CHECK(cudaMemcpy(d_img.ptr, img.data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask.ptr, mask.data.data(), n, cudaMemcpyHostToDevice));
    st.h2d_ms = timer.stop_ms();

    // 1. classify. The k-means itself runs on the host over the 4096-bin
    // histogram (microseconds); only the histogram crosses the bus.
    timer.start();
    CUDA_CHECK(cudaMemset(d_hist.ptr, 0, p.bins * sizeof(unsigned int)));
    const float scale = float(p.bins) / p.hist_max;
    histogram_kernel<<<160, 256, p.bins * sizeof(unsigned int)>>>(d_img.ptr, d_mask.ptr, n, p.bins, scale, d_hist.ptr);
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned int> h32(p.bins);
    CUDA_CHECK(cudaMemcpy(h32.data(), d_hist.ptr, p.bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    std::vector<int64_t> hist(h32.begin(), h32.end());
    kmeans_thresholds(hist, p.hist_max, st.centers, st.thresholds);
    classify_kernel<<<blocks1d(n), kThreads1D>>>(d_img.ptr, d_mask.ptr, d_seg.ptr, n, st.thresholds[0],
                                                 st.thresholds[1]);
    CUDA_CHECK(cudaGetLastError());
    st.classify_ms = timer.stop_ms();

    // 2. optional opening. d_tmp holds the eroded mask, then the refilled labels.
    timer.start();
    if (p.open) {
        erode_kernel<<<l3.grid, l3.block>>>(d_seg.ptr, d_tmp.ptr, d);
        dilate_keep_kernel<<<l3.grid, l3.block>>>(d_seg.ptr, d_tmp.ptr, d_keep.ptr, d);
        fill_kernel<<<l3.grid, l3.block>>>(d_seg.ptr, d_keep.ptr, d_tmp.ptr, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_seg.ptr, d_tmp.ptr, n, cudaMemcpyDeviceToDevice));
    }
    st.open_ms = timer.stop_ms();

    // 3. label and count.
    timer.start();
    ccl_device(d_seg.ptr, d_L.ptr, d, n);
    count_device(d_seg.ptr, d_L.ptr, n, d_counts.ptr, st.comps_before);
    st.ccl_ms = timer.stop_ms();

    // 4. remove small components and count again.
    timer.start();
    if (p.min_size > 0) {
        CUDA_CHECK(cudaMemset(d_size.ptr, 0, n * sizeof(int32_t)));
        component_size_kernel<<<blocks1d(n), kThreads1D>>>(d_L.ptr, n, d_size.ptr);
        keep_large_kernel<<<blocks1d(n), kThreads1D>>>(d_L.ptr, d_size.ptr, n, p.min_size, d_keep.ptr);
        fill_kernel<<<l3.grid, l3.block>>>(d_seg.ptr, d_keep.ptr, d_tmp.ptr, d);
        CUDA_CHECK(cudaGetLastError());
        std::swap(d_seg.ptr, d_tmp.ptr);
        ccl_device(d_seg.ptr, d_L.ptr, d, n);
        count_device(d_seg.ptr, d_L.ptr, n, d_counts.ptr, st.comps_after);
    } else {
        std::copy(st.comps_before, st.comps_before + 4, st.comps_after);
    }
    st.cleanup_ms = timer.stop_ms();

    timer.start();
    seg = VolumeU8(img.nx, img.ny, img.nz);
    CUDA_CHECK(cudaMemcpy(seg.data.data(), d_seg.ptr, n, cudaMemcpyDeviceToHost));
    st.d2h_ms = timer.stop_ms();
}
