#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include "segment.hpp"

namespace {

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Union-find. Linking always puts the larger root under the smaller one, so a
// root is the smallest index in its tree and the final labels do not depend
// on the order in which pairs are merged. The sequential version compresses
// paths as it goes; the parallel one uses relaxed atomics and a CAS-based
// atomic min (the CPU analogue of the GPU's atomicMin), as in Playne and
// Hawick's GPU union-find.
// ---------------------------------------------------------------------------
int32_t find_seq(int32_t* L, int32_t i) {
    while (L[i] != i) {
        L[i] = L[L[i]];  // path halving: point to the grandparent
        i = L[i];
    }
    return i;
}

void unite_seq(int32_t* L, int32_t a, int32_t b) {
    a = find_seq(L, a);
    b = find_seq(L, b);
    if (a < b) L[b] = a;
    else if (b < a) L[a] = b;
}

inline int32_t load(const int32_t* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }

int32_t find_par(const int32_t* L, int32_t i) {
    int32_t p;
    while ((p = load(&L[i])) != i) i = p;
    return i;
}

// Atomically sets *p = min(*p, v) and returns the previous value.
int32_t atomic_min_old(int32_t* p, int32_t v) {
    int32_t old = load(p);
    while (old > v && !__atomic_compare_exchange_n(p, &old, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    }
    return old;
}

void unite_par(int32_t* L, int32_t a, int32_t b) {
    for (;;) {
        a = find_par(L, a);
        b = find_par(L, b);
        if (a == b) return;
        if (a < b) std::swap(a, b);  // now b < a: hang root a under b
        const int32_t old = atomic_min_old(&L[a], b);
        if (old == a) return;        // a was still a root, link made
        a = old;                     // a was linked meanwhile; retry from its new parent
    }
}

void ccl_impl(const VolumeU8& seg, std::vector<int32_t>& L, bool parallel) {
    const int nx = seg.nx, ny = seg.ny, nz = seg.nz;
    const size_t n = seg.size();
    L.resize(n);
    const uint8_t* s = seg.data.data();
    int32_t* l = L.data();

#pragma omp parallel for schedule(static) if (parallel)
    for (size_t i = 0; i < n; ++i) l[i] = s[i] ? int32_t(i) : -1;

#pragma omp parallel for collapse(2) schedule(static) if (parallel)
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const size_t i = seg.idx(x, y, z);
                if (!s[i]) continue;
                for (int k = 0; k < 13; ++k) {
                    int dx, dy, dz;
                    backward_offset(k, dx, dy, dz);
                    const int xx = x + dx, yy = y + dy, zz = z + dz;
                    if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny) continue;
                    const size_t j = seg.idx(xx, yy, zz);
                    if (s[j] != s[i]) continue;
                    if (parallel) unite_par(l, int32_t(i), int32_t(j));
                    else unite_seq(l, int32_t(i), int32_t(j));
                }
            }

#pragma omp parallel for schedule(static) if (parallel)
    for (size_t i = 0; i < n; ++i)
        if (l[i] >= 0) {
            const int32_t r = parallel ? find_par(l, int32_t(i)) : find_seq(l, int32_t(i));
            __atomic_store_n(&l[i], r, __ATOMIC_RELAXED);
        }
}

// Majority-vote refill: every foreground voxel with keep == 0 takes the most
// frequent class among its kept foreground 26-neighbors (lowest class on a
// tie); if it has none, it keeps its label.
void fill(const VolumeU8& seg, const std::vector<uint8_t>& keep, VolumeU8& out, bool parallel) {
    const int nx = seg.nx, ny = seg.ny, nz = seg.nz;
    out = seg;
#pragma omp parallel for collapse(2) schedule(static) if (parallel)
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const size_t i = seg.idx(x, y, z);
                if (!seg.data[i] || keep[i]) continue;
                int cnt[4] = {0, 0, 0, 0};
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int xx = x + dx, yy = y + dy, zz = z + dz;
                            if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
                            const size_t j = seg.idx(xx, yy, zz);
                            if (seg.data[j] && keep[j]) ++cnt[seg.data[j]];
                        }
                int best = 1;
                for (int c = 2; c <= 3; ++c)
                    if (cnt[c] > cnt[best]) best = c;
                if (cnt[best] > 0) out.data[i] = uint8_t(best);
            }
}

// Per-class opening with the 6-connected cross: erosion keeps voxels whose 6
// in-volume neighbors share their class; dilation keeps voxels that are, or
// touch, an eroded voxel of their own class. Removed voxels are refilled.
void open_stage(VolumeU8& seg, bool parallel) {
    const int nx = seg.nx, ny = seg.ny, nz = seg.nz;
    const size_t n = seg.size();
    std::vector<uint8_t> E(n), keep(n);
    const uint8_t* s = seg.data.data();
    auto same6 = [&](int x, int y, int z, uint8_t c, const auto& pred) {
        const int d[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (auto& o : d) {
            const int xx = x + o[0], yy = y + o[1], zz = z + o[2];
            if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
            if (pred(seg.idx(xx, yy, zz), c)) return true;
        }
        return false;
    };
#pragma omp parallel for collapse(2) schedule(static) if (parallel)
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const size_t i = seg.idx(x, y, z);
                const uint8_t c = s[i];
                // Eroded iff no in-volume 6-neighbor has a different class.
                E[i] = c && !same6(x, y, z, c, [&](size_t j, uint8_t cc) { return s[j] != cc; });
            }
#pragma omp parallel for collapse(2) schedule(static) if (parallel)
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const size_t i = seg.idx(x, y, z);
                const uint8_t c = s[i];
                keep[i] = !c || E[i] || same6(x, y, z, c, [&](size_t j, uint8_t cc) { return E[j] && s[j] == cc; });
            }
    VolumeU8 out;
    fill(seg, keep, out, parallel);
    seg = std::move(out);
}

void count_components(const VolumeU8& seg, const std::vector<int32_t>& L, int64_t comps[4], bool parallel) {
    int64_t c1 = 0, c2 = 0, c3 = 0;
    const size_t n = seg.size();
#pragma omp parallel for reduction(+ : c1, c2, c3) schedule(static) if (parallel)
    for (size_t i = 0; i < n; ++i)
        if (L[i] == int32_t(i)) {
            const uint8_t c = seg.data[i];
            c1 += c == 1;
            c2 += c == 2;
            c3 += c == 3;
        }
    comps[1] = c1;
    comps[2] = c2;
    comps[3] = c3;
    comps[0] = c1 + c2 + c3;
}

}  // namespace

void kmeans_thresholds(const std::vector<int64_t>& hist, float hist_max, float centers[3], float thresholds[2]) {
    const int bins = int(hist.size());
    const double w = double(hist_max) / bins;
    int64_t total = 0;
    for (int64_t h : hist) total += h;
    // Start from the 15th, 50th and 85th percentiles.
    double c[3];
    const double q[3] = {0.15, 0.50, 0.85};
    for (int k = 0, b = 0; k < 3; ++k) {
        int64_t acc = 0;
        for (b = 0; b < bins; ++b) {
            acc += hist[b];
            if (acc >= q[k] * total) break;
        }
        c[k] = (std::min(b, bins - 1) + 0.5) * w;
    }
    for (int it = 0; it < 200; ++it) {
        const double t0 = 0.5 * (c[0] + c[1]), t1 = 0.5 * (c[1] + c[2]);
        double sum[3] = {0, 0, 0}, cnt[3] = {0, 0, 0};
        for (int b = 0; b < bins; ++b) {
            const double x = (b + 0.5) * w;
            const int k = x < t0 ? 0 : (x < t1 ? 1 : 2);
            sum[k] += x * hist[b];
            cnt[k] += hist[b];
        }
        bool changed = false;
        for (int k = 0; k < 3; ++k) {
            const double nc = cnt[k] > 0 ? sum[k] / cnt[k] : c[k];
            changed |= nc != c[k];
            c[k] = nc;
        }
        if (!changed) break;
    }
    std::sort(c, c + 3);
    for (int k = 0; k < 3; ++k) centers[k] = float(c[k]);
    thresholds[0] = float(0.5 * (c[0] + c[1]));
    thresholds[1] = float(0.5 * (c[1] + c[2]));
}

double ccl_cpu(const VolumeU8& seg, std::vector<int32_t>& labels, bool parallel) {
    const double t0 = now_ms();
    ccl_impl(seg, labels, parallel);
    return now_ms() - t0;
}

void segment_cpu(const VolumeF& img, const VolumeU8& mask, VolumeU8& seg, const SegmentParams& p, bool parallel,
                 SegmentStats& st) {
    const size_t n = img.size();
    const float scale = float(p.bins) / p.hist_max;

    // 1. classify: histogram (per-thread copies merged at the end), k-means, map.
    double t = now_ms();
    std::vector<int64_t> hist(p.bins, 0);
#pragma omp parallel if (parallel)
    {
        std::vector<int64_t> local(p.bins, 0);
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; ++i)
            if (mask.data[i]) ++local[hist_bin(img.data[i], p.bins, scale)];
#pragma omp critical
        for (int b = 0; b < p.bins; ++b) hist[b] += local[b];
    }
    kmeans_thresholds(hist, p.hist_max, st.centers, st.thresholds);
    seg = VolumeU8(img.nx, img.ny, img.nz);
    const float t0 = st.thresholds[0], t1 = st.thresholds[1];
#pragma omp parallel for schedule(static) if (parallel)
    for (size_t i = 0; i < n; ++i) seg.data[i] = mask.data[i] ? classify_voxel(img.data[i], t0, t1) : 0;
    st.classify_ms = now_ms() - t;

    // 2. optional opening.
    t = now_ms();
    if (p.open) open_stage(seg, parallel);
    st.open_ms = now_ms() - t;

    // 3. label and count.
    t = now_ms();
    std::vector<int32_t> L;
    ccl_impl(seg, L, parallel);
    count_components(seg, L, st.comps_before, parallel);
    st.ccl_ms = now_ms() - t;

    // 4. remove small components, then label again to count what is left.
    t = now_ms();
    if (p.min_size > 0) {
        std::vector<int32_t> size(n, 0);
#pragma omp parallel for schedule(static) if (parallel)
        for (size_t i = 0; i < n; ++i)
            if (L[i] >= 0) {
#pragma omp atomic
                ++size[L[i]];
            }
        std::vector<uint8_t> keep(n);
#pragma omp parallel for schedule(static) if (parallel)
        for (size_t i = 0; i < n; ++i) keep[i] = L[i] < 0 || size[L[i]] >= p.min_size;
        VolumeU8 out;
        fill(seg, keep, out, parallel);
        seg = std::move(out);
        ccl_impl(seg, L, parallel);
        count_components(seg, L, st.comps_after, parallel);
    } else {
        std::copy(st.comps_before, st.comps_before + 4, st.comps_after);
    }
    st.cleanup_ms = now_ms() - t;
}
