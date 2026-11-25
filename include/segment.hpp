#pragma once

#include <cstdint>
#include <vector>

#include "volume.hpp"

// Downstream analysis: tissue segmentation of a (denoised) T1 volume inside a
// brain mask, optional morphological opening, and cleanup of small fragments
// found by 3D connected-component labeling.
//
// Pipeline (identical in every implementation, so outputs match exactly):
//   1. classify  histogram of the masked intensities, 1D k-means (3 classes)
//                on the histogram, then label each voxel 1 CSF, 2 GM, 3 WM
//                by the two midpoint thresholds (T1: CSF darkest)
//   2. open      optional: per-class binary opening with the 6-connected
//                cross; voxels removed by it are refilled by majority vote of
//                their kept 26-neighbors
//   3. ccl       26-connected labeling of same-class regions (union-find);
//                the component label is the smallest voxel index it contains
//   4. cleanup   voxels of components smaller than min_size are refilled by
//                majority vote of their kept 26-neighbors, then labeled again
//                to count the remaining fragments

struct SegmentParams {
    int bins = 4096;         // histogram bins over [0, hist_max)
    float hist_max = 2.0f;   // intensities are normalized to about [0, 1]
    bool open = false;       // run the opening stage
    int min_size = 10;       // components below this size are removed; 0 = no cleanup
};

struct SegmentStats {
    float thresholds[2] = {0, 0};
    float centers[3] = {0, 0, 0};
    int64_t comps_before[4] = {0, 0, 0, 0};  // per class 1..3 after step 2; [0] = total
    int64_t comps_after[4] = {0, 0, 0, 0};   // after cleanup
    // Stage times in milliseconds. GPU runs also report transfers.
    double classify_ms = 0, open_ms = 0, ccl_ms = 0, cleanup_ms = 0, h2d_ms = 0, d2h_ms = 0;
    double total() const { return classify_ms + open_ms + ccl_ms + cleanup_ms + h2d_ms + d2h_ms; }
};

void segment_cpu(const VolumeF& img, const VolumeU8& mask, VolumeU8& seg, const SegmentParams& p, bool parallel,
                 SegmentStats& st);
void segment_cuda(const VolumeF& img, const VolumeU8& mask, VolumeU8& seg, const SegmentParams& p,
                  SegmentStats& st);

// Connected-component labeling alone (26-connected, same nonzero value), for
// benchmarking and for exact cross-checks. labels[i] = smallest voxel index of
// i's component, or -1 for background. Returns the time in milliseconds
// (kernel time only for the GPU).
double ccl_cpu(const VolumeU8& seg, std::vector<int32_t>& labels, bool parallel);
double ccl_cuda(const VolumeU8& seg, std::vector<int32_t>& labels);

// Shared by every implementation: 1D k-means with 3 clusters on a histogram.
// Deterministic, so identical histograms give identical thresholds.
void kmeans_thresholds(const std::vector<int64_t>& hist, float hist_max, float centers[3], float thresholds[2]);

#ifdef __CUDACC__
#define SEG_HD __host__ __device__
#else
#define SEG_HD
#endif

// Bin of an intensity, shared by every implementation. `scale` is
// bins / hist_max computed once on the host, so CPU and GPU bin identically.
SEG_HD inline int hist_bin(float v, int bins, float scale) {
    const int b = int(v * scale);
    return b < 0 ? 0 : (b >= bins ? bins - 1 : b);
}

// Class of a masked voxel from the two thresholds: 1 CSF, 2 GM, 3 WM.
SEG_HD inline uint8_t classify_voxel(float v, float t0, float t1) { return v < t0 ? 1 : (v < t1 ? 2 : 3); }

// The k-th (0..12) "backward" neighbor of the 26-neighborhood, i.e. one that
// comes earlier in x-fastest order: the 9 in the plane below, the 3 in the row
// below, and the voxel to the left. Linking every voxel to these covers each
// adjacent pair exactly once.
SEG_HD inline void backward_offset(int k, int& dx, int& dy, int& dz) {
    if (k < 9) {
        dz = -1;
        dy = k / 3 - 1;
        dx = k % 3 - 1;
    } else if (k < 12) {
        dz = 0;
        dy = -1;
        dx = k - 10;
    } else {
        dz = 0;
        dy = 0;
        dx = -1;
    }
}
