#pragma once

#include "volume.hpp"

// Parameters shared by the CPU and GPU implementations so every variant
// computes the same thing and can be checked against the sequential baseline.

struct BilateralParams {
    int radius = 2;          // window is (2r+1)^3
    float sigma_spatial = 1.5f;
    float sigma_range = 0.1f;  // intensities are normalized to [0, 1]
};

struct NlmParams {
    int search_radius = 3;   // search window (2S+1)^3
    int patch_radius = 1;    // patch (2P+1)^3
    float h = 0.05f;         // filtering strength
};

struct DiffusionParams {
    int iterations = 20;
    float kappa = 0.05f;     // edge-stopping threshold
    float dt = 1.0f / 7.0f;  // explicit scheme is stable for dt <= 1/6 in 3D
};

// CPU implementations. `parallel = false` is the sequential baseline; `true`
// uses OpenMP with however many threads omp_set_num_threads() configured.
void bilateral_cpu(const VolumeF& in, VolumeF& out, const BilateralParams& p, bool parallel);
void nlm_cpu(const VolumeF& in, VolumeF& out, const NlmParams& p, bool parallel);
void diffusion_cpu(const VolumeF& in, VolumeF& out, const DiffusionParams& p, bool parallel);

// GPU implementations. Timings (milliseconds) are split so the report can
// show transfer cost separately from compute.
struct GpuTiming {
    float h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
    float total() const { return h2d_ms + kernel_ms + d2h_ms; }
};

struct LaunchConfig {
    int bx = 8, by = 8, bz = 4;
};

GpuTiming bilateral_cuda_naive(const VolumeF& in, VolumeF& out, const BilateralParams& p,
                               const LaunchConfig& cfg);

void print_device_info();
