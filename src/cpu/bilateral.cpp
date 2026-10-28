#include <cmath>
#include <vector>

#include "filters.hpp"

void bilateral_cpu(const VolumeF& in, VolumeF& out, const BilateralParams& p, bool parallel) {
    out = VolumeF(in.nx, in.ny, in.nz);
    const int r = p.radius;
    const int w = 2 * r + 1;
    const float inv_2ss = 1.0f / (2.0f * p.sigma_spatial * p.sigma_spatial);
    const float inv_2sr = 1.0f / (2.0f * p.sigma_range * p.sigma_range);

    // Spatial weights depend only on the offset, so precompute them once.
    std::vector<float> spatial(size_t(w) * w * w);
    for (int dz = -r; dz <= r; ++dz)
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
                spatial[((dz + r) * w + (dy + r)) * w + (dx + r)] =
                    std::exp(-float(dx * dx + dy * dy + dz * dz) * inv_2ss);

#pragma omp parallel for collapse(2) schedule(static) if (parallel)
    for (int z = 0; z < in.nz; ++z) {
        for (int y = 0; y < in.ny; ++y) {
            for (int x = 0; x < in.nx; ++x) {
                const float c = in.at(x, y, z);
                float sum = 0.0f, wsum = 0.0f;
                for (int dz = -r; dz <= r; ++dz)
                    for (int dy = -r; dy <= r; ++dy)
                        for (int dx = -r; dx <= r; ++dx) {
                            const float v = in.clamped(x + dx, y + dy, z + dz);
                            const float d = v - c;
                            const float wt =
                                spatial[((dz + r) * w + (dy + r)) * w + (dx + r)] * std::exp(-d * d * inv_2sr);
                            sum += wt * v;
                            wsum += wt;
                        }
                out.at(x, y, z) = sum / wsum;
            }
        }
    }
}
