#include <cmath>
#include <utility>

#include "filters.hpp"

// Perona-Malik anisotropic diffusion, explicit 6-neighbor scheme with
// conductance g(d) = exp(-(d/kappa)^2) and zero-flux (replicate) borders.
void diffusion_cpu(const VolumeF& in, VolumeF& out, const DiffusionParams& p, bool parallel) {
    VolumeF cur = in;
    VolumeF next(in.nx, in.ny, in.nz);
    const float inv_k2 = 1.0f / (p.kappa * p.kappa);
    auto g = [inv_k2](float d) { return std::exp(-d * d * inv_k2); };

    for (int it = 0; it < p.iterations; ++it) {
#pragma omp parallel for collapse(2) schedule(static) if (parallel)
        for (int z = 0; z < in.nz; ++z) {
            for (int y = 0; y < in.ny; ++y) {
                for (int x = 0; x < in.nx; ++x) {
                    const float c = cur.at(x, y, z);
                    const float dxp = cur.clamped(x + 1, y, z) - c;
                    const float dxm = cur.clamped(x - 1, y, z) - c;
                    const float dyp = cur.clamped(x, y + 1, z) - c;
                    const float dym = cur.clamped(x, y - 1, z) - c;
                    const float dzp = cur.clamped(x, y, z + 1) - c;
                    const float dzm = cur.clamped(x, y, z - 1) - c;
                    const float flux = g(dxp) * dxp + g(dxm) * dxm + g(dyp) * dyp + g(dym) * dym +
                                       g(dzp) * dzp + g(dzm) * dzm;
                    next.at(x, y, z) = c + p.dt * flux;
                }
            }
        }
        std::swap(cur, next);
    }
    out = std::move(cur);
}
