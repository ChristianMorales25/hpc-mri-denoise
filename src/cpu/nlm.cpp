#include <algorithm>
#include <cmath>

#include "filters.hpp"

// Pixelwise non-local means (Buades et al.) in 3D. Each output voxel is a
// weighted average of the voxels in its search window, weighted by how similar
// their surrounding patches are to the patch around the center voxel.
// Cost per voxel: (2S+1)^3 * (2P+1)^3, which is what makes this the main
// workload of the project.
void nlm_cpu(const VolumeF& in, VolumeF& out, const NlmParams& p, bool parallel) {
    out = VolumeF(in.nx, in.ny, in.nz);
    const int S = p.search_radius;
    const int P = p.patch_radius;
    const float patch_n = float((2 * P + 1) * (2 * P + 1) * (2 * P + 1));
    const float inv_h2 = 1.0f / (p.h * p.h);

#pragma omp parallel for collapse(2) schedule(dynamic, 1) if (parallel)
    for (int z = 0; z < in.nz; ++z) {
        for (int y = 0; y < in.ny; ++y) {
            for (int x = 0; x < in.nx; ++x) {
                float sum = 0.0f, wsum = 0.0f, wmax = 0.0f;
                for (int sz = -S; sz <= S; ++sz)
                    for (int sy = -S; sy <= S; ++sy)
                        for (int sx = -S; sx <= S; ++sx) {
                            if (sx == 0 && sy == 0 && sz == 0) continue;
                            float d2 = 0.0f;
                            for (int pz = -P; pz <= P; ++pz)
                                for (int py = -P; py <= P; ++py)
                                    for (int px = -P; px <= P; ++px) {
                                        const float a = in.clamped(x + px, y + py, z + pz);
                                        const float b = in.clamped(x + sx + px, y + sy + py, z + sz + pz);
                                        d2 += (a - b) * (a - b);
                                    }
                            const float wt = std::exp(-(d2 / patch_n) * inv_h2);
                            sum += wt * in.clamped(x + sx, y + sy, z + sz);
                            wsum += wt;
                            wmax = std::max(wmax, wt);
                        }
                // The center voxel would always get weight 1; give it the
                // largest neighbor weight instead so it does not dominate.
                if (wmax == 0.0f) wmax = 1.0f;
                sum += wmax * in.at(x, y, z);
                wsum += wmax;
                out.at(x, y, z) = sum / wsum;
            }
        }
    }
}
