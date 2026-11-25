// Command-line driver for the downstream stage. Prints one JSON line with the
// median stage times, the per-class Dice against ground truth, and the
// component counts before and after cleanup.
//
//   segment --impl cuda -i results/nlm.vol --mask data/brainweb/labels.vol -o results/seg.vol
//   segment --mode ccl --impl omp -i results/seg.vol      (labeling alone)

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "filters.hpp"
#include "segment.hpp"

namespace {

struct Args {
    std::string mode = "pipeline", impl = "seq", input, mask, truth, output, ref;
    int threads = 0, reps = 3;
    SegmentParams seg;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: segment [--mode pipeline|ccl] --impl seq|omp|cuda -i IN [options]\n"
                 "  pipeline: IN is a float volume; --mask LABELS (brain mask = labels > 0) is required\n"
                 "  ccl:      IN is a uint8 segmentation; times connected-component labeling alone\n"
                 "  --truth LABELS     ground truth for Dice (default: the --mask file)\n"
                 "  --open             enable the morphological opening stage\n"
                 "  --min-size N       remove components smaller than N voxels (default 10, 0 = off)\n"
                 "  --threads N --reps N -o OUT\n"
                 "  --ref FILE         count voxels (pipeline) or labels (ccl) that differ from FILE\n");
    std::exit(2);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (k == "--mode") a.mode = next();
        else if (k == "--impl") a.impl = next();
        else if (k == "-i") a.input = next();
        else if (k == "-o") a.output = next();
        else if (k == "--mask") a.mask = next();
        else if (k == "--truth") a.truth = next();
        else if (k == "--ref") a.ref = next();
        else if (k == "--threads") a.threads = std::atoi(next());
        else if (k == "--reps") a.reps = std::max(1, std::atoi(next()));
        else if (k == "--open") a.seg.open = true;
        else if (k == "--min-size") a.seg.min_size = std::atoi(next());
        else usage();
    }
    if (a.input.empty() || (a.mode == "pipeline" && a.mask.empty())) usage();
    if (a.impl != "seq" && a.impl != "omp" && a.impl != "cuda") usage();
    if (a.mode != "pipeline" && a.mode != "ccl") usage();
    if (a.truth.empty()) a.truth = a.mask;
    return a;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int run_ccl(const Args& a, int threads) {
    const VolumeU8 seg = read_volume_u8(a.input);
    std::vector<int32_t> L;
    std::vector<double> t;
    const bool gpu = a.impl == "cuda";
    for (int r = 0; r < a.reps + (gpu ? 1 : 0); ++r) {  // GPU: one untimed warm-up
        const double ms = gpu ? ccl_cuda(seg, L) : ccl_cpu(seg, L, a.impl == "omp");
        if (!(gpu && r == 0)) t.push_back(ms);
    }
    int64_t comps = 0;
    for (size_t i = 0; i < L.size(); ++i) comps += L[i] == int32_t(i);
    std::printf("{\"mode\":\"ccl\",\"impl\":\"%s\",\"threads\":%d,\"nx\":%d,\"ny\":%d,\"nz\":%d,\"reps\":%d,"
                "\"time_ms\":%.3f,\"components\":%lld",
                a.impl.c_str(), threads, seg.nx, seg.ny, seg.nz, a.reps, median(t), (long long)comps);
    if (!a.ref.empty()) {
        // Labels are canonical (smallest index), so any two correct runs agree exactly.
        std::vector<int32_t> R;
        ccl_cpu(seg, R, false);
        int64_t diff = 0;
        for (size_t i = 0; i < L.size(); ++i) diff += L[i] != R[i];
        std::printf(",\"label_mismatches_vs_seq\":%lld", (long long)diff);
    }
    std::printf("}\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    if (a.threads > 0) omp_set_num_threads(a.threads);
    const int threads = a.impl == "seq" ? 1 : (a.impl == "omp" ? omp_get_max_threads() : 0);
    if (a.mode == "ccl") return run_ccl(a, threads);

    const VolumeF img = read_volume_f32(a.input);
    VolumeU8 mask = read_volume_u8(a.mask);
    for (auto& m : mask.data) m = m > 0;
    const VolumeU8 truth = read_volume_u8(a.truth);
    if (mask.nx != img.nx || mask.ny != img.ny || mask.nz != img.nz) {
        std::fprintf(stderr, "mask and image sizes differ\n");
        return 1;
    }

    VolumeU8 seg;
    SegmentStats st;
    std::vector<double> cls, opn, ccl, cln, h2d, d2h, tot;
    const bool gpu = a.impl == "cuda";
    for (int r = 0; r < a.reps + (gpu ? 1 : 0); ++r) {
        st = SegmentStats{};
        if (gpu) segment_cuda(img, mask, seg, a.seg, st);
        else segment_cpu(img, mask, seg, a.seg, a.impl == "omp", st);
        if (gpu && r == 0) continue;
        cls.push_back(st.classify_ms);
        opn.push_back(st.open_ms);
        ccl.push_back(st.ccl_ms);
        cln.push_back(st.cleanup_ms);
        h2d.push_back(st.h2d_ms);
        d2h.push_back(st.d2h_ms);
        tot.push_back(st.total());
    }
    if (!a.output.empty()) write_volume_u8(a.output, seg);

    // Dice per tissue class against the ground-truth labels (1 CSF, 2 GM, 3 WM).
    double dice[4] = {0, 0, 0, 0};
    for (int c = 1; c <= 3; ++c) {
        int64_t inter = 0, ns = 0, nt = 0;
        for (size_t i = 0; i < seg.size(); ++i) {
            const bool s = seg.data[i] == c, t = truth.data[i] == c;
            inter += s && t;
            ns += s;
            nt += t;
        }
        dice[c] = ns + nt ? 2.0 * inter / double(ns + nt) : 1.0;
    }
    dice[0] = (dice[1] + dice[2] + dice[3]) / 3.0;

    std::printf("{\"mode\":\"pipeline\",\"impl\":\"%s\",\"threads\":%d,\"input\":\"%s\",\"nx\":%d,\"ny\":%d,\"nz\":%d,"
                "\"open\":%d,\"min_size\":%d,\"reps\":%d,\"time_ms\":%.3f,\"classify_ms\":%.3f,\"open_ms\":%.3f,"
                "\"ccl_ms\":%.3f,\"cleanup_ms\":%.3f,\"h2d_ms\":%.3f,\"d2h_ms\":%.3f,"
                "\"t0\":%.5f,\"t1\":%.5f,\"dice_csf\":%.4f,\"dice_gm\":%.4f,\"dice_wm\":%.4f,\"dice_mean\":%.4f,"
                "\"comps_before\":[%lld,%lld,%lld],\"comps_after\":[%lld,%lld,%lld]",
                a.impl.c_str(), threads, a.input.c_str(), img.nx, img.ny, img.nz, int(a.seg.open), a.seg.min_size,
                a.reps, median(tot), median(cls), median(opn), median(ccl), median(cln), median(h2d), median(d2h),
                st.thresholds[0], st.thresholds[1], dice[1], dice[2], dice[3], dice[0],
                (long long)st.comps_before[1], (long long)st.comps_before[2], (long long)st.comps_before[3],
                (long long)st.comps_after[1], (long long)st.comps_after[2], (long long)st.comps_after[3]);
    if (!a.ref.empty()) {
        const VolumeU8 ref = read_volume_u8(a.ref);
        int64_t diff = 0;
        for (size_t i = 0; i < seg.size(); ++i) diff += seg.data[i] != ref.data[i];
        std::printf(",\"voxel_mismatches_vs_ref\":%lld", (long long)diff);
    }
    std::printf("}\n");
    return 0;
}
