// Command-line driver: runs one filter with one implementation and prints a
// single JSON line with timings so scripts/bench.py can collect results.
//
//   denoise --algo nlm --impl omp --threads 12 -i data/x.vol -o results/y.vol

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "filters.hpp"

namespace {

struct Args {
    std::string algo = "bilateral", impl = "seq", input, output;
    int threads = 0, reps = 3;
    LaunchConfig launch;
    BilateralParams bil;
    NlmParams nlm;
    DiffusionParams dif;
    bool info = false;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: denoise --algo {bilateral|nlm|diffusion} --impl IMPL -i IN [-o OUT]\n"
                 "  IMPL: seq | omp | cuda (naive) | cuda-smem, cuda-sep (nlm only)\n"
                 "  --threads N        OpenMP threads (default: all)\n"
                 "  --reps N           timed repetitions, median reported (default 3)\n"
                 "  --block X,Y,Z      CUDA block shape (default 8,8,4)\n"
                 "  --chunk N          z-slices per NLM launch, 0 = all (default 16; avoids TDR)\n"
                 "  bilateral: --radius R --sigma-s F --sigma-r F\n"
                 "  nlm:       --search S --patch P --h F\n"
                 "  diffusion: --iters N --kappa F --dt F\n"
                 "  --device-info      print GPU properties and exit\n");
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
        if (k == "--algo") a.algo = next();
        else if (k == "--impl") a.impl = next();
        else if (k == "-i") a.input = next();
        else if (k == "-o") a.output = next();
        else if (k == "--threads") a.threads = std::atoi(next());
        else if (k == "--reps") a.reps = std::max(1, std::atoi(next()));
        else if (k == "--block") {
            if (std::sscanf(next(), "%d,%d,%d", &a.launch.bx, &a.launch.by, &a.launch.bz) != 3) usage();
        }
        else if (k == "--chunk") a.launch.chunk_z = std::atoi(next());
        else if (k == "--radius") a.bil.radius = std::atoi(next());
        else if (k == "--sigma-s") a.bil.sigma_spatial = std::atof(next());
        else if (k == "--sigma-r") a.bil.sigma_range = std::atof(next());
        else if (k == "--search") a.nlm.search_radius = std::atoi(next());
        else if (k == "--patch") a.nlm.patch_radius = std::atoi(next());
        else if (k == "--h") a.nlm.h = std::atof(next());
        else if (k == "--iters") a.dif.iterations = std::atoi(next());
        else if (k == "--kappa") a.dif.kappa = std::atof(next());
        else if (k == "--dt") a.dif.dt = std::atof(next());
        else if (k == "--device-info") a.info = true;
        else usage();
    }
    if (!a.info && a.input.empty()) usage();
    return a;
}

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.info) {
        print_device_info();
        return 0;
    }
    if (a.threads > 0) omp_set_num_threads(a.threads);
    const int threads = a.impl == "seq" ? 1 : (a.impl == "omp" ? omp_get_max_threads() : 0);

    const VolumeF in = read_volume_f32(a.input);
    VolumeF out;

    std::vector<double> total, h2d, kernel, d2h;
    const bool gpu = a.impl.rfind("cuda", 0) == 0;
    if (gpu && a.impl != "cuda" && a.algo != "nlm") {
        std::fprintf(stderr, "%s is only implemented for nlm\n", a.impl.c_str());
        return 1;
    }
    // One untimed warm-up run on the GPU absorbs context creation and JIT.
    const int runs = a.reps + (gpu ? 1 : 0);
    for (int r = 0; r < runs; ++r) {
        if (gpu) {
            GpuTiming t;
            if (a.algo == "bilateral") t = bilateral_cuda_naive(in, out, a.bil, a.launch);
            else if (a.algo == "nlm") {
                const NlmVariant v = a.impl == "cuda-smem"  ? NlmVariant::Smem
                                     : a.impl == "cuda-sep" ? NlmVariant::Sep
                                                            : NlmVariant::Naive;
                if (a.impl != "cuda" && a.impl != "cuda-smem" && a.impl != "cuda-sep") usage();
                t = nlm_cuda(in, out, a.nlm, a.launch, v);
            }
            else if (a.algo == "diffusion") t = diffusion_cuda_naive(in, out, a.dif, a.launch);
            else usage();
            if (r == 0) continue;
            total.push_back(t.total());
            h2d.push_back(t.h2d_ms);
            kernel.push_back(t.kernel_ms);
            d2h.push_back(t.d2h_ms);
        } else {
            const bool par = a.impl == "omp";
            if (!par && a.impl != "seq") usage();
            const double t0 = now_ms();
            if (a.algo == "bilateral") bilateral_cpu(in, out, a.bil, par);
            else if (a.algo == "nlm") nlm_cpu(in, out, a.nlm, par);
            else if (a.algo == "diffusion") diffusion_cpu(in, out, a.dif, par);
            else usage();
            total.push_back(now_ms() - t0);
        }
    }

    if (!a.output.empty()) write_volume_f32(a.output, out);

    std::printf("{\"algo\":\"%s\",\"impl\":\"%s\",\"threads\":%d,\"nx\":%d,\"ny\":%d,\"nz\":%d,"
                "\"block\":\"%d,%d,%d\",\"reps\":%d,\"time_ms\":%.3f",
                a.algo.c_str(), a.impl.c_str(), threads, in.nx, in.ny, in.nz, a.launch.bx, a.launch.by,
                a.launch.bz, a.reps, median(total));
    if (gpu)
        std::printf(",\"h2d_ms\":%.3f,\"kernel_ms\":%.3f,\"d2h_ms\":%.3f", median(h2d), median(kernel),
                    median(d2h));
    std::printf("}\n");
    return 0;
}
