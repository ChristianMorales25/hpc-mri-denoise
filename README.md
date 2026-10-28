# GPU-Accelerated 3D MRI Denoising

Parallel 3D denoising of brain MRI (bilateral filter, non-local means,
Perona-Malik anisotropic diffusion) with sequential, OpenMP, and CUDA
implementations, plus a downstream tissue-segmentation stage (3D morphology
and GPU connected-component labeling) used to measure whether denoising
actually helps.

CIIC 5019 High-Performance Computing, University of Puerto Rico at Mayagüez.

## Status

| Stage | Sequential | OpenMP | CUDA naive | CUDA optimized |
|---|:-:|:-:|:-:|:-:|
| Bilateral filter | ✅ | ✅ | ✅ | ⬜ |
| Non-local means | ✅ | ✅ | ✅ | ✅ `cuda-smem`, `cuda-sep` |
| Anisotropic diffusion | ✅ | ✅ | ✅ | ⬜ |
| Segmentation + morphology | ⬜ | ⬜ | ⬜ | ⬜ |
| Connected-component labeling | ⬜ | ⬜ | ⬜ | ⬜ |

## Requirements

- Linux or WSL2 (developed on Ubuntu 24.04 under WSL2)
- CUDA toolkit 12.x, an NVIDIA GPU (developed on an RTX 4050 Laptop, sm_89)
- GCC with OpenMP, CMake >= 3.24
- Python 3.10+ for data preparation and evaluation

## Build

```bash
# If nvcc is not on PATH, point CMake at it:
cmake -B build -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc
cmake --build build -j
./build/denoise --device-info
```

For a different GPU, pass `-DCMAKE_CUDA_ARCHITECTURES=<sm>` (e.g. `86`).

## Python environment

```bash
python3 -m venv .venv          # if ensurepip is missing: --without-pip, then get-pip.py
.venv/bin/pip install -r requirements.txt
```

## Data

**Synthetic phantom** (no download; any size, used for development and weak scaling):

```bash
.venv/bin/python scripts/prep_data.py synthetic --size 128 --out data/synth128
```

**BrainWeb** simulated T1 MRI, 181×217×181 at 1 mm. From
<https://brainweb.bic.mni.mcgill.ca/brainweb/> download, as *raw byte* files:

1. Simulated brain database → *Normal brain* → modality **T1**, slice thickness
   **1 mm**, noise **0%**, intensity non-uniformity **0%** (the clean reference).
2. Anatomical model → *Normal brain* → **discrete (crisp)** phantom (tissue labels).

The crisp phantom is on the *Anatomical Models* page, not in the simulator
form; its direct link is
<https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?alias=phantom_1.0mm_normal_crisp&download=1>.
Choose *raw byte (unsigned)* and *gnuzip*. Then, with both `.rawb.gz` files in
one folder:

```bash
scripts/import_brainweb.sh /mnt/c/Users/<you>/Downloads
```

This copies them into `data/raw`, checks their sizes, and runs
`prep_data.py brainweb` to build `data/brainweb`.
`scripts/preview.py` renders mid-slices of any volumes for a quick visual check.

Both commands write `clean.vol`, `labels.vol`, and `noisy_<p>.vol` for Rician
noise levels 3, 5, 7, 9 % (change with `--noise`).

## Running

```bash
./build/denoise --algo nlm --impl omp --threads 12 --search 3 --patch 1 --h 0.05 \
    -i data/synth128/noisy_5.vol -o results/nlm.vol
```

`--impl` is `seq`, `omp`, or `cuda` (naive GPU) for every filter. NLM also has
`cuda-smem` (shared-memory tile) and `cuda-sep` (shared memory plus separable
patch distances); `scripts/nlm_variants.sh` and `scripts/nlm_block_sweep.sh`
compare them.

Each run prints one JSON line with the median time over `--reps` runs. CUDA
runs also report host-to-device, kernel, and device-to-host time separately.

Evaluate quality (PSNR/SSIM, whole volume and tissue only) and check a
parallel output against the sequential one:

```bash
.venv/bin/python scripts/evaluate.py --clean data/synth128/clean.vol \
    --labels data/synth128/labels.vol --ref results/nlm_seq.vol results/nlm_*.vol
```

Benchmark sweeps go through `scripts/bench.py` (see its docstring), and
`scripts/smoke_test.sh` runs every implementation on a small volume.

## File format

`.vol` files are a 16-byte header (`VOLF` or `VOLB` magic, then `int32 nx, ny, nz`)
followed by float32 or uint8 voxels, x fastest. See `include/volume.hpp` and
`scripts/volio.py`.

## Benchmarking notes

- Run plugged in and close other GPU-heavy applications; laptop clocks
  throttle under sustained load. Results report the median of several runs.
- On Windows/WSL2 the display driver kills kernels that run longer than
  about 2 s (TDR), so long kernels are split into chunked launches.
