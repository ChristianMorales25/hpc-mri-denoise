"""Run every performance and quality experiment. Each experiment writes one
CSV to results/final/. Experiments whose CSV already exists are skipped, so an
interrupted run can be resumed; name experiments explicitly (or use --force)
to rerun them.

  .venv/bin/python scripts/experiments.py              # everything missing
  .venv/bin/python scripts/experiments.py quality      # one experiment
  .venv/bin/python scripts/experiments.py --list

Then scripts/make_figures.py plots the CSVs into results/figures/.
Times are medians over repetitions; GPU runs exclude one warm-up run.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from volio import read_vol  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "results/final"
DEN = f"{OUT}/denoised"
BW = "data/brainweb"
BW_IN = f"{BW}/noisy_5.vol"
PY = sys.executable
FILTERS = ("bilateral", "diffusion", "nlm")
# Best GPU variant of each filter at its default parameters (docs/plan.md).
BEST = {"bilateral": "cuda-smem", "diffusion": "cuda-fused", "nlm": "cuda-smem"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_json(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def denoise(algo, impl, inp, reps=3, threads=None, out=None, extra=()):
    cmd = ["./build/denoise", "--algo", algo, "--impl", impl, "--reps", str(reps), "-i", inp, *map(str, extra)]
    if threads:
        cmd += ["--threads", str(threads)]
    if out:
        cmd += ["-o", out]
    return run_json(cmd)


def segment(impl, inp, mask, reps=3, threads=None, out=None, extra=()):
    cmd = ["./build/segment", "--impl", impl, "--reps", str(reps), "-i", inp, "--mask", mask, *map(str, extra)]
    if threads:
        cmd += ["--threads", str(threads)]
    if out:
        cmd += ["-o", out]
    return run_json(cmd)


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = f"{OUT}/{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    log(f"wrote {path} ({len(rows)} rows)")


def synthetic(size):
    d = f"data/synth{size}"
    if not os.path.exists(f"{d}/noisy_5.vol"):
        subprocess.run([PY, "scripts/prep_data.py", "synthetic", "--size", str(size), "--out", d, "--noise", "5"],
                       check=True, capture_output=True)
    return d


# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------

def exp_baseline():
    """Every implementation of every stage on BrainWeb at 5 % noise."""
    rows = []
    for algo in FILTERS:
        impls = ["seq", "omp", "cuda", BEST[algo]] + (["cuda-sep"] if algo == "nlm" else [])
        for impl in impls:
            reps = 1 if (algo == "nlm" and impl == "seq") else (3 if impl in ("seq", "omp") else 5)
            log(f"baseline {algo} {impl}")
            rows.append({"stage": algo, **denoise(algo, impl, BW_IN, reps)})
    seg_in = f"{DEN}/nlm_5.vol" if os.path.exists(f"{DEN}/nlm_5.vol") else BW_IN
    for impl in ("seq", "omp", "cuda"):
        log(f"baseline segment {impl}")
        rows.append({"stage": "segment", "algo": "segment",
                     **segment(impl, seg_in, f"{BW}/labels.vol", 5 if impl == "cuda" else 3)})
    write_csv("baseline", rows)


THREADS = (1, 2, 4, 6, 8, 10, 12)


def exp_strong():
    """OpenMP strong scaling on BrainWeb (fixed problem, 1..12 threads)."""
    rows = []
    for algo in FILTERS:
        for t in THREADS:
            log(f"strong {algo} {t} threads")
            rows.append({"stage": algo, **denoise(algo, "omp", BW_IN, 1 if algo == "nlm" else 3, threads=t)})
    for t in THREADS:
        log(f"strong segment {t} threads")
        rows.append({"stage": "segment", "algo": "segment",
                     **segment("omp", BW_IN, f"{BW}/labels.vol", 3, threads=t)})
    write_csv("strong", rows)


def exp_weak():
    """OpenMP weak scaling: synthetic cube whose voxel count grows with threads."""
    rows = []
    base = {"bilateral": 96, "diffusion": 96, "nlm": 64}
    for algo in FILTERS:
        for t in (1, 2, 4, 6, 8, 12):
            n = round(base[algo] * t ** (1 / 3))
            inp = f"{synthetic(n)}/noisy_5.vol"
            log(f"weak {algo} {t} threads, {n}^3")
            rows.append({"stage": algo, "base": base[algo], **denoise(algo, "omp", inp, 3, threads=t)})
    write_csv("weak", rows)


SIZES = (64, 96, 128, 192, 256)


def exp_size():
    """Problem size: synthetic cubes 64^3..256^3 plus BrainWeb, every implementation."""
    rows = []
    inputs = [(f"{synthetic(s)}/noisy_5.vol", f"data/synth{s}/labels.vol", s ** 3) for s in SIZES]
    inputs.append((BW_IN, f"{BW}/labels.vol", 181 * 217 * 181))
    for inp, labels, vox in inputs:
        big = vox > 140 ** 3
        for algo in FILTERS:
            for impl in ("seq", "omp", "cuda", BEST[algo]):
                if impl == "seq" and algo == "nlm" and big:
                    continue  # minutes per run; the strong-scaling 1-thread runs cover it
                reps = 1 if (big and impl in ("seq", "omp") and algo == "nlm") else (
                    1 if (big and impl == "seq") else (3 if impl in ("seq", "omp") else 5))
                log(f"size {inp} {algo} {impl}")
                rows.append({"stage": algo, "input": inp, "voxels": vox, **denoise(algo, impl, inp, reps)})
        for impl in ("seq", "omp", "cuda"):
            log(f"size {inp} segment {impl}")
            r = segment(impl, inp, labels, 5 if impl == "cuda" else 3)
            rows.append({"stage": "segment", "algo": "segment", "voxels": vox, **r})
    write_csv("size", rows)


def exp_nlm_variants():
    """NLM GPU optimization ladder over search and patch radii."""
    rows = []
    for s, p in ((2, 1), (3, 1), (5, 1), (2, 2), (3, 2), (4, 2)):
        for impl in ("cuda", "cuda-smem", "cuda-sep"):
            log(f"nlm S={s} P={p} {impl}")
            r = denoise("nlm", impl, BW_IN, 3, extra=("--search", s, "--patch", p))
            rows.append({"search": s, "patch": p, **r})
    write_csv("nlm_variants", rows)


def exp_gpu_config():
    """Block-shape and fusion sweeps (existing scripts), copied into results/final."""
    for cmd, src, dst in (
        (["scripts/nlm_block_sweep.sh", "3", "1"], "results/nlm_blocks_S3P1.csv", "nlm_blocks_S3P1"),
        (["scripts/gpu_ladder.sh"], "results/gpu_ladder.csv", "gpu_ladder"),
        (["scripts/diffusion_sizes.sh"], "results/diffusion_sizes.csv", "diffusion_sizes"),
    ):
        log(f"gpu_config {' '.join(cmd)}")
        subprocess.run(["bash", *cmd], check=True, capture_output=True)
        with open(src) as f:
            rows = list(csv.DictReader(f))
        write_csv(dst, rows)


def exp_ccl():
    """Connected-component labeling alone on a fragmented and a clean segmentation."""
    rows = []
    for name, inp in (("noisy_5", BW_IN), ("nlm_5", f"{DEN}/nlm_5.vol")):
        if not os.path.exists(inp):
            continue
        seg = f"{OUT}/seg_raw_{name}.vol"
        segment("cuda", inp, f"{BW}/labels.vol", 1, out=seg, extra=("--min-size", "0"))
        for impl in ("seq", "omp", "cuda"):
            log(f"ccl {name} {impl}")
            r = run_json(["./build/segment", "--mode", "ccl", "--impl", impl, "--reps", "5" if impl == "cuda" else "3",
                          "-i", seg, "--ref", "seq"])
            rows.append({"segmentation": name, **r})
    write_csv("ccl", rows)


# --------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------

_clean = _labels = _crop = None


def tissue_metrics(img):
    """PSNR and SSIM inside the tissue mask. SSIM is computed on the mask's
    bounding box plus a margin wider than the 7-voxel window, which gives the
    same values at mask voxels as the full volume at a fraction of the cost."""
    global _clean, _labels, _crop
    if _clean is None:
        _clean = read_vol(f"{BW}/clean.vol")
        _labels = read_vol(f"{BW}/labels.vol")
        idx = np.argwhere(_labels > 0)
        lo = np.maximum(idx.min(0) - 5, 0)
        hi = np.minimum(idx.max(0) + 6, _labels.shape)
        _crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    m = _labels > 0
    mse = float(np.mean((_clean[m].astype(np.float64) - img[m]) ** 2))
    ssim_map = structural_similarity(_clean[_crop], img[_crop], data_range=1.0, full=True)[1]
    return {"psnr_tissue": round(10 * np.log10(1 / mse), 3),
            "ssim_tissue": round(float(ssim_map[m[_crop]].mean()), 4)}


SWEEPS = {
    "bilateral": [("--sigma-r", v) for v in (0.05, 0.08, 0.1, 0.15, 0.2)],
    "nlm": [("--h", v) for v in (0.02, 0.03, 0.04, 0.05, 0.07, 0.1)],
    "diffusion": [(("--kappa", k), ("--iters", it)) for k in (0.02, 0.03, 0.05, 0.08) for it in (10, 20, 40)],
}
NOISE = (3, 5, 7, 9)


def exp_quality():
    """Parameter sweep per filter and noise level; keeps the best output (by
    tissue PSNR) of each filter at each noise level for the downstream study."""
    os.makedirs(DEN, exist_ok=True)
    tmp = f"{DEN}/_tmp.vol"
    rows = []
    for noise in NOISE:
        inp = f"{BW}/noisy_{noise}.vol"
        rows.append({"noise": noise, "filter": "noisy", "param": "", **tissue_metrics(read_vol(inp))})
        for algo in FILTERS:
            best = None
            for opt in SWEEPS[algo]:
                args = opt if isinstance(opt[0], tuple) else (opt,)
                extra = [str(x) for pair in args for x in pair]
                r = denoise(algo, BEST[algo], inp, 1, out=tmp, extra=extra)
                q = tissue_metrics(read_vol(tmp))
                param = " ".join(extra)
                log(f"quality {noise}% {algo} {param}: {q}")
                rows.append({"noise": noise, "filter": algo, "param": param, "kernel_ms": r["kernel_ms"], **q})
                if best is None or q["psnr_tissue"] > best:
                    best = q["psnr_tissue"]
                    os.replace(tmp, f"{DEN}/{algo}_{noise}.vol")
    if os.path.exists(tmp):
        os.remove(tmp)
    write_csv("quality", rows)


def exp_downstream():
    """Segmentation quality (Dice, fragments) for clean, noisy, and each
    filter's best output, at every noise level and cleanup setting."""
    rows = []
    cleanups = {"none": ("--min-size", "0"), "small10": ("--min-size", "10"), "open+small10": ("--open",)}
    for noise in NOISE:
        inputs = {"clean": f"{BW}/clean.vol", "noisy": f"{BW}/noisy_{noise}.vol"}
        inputs.update({a: f"{DEN}/{a}_{noise}.vol" for a in FILTERS})
        for name, inp in inputs.items():
            for cname, extra in cleanups.items():
                out = f"{DEN}/seg_{name}_{noise}.vol" if cname == "small10" else None
                r = segment("cuda", inp, f"{BW}/labels.vol", 1, out=out, extra=extra)
                log(f"downstream {noise}% {name} {cname}: dice {r['dice_mean']}")
                rows.append({"noise": noise, **r, "file": r["input"], "input": name, "cleanup": cname})
    write_csv("downstream", rows)


EXPERIMENTS = {
    "quality": exp_quality,          # first: produces the denoised volumes the others use
    "downstream": exp_downstream,
    "baseline": exp_baseline,
    "ccl": exp_ccl,
    "nlm_variants": exp_nlm_variants,
    "gpu_config": exp_gpu_config,
    "size": exp_size,
    "weak": exp_weak,
    "strong": exp_strong,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="experiments to run (default: all missing)")
    ap.add_argument("--force", action="store_true", help="rerun even if the CSV exists")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, f in EXPERIMENTS.items():
            print(f"{k:14s} {f.__doc__.splitlines()[0]}")
        return
    os.makedirs(OUT, exist_ok=True)
    names = args.names or list(EXPERIMENTS)
    for n in names:
        # gpu_config writes several CSVs; diffusion_sizes is its last one.
        marker = "diffusion_sizes" if n == "gpu_config" else n
        done = os.path.exists(f"{OUT}/{marker}.csv")
        if done and not (args.force or args.names):
            log(f"skip {n} (CSV exists)")
            continue
        log(f"=== {n}")
        t = time.time()
        EXPERIMENTS[n]()
        log(f"=== {n} done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
