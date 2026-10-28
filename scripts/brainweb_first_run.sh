#!/usr/bin/env bash
# First pass on real data: sanity-check the BrainWeb volumes, then time every
# filter on noisy_5 (OpenMP and CUDA) and score the outputs.
# NLM sequential is skipped here; at S=3 it takes several minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
BIN=./build/denoise
D=data/brainweb
OUT=results/brainweb_first
mkdir -p $OUT

$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, "scripts")
from volio import read_vol
c, l = read_vol("data/brainweb/clean.vol"), read_vol("data/brainweb/labels.vol")
print("shape", c.shape, "range", float(c.min()), float(c.max()))
for k, name in enumerate(["other", "CSF", "GM", "WM"]):
    m = l == k
    print(f"label {k} {name:5s} voxels {int(m.sum()):8d}  mean intensity {float(c[m].mean()):.3f}")
EOF

$PY scripts/preview.py $D/clean.vol $D/noisy_5.vol --labels $D/labels.vol --out $OUT/preview.png

IN=$D/noisy_5.vol
for impl in seq omp cuda; do
    $BIN --algo bilateral --impl $impl --reps 1 -i $IN -o $OUT/bil_$impl.vol
    $BIN --algo diffusion --impl $impl --reps 1 -i $IN -o $OUT/dif_$impl.vol
done
for impl in omp cuda; do
    $BIN --algo nlm --impl $impl --reps 1 --search 3 -i $IN -o $OUT/nlm_$impl.vol
done

eval_args=(--clean $D/clean.vol --labels $D/labels.vol)
$PY scripts/evaluate.py "${eval_args[@]}" $IN
$PY scripts/evaluate.py "${eval_args[@]}" --ref $OUT/bil_seq.vol $OUT/bil_omp.vol $OUT/bil_cuda.vol
$PY scripts/evaluate.py "${eval_args[@]}" --ref $OUT/dif_seq.vol $OUT/dif_omp.vol $OUT/dif_cuda.vol
$PY scripts/evaluate.py "${eval_args[@]}" --ref $OUT/nlm_omp.vol $OUT/nlm_cuda.vol

$PY scripts/preview.py $D/clean.vol $IN $OUT/bil_cuda.vol $OUT/dif_cuda.vol $OUT/nlm_cuda.vol \
    --out $OUT/denoised.png
