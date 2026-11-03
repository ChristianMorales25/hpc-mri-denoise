#!/usr/bin/env bash
# Quick end-to-end check: build output exists, every implementation runs on a
# small synthetic volume, and parallel outputs match the sequential baseline.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
BIN=./build/denoise
DATA=data/synth96
OUT=results/smoke
IN=$DATA/noisy_5.vol

[ -f "$DATA/clean.vol" ] || $PY scripts/prep_data.py synthetic --size 96 --out "$DATA" --noise 5
mkdir -p "$OUT"

for impl in seq omp cuda; do
    $BIN --algo bilateral --impl $impl --reps 1 -i $IN -o $OUT/bil_$impl.vol
    $BIN --algo diffusion --impl $impl --reps 1 -i $IN -o $OUT/dif_$impl.vol
    $BIN --algo nlm --impl $impl --reps 1 --search 2 -i $IN -o $OUT/nlm_$impl.vol
done
$BIN --algo bilateral --impl cuda-smem  --reps 1 -i $IN -o $OUT/bil_cuda-smem.vol
$BIN --algo diffusion --impl cuda-smem  --reps 1 -i $IN -o $OUT/dif_cuda-smem.vol
$BIN --algo diffusion --impl cuda-fused --reps 1 -i $IN -o $OUT/dif_cuda-fused.vol
$BIN --algo nlm       --impl cuda-smem  --reps 1 --search 2 -i $IN -o $OUT/nlm_cuda-smem.vol
$BIN --algo nlm       --impl cuda-sep   --reps 1 --search 2 -i $IN -o $OUT/nlm_cuda-sep.vol

eval_args=(--clean $DATA/clean.vol --labels $DATA/labels.vol)
echo "--- noisy input"
$PY scripts/evaluate.py "${eval_args[@]}" $IN
for f in bil dif nlm; do
    echo "--- $f (max_abs_diff vs seq)"
    $PY scripts/evaluate.py "${eval_args[@]}" --ref $OUT/${f}_seq.vol $OUT/${f}_*.vol
done
