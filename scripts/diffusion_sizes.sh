#!/usr/bin/env bash
# Diffusion GPU variants across problem sizes, to show where temporal
# blocking pays off: small volumes fit in L2 and the naive kernel wins;
# once the volume spills to DRAM, fusing iterations saves bandwidth.
# Writes results/diffusion_sizes.csv.
#   scripts/diffusion_sizes.sh [REPS]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./build/denoise
PY=.venv/bin/python
REPS=${1:-5}
CSV=results/diffusion_sizes.csv
for s in 64 96 128 192 256; do
    [ -f data/synth$s/noisy_5.vol ] || $PY scripts/prep_data.py synthetic --size $s --out data/synth$s --noise 5 >/dev/null
done

k_ms() { sed -E 's/.*"kernel_ms":([0-9.]+).*/\1/'; }
echo "input,voxels,naive,smem,fused_K1,fused_K2,fused_K3" > $CSV
for d in synth64 synth96 synth128 synth192 brainweb synth256; do
    IN=data/$d/noisy_5.vol
    [ -f "$IN" ] || continue
    row="$d"
    vox=$($BIN --algo diffusion --impl cuda --reps 1 -i $IN | sed -E 's/.*"nx":([0-9]+),"ny":([0-9]+),"nz":([0-9]+).*/\1*\2*\3/')
    row="$row,$((vox))"
    row="$row,$($BIN --algo diffusion --impl cuda --reps $REPS -i $IN | k_ms)"
    row="$row,$($BIN --algo diffusion --impl cuda-smem --reps $REPS -i $IN | k_ms)"
    for k in 1 2 3; do
        row="$row,$($BIN --algo diffusion --impl cuda-fused --fuse $k --block 32,16,1 --reps $REPS -i $IN | k_ms)"
    done
    echo "$row" | tee -a $CSV
done
