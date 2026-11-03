#!/usr/bin/env bash
# Correctness of the optimized bilateral and diffusion GPU variants: every
# output is compared with the OpenMP output (which matches the sequential
# baseline exactly) over several block shapes, fusion depths, and z chunks,
# including ones that do not divide the volume or the iteration count.
#   scripts/check_variants.sh [INPUT]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
BIN=./build/denoise
IN=${1:-data/brainweb/noisy_5.vol}
OUT=results/check_variants
rm -rf $OUT && mkdir -p $OUT

$BIN --algo bilateral --impl omp --reps 1 -i $IN -o $OUT/bil_ref.vol >/dev/null
$BIN --algo bilateral --impl omp --reps 1 --radius 4 -i $IN -o $OUT/bil_ref_r4.vol >/dev/null
$BIN --algo diffusion --impl omp --reps 1 -i $IN -o $OUT/dif_ref.vol >/dev/null

run() {  # name, then denoise args
    local name=$1; shift
    $BIN "$@" --reps 1 -i $IN -o $OUT/$name.vol >/dev/null
}
for blk in 8,8,4 32,4,2 16,16,1 7,5,3; do
    b=${blk//,/x}
    run bil_cuda_$b --algo bilateral --impl cuda      --block $blk
    run bil_smem_$b --algo bilateral --impl cuda-smem --block $blk
    run dif_cuda_$b --algo diffusion --impl cuda      --block $blk
    run dif_smem_$b --algo diffusion --impl cuda-smem --block $blk
done
run bil_smem_r4 --algo bilateral --impl cuda-smem --radius 4

# Fused diffusion: 2D blocks, block X and Y must exceed 2*K.
for blk in 32,16,1 32,8,1 19,13,1; do
    b=${blk//,/x}
    for k in 1 2 3 4 6; do
        [ "${blk%%,*}" -gt $((2 * k)) ] && [ "$(cut -d, -f2 <<<"$blk")" -gt $((2 * k)) ] || continue
        for zc in 32 7 0; do
            run dif_fused_K${k}_z${zc}_$b --algo diffusion --impl cuda-fused --fuse $k --zchunk $zc --block $blk
        done
    done
done

$PY - "$OUT" <<'EOF'
import glob, os, sys
import numpy as np
sys.path.insert(0, "scripts")
from volio import read_vol
out = sys.argv[1]
worst = 0.0
for f in sorted(glob.glob(f"{out}/*.vol")):
    name = os.path.basename(f)[:-4]
    if "_ref" in name:
        continue
    ref = f"{out}/bil_ref_r4.vol" if name.endswith("_r4") else f"{out}/{name[:3]}_ref.vol"
    d = float(np.abs(read_vol(f) - read_vol(ref)).max())
    worst = max(worst, d)
    print(f"{name:28s} max |diff| vs omp = {d:.2e}")
print(f"worst: {worst:.2e}")
sys.exit(0 if worst < 1e-4 else 1)
EOF
