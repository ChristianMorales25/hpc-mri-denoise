#!/usr/bin/env bash
# Kernel-time ladder for the bilateral and diffusion GPU variants on one input
# (median of --reps), plus a fusion-depth and block-shape sweep for diffusion.
# Results go to results/gpu_ladder.csv and a summary is printed.
#   scripts/gpu_ladder.sh [INPUT] [REPS]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./build/denoise
IN=${1:-data/brainweb/noisy_5.vol}
REPS=${2:-5}
CSV=results/gpu_ladder.csv
echo "algo,impl,param,block,kernel_ms,total_ms" > $CSV

row() {  # algo impl param block, then extra args
    local algo=$1 impl=$2 param=$3 block=$4; shift 4
    local j k t b=${block//,/x}   # no commas inside CSV fields
    if ! j=$($BIN --algo "$algo" --impl "$impl" --block "$block" --reps "$REPS" -i "$IN" "$@" 2>/dev/null); then
        echo "$algo,$impl,$param,$b,," >> $CSV   # e.g. the tile exceeds shared memory
        return
    fi
    k=$(sed -E 's/.*"kernel_ms":([0-9.]+).*/\1/' <<<"$j")
    t=$(sed -E 's/.*"time_ms":([0-9.]+).*/\1/' <<<"$j")
    echo "$algo,$impl,$param,$b,$k,$t" >> $CSV
}

# 3D blocks for the one-thread-per-voxel kernels.
for blk in 8,8,4 16,8,2 32,4,2 32,8,1 16,16,1; do
    for r in 2 3; do
        row bilateral cuda      r=$r $blk --radius $r
        row bilateral cuda-smem r=$r $blk --radius $r
    done
    row diffusion cuda      it=20 $blk
    row diffusion cuda-smem it=20 $blk
done
# 2D blocks for fused diffusion (one thread per x-y column, z-marching).
for blk in 32,8,1 32,16,1 64,8,1 32,32,1 64,16,1; do
    for k in 1 2 3 4 6 8; do
        row diffusion cuda-fused K=$k $blk --fuse $k
    done
done
# z-chunk length at a fixed shape and depth.
for zc in 8 16 32 64 0; do
    row diffusion cuda-fused "K=4 zc=$zc" 32,16,1 --fuse 4 --zchunk $zc
done

.venv/bin/python - $CSV <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
keys = list(dict.fromkeys((r["algo"], r["impl"], r["param"]) for r in rows))
blocks = list(dict.fromkeys(r["block"] for r in rows))
t = {(r["algo"], r["impl"], r["param"], r["block"]): (float(r["kernel_ms"]) if r["kernel_ms"] else None)
     for r in rows}
fmt = lambda v: f"{v:10.2f}" if v is not None else f"{'-':>10s}"
print(f"{'kernel ms':32s}" + "".join(f"{b:>10s}" for b in blocks))
for k in keys:
    print(f"{' '.join(k):32s}" + "".join(fmt(t.get(k + (b,))) for b in blocks))
EOF
