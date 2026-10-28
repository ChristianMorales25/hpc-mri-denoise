#!/usr/bin/env bash
# Compare the NLM GPU variants (naive, smem, sep) for correctness against a
# reference output and for speed, over a few search/patch radii.
#   scripts/nlm_variants.sh [INPUT] [REFERENCE_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
BIN=./build/denoise
IN=${1:-data/brainweb/noisy_5.vol}
OUT=results/nlm_variants
mkdir -p $OUT

for sp in "3 1" "3 2" "5 1"; do
    read -r S P <<<"$sp"
    echo "=== search=$S patch=$P"
    for impl in cuda cuda-smem cuda-sep; do
        $BIN --algo nlm --impl $impl --search $S --patch $P --reps 3 -i $IN -o $OUT/nlm_S${S}P${P}_$impl.vol
    done
    $PY - "$OUT" "$S" "$P" <<'EOF'
import sys, numpy as np
sys.path.insert(0, "scripts")
from volio import read_vol
out, S, P = sys.argv[1:]
ref = read_vol(f"{out}/nlm_S{S}P{P}_cuda.vol")
for v in ("cuda-smem", "cuda-sep"):
    d = np.abs(read_vol(f"{out}/nlm_S{S}P{P}_{v}.vol") - ref).max()
    print(f"  {v:10s} max |diff| vs naive = {d:.2e}")
EOF
done
