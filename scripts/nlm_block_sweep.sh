#!/usr/bin/env bash
# Block-shape sweep for the NLM GPU variants on one input.
#   scripts/nlm_block_sweep.sh [SEARCH] [PATCH] [INPUT]
set -euo pipefail
cd "$(dirname "$0")/.."

S=${1:-3}
P=${2:-1}
IN=${3:-data/brainweb/noisy_5.vol}
CSV=results/nlm_blocks_S${S}P${P}.csv
rm -f "$CSV"

BLOCKS="8,8,4 16,8,2 16,8,4 32,4,2 32,8,1 16,16,1 8,8,8 32,4,4"
for impl in cuda cuda-smem cuda-sep; do
    # shellcheck disable=SC2086
    .venv/bin/python scripts/bench.py --algo nlm --impl $impl --reps 3 --inputs "$IN" \
        --blocks $BLOCKS --csv "$CSV" -- --search "$S" --patch "$P" >/dev/null
done

.venv/bin/python - "$CSV" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
impls = list(dict.fromkeys(r["impl"] for r in rows))
blocks = list(dict.fromkeys(r["block"] for r in rows))
t = {(r["impl"], r["block"]): float(r["kernel_ms"]) for r in rows}
print(f"{'block':>9s} " + " ".join(f"{i:>10s}" for i in impls))
for b in blocks:
    print(f"{b:>9s} " + " ".join(f"{t[(i, b)]:10.1f}" for i in impls))
EOF
