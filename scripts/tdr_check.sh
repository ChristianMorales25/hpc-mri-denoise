#!/usr/bin/env bash
# Runs the heaviest NLM setting on a large volume with and without z-slab
# chunking, to confirm chunked launches stay under the WSL/Windows TDR limit.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=data/synth192
[ -f "$DATA/noisy_5.vol" ] || .venv/bin/python scripts/prep_data.py synthetic --size 192 --out "$DATA" --noise 5

for chunk in 16 8 0; do
    echo "--- chunk=$chunk"
    ./build/denoise --algo nlm --impl cuda --search 5 --patch 1 --reps 1 --chunk $chunk -i $DATA/noisy_5.vol || echo "FAILED (exit $?)"
done
