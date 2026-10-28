#!/usr/bin/env bash
# Copy the two BrainWeb downloads into data/raw, decompress, check their sizes,
# and build data/brainweb with prep_data.py.
#   scripts/import_brainweb.sh DOWNLOAD_DIR    (e.g. /mnt/c/Users/<you>/Downloads)
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=${1:?usage: $0 DOWNLOAD_DIR containing the two .rawb.gz files}
RAW=data/raw
EXPECTED=$((181 * 217 * 181))
mkdir -p "$RAW"

for name in t1_icbm_normal_1mm_pn0_rf0 phantom_1.0mm_normal_crisp; do
    if [ ! -f "$RAW/$name.rawb" ]; then
        cp "$SRC/$name.rawb.gz" "$RAW/"
        gunzip -f "$RAW/$name.rawb.gz"
    fi
    size=$(stat -c %s "$RAW/$name.rawb")
    [ "$size" -eq "$EXPECTED" ] || { echo "$name.rawb: $size bytes, expected $EXPECTED" >&2; exit 1; }
    echo "$name.rawb OK ($size bytes)"
done

.venv/bin/python scripts/prep_data.py brainweb \
    --t1 "$RAW/t1_icbm_normal_1mm_pn0_rf0.rawb" \
    --crisp "$RAW/phantom_1.0mm_normal_crisp.rawb" \
    --out data/brainweb
