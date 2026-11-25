#!/usr/bin/env bash
# Correctness of the segmentation stage: the OpenMP and CUDA pipelines must
# produce exactly the sequential segmentation (voxel for voxel), and their
# connected-component labels must equal the sequential labels exactly (the
# labels are canonical: smallest voxel index in each component).
#   scripts/check_segment.sh [INPUT] [MASK]
set -euo pipefail
cd "$(dirname "$0")/.."

SEG=./build/segment
IN=${1:-data/brainweb/noisy_5.vol}
MASK=${2:-data/brainweb/labels.vol}
OUT=results/check_segment
mkdir -p $OUT
fail=0

for opt in "" "--open" "--min-size 0" "--open --min-size 50"; do
    tag=$(tr -d ' -' <<<"${opt:-default}")
    # shellcheck disable=SC2086
    $SEG --impl seq --reps 1 -i $IN --mask $MASK $opt -o $OUT/seq_$tag.vol >/dev/null
    for impl in omp cuda; do
        # shellcheck disable=SC2086
        j=$($SEG --impl $impl --reps 1 -i $IN --mask $MASK $opt --ref $OUT/seq_$tag.vol)
        m=$(sed -E 's/.*"voxel_mismatches_vs_ref":([0-9]+).*/\1/' <<<"$j")
        printf "%-22s %-5s voxel mismatches vs seq = %s\n" "${opt:-default}" $impl "$m"
        [ "$m" = 0 ] || fail=1
    done
done

for impl in omp cuda; do
    j=$($SEG --mode ccl --impl $impl --reps 1 -i $OUT/seq_open.vol --ref seq)
    m=$(sed -E 's/.*"label_mismatches_vs_seq":([0-9]+).*/\1/' <<<"$j")
    printf "%-22s %-5s label mismatches vs seq = %s\n" "ccl" $impl "$m"
    [ "$m" = 0 ] || fail=1
done
exit $fail
