#!/usr/bin/env bash
# Reproduce every result from a fresh clone, after the BrainWeb files are
# imported (scripts/import_brainweb.sh) and the Python environment exists
# (see README). Takes about 1.5 hours on an RTX 4050 Laptop + Ryzen 5 7535HS,
# most of it sequential and low-thread-count NLM runs.
#
#   scripts/run_all.sh            # resumes: finished experiments are skipped
#   FORCE=1 scripts/run_all.sh    # rerun every experiment
set -euo pipefail
cd "$(dirname "$0")/.."

NVCC=${NVCC:-$(command -v nvcc || echo "$HOME/cuda/bin/nvcc")}
cmake -B build -DCMAKE_CUDA_COMPILER="$NVCC" >/dev/null
cmake --build build -j

echo "== correctness checks"
scripts/smoke_test.sh >/dev/null
scripts/check_variants.sh | tail -1
scripts/check_segment.sh

echo "== experiments (results/final)"
.venv/bin/python scripts/experiments.py ${FORCE:+--force}

echo "== figures (results/figures)"
.venv/bin/python scripts/make_figures.py
