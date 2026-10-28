"""Run the denoise binary over a grid of configurations and append to a CSV.

Strong scaling for NLM on one input, 1..12 threads:
  python scripts/bench.py --algo nlm --impl omp --threads 1 2 4 6 8 12 \\
      --inputs data/synth128/noisy_5.vol --csv results/strong_nlm.csv

CUDA block-shape sweep:
  python scripts/bench.py --algo bilateral --impl cuda \\
      --blocks 32,4,2 16,8,2 8,8,4 8,4,8 --inputs data/synth128/noisy_5.vol \\
      --csv results/blocks_bilateral.csv

Anything after "--" is passed through to the binary (filter parameters).
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]

    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--impl", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[0])
    ap.add_argument("--blocks", nargs="+", default=["8,8,4"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "denoise"))
    args = ap.parse_args(argv)

    rows = []
    for inp, th, blk in itertools.product(args.inputs, args.threads, args.blocks):
        cmd = [args.bin, "--algo", args.algo, "--impl", args.impl, "-i", inp,
               "--reps", str(args.reps), "--block", blk, *extra]
        if th > 0:
            cmd += ["--threads", str(th)]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        row = json.loads(out.strip().splitlines()[-1])
        row["input"] = inp
        row["params"] = " ".join(extra)
        rows.append(row)
        print(json.dumps(row), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
