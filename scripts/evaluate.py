"""Quality metrics for denoised volumes.

  python scripts/evaluate.py --clean data/synth128/clean.vol \\
      --labels data/synth128/labels.vol results/out_*.vol

Reports PSNR and SSIM against the clean volume, over the whole volume and
inside the tissue mask (labels > 0). With --ref, also reports the maximum
absolute difference from a reference output, which is how the parallel
implementations are checked against the sequential baseline.
"""

import argparse
import json

import numpy as np
from skimage.metrics import structural_similarity

from volio import read_vol


def psnr(ref, img, mask=None):
    diff = (ref - img) if mask is None else (ref - img)[mask]
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volumes", nargs="+")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--ref", help="reference output to diff against")
    args = ap.parse_args()

    clean = read_vol(args.clean)
    mask = read_vol(args.labels) > 0 if args.labels else None
    ref = read_vol(args.ref) if args.ref else None

    for path in args.volumes:
        img = read_vol(path)
        row = {"file": path, "psnr": round(psnr(clean, img), 3)}
        ssim_map = structural_similarity(clean, img, data_range=1.0, full=True)[1]
        row["ssim"] = round(float(ssim_map.mean()), 4)
        if mask is not None:
            row["psnr_tissue"] = round(psnr(clean, img, mask), 3)
            row["ssim_tissue"] = round(float(ssim_map[mask].mean()), 4)
        if ref is not None:
            row["max_abs_diff"] = float(np.max(np.abs(img - ref)))
        print(json.dumps(row))


if __name__ == "__main__":
    main()
