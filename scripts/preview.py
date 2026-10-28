"""Save a figure with axial/coronal/sagittal mid-slices of one or more volumes.

  python scripts/preview.py data/brainweb/clean.vol data/brainweb/noisy_5.vol \\
      --labels data/brainweb/labels.vol --out results/preview.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volio import read_vol


def mid_slices(v):
    nz, ny, nx = v.shape
    return [v[nz // 2], v[:, ny // 2, :], v[:, :, nx // 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volumes", nargs="+")
    ap.add_argument("--labels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [(os.path.basename(p), read_vol(p), "gray") for p in args.volumes]
    if args.labels:
        rows.append(("labels", read_vol(args.labels), "viridis"))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
    for r, (name, vol, cmap) in enumerate(rows):
        vmax = None if cmap != "gray" else 1.0
        for c, (sl, view) in enumerate(zip(mid_slices(vol), ("axial", "coronal", "sagittal"))):
            ax = axes[r, c]
            ax.imshow(np.flipud(sl) if c else sl, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(view)
        axes[r, 0].set_ylabel(name)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
