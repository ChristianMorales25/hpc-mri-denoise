"""Prepare input volumes: a clean volume, tissue labels, and noisy copies.

Two sources:

  synthetic   Brain-like ellipsoid phantom at any size. No download needed;
              used for development and for weak-scaling runs where the
              problem size must grow with the thread count.

  brainweb    BrainWeb simulated T1 MRI (https://brainweb.bic.mni.mcgill.ca).
              Download the noise-free T1 volume (pn0, rf0) and the discrete
              "crisp" phantom as raw bytes (see README), then point this
              script at both files.

Noise is Rician, the distribution of magnitude MRI, with sigma given as a
percentage of the brightest tissue (the BrainWeb convention).

Output layout (in --out):
  clean.vol        float32 in [0, 1]
  labels.vol       uint8: 0 background/other, 1 CSF, 2 gray matter, 3 white matter
  noisy_<p>.vol    one per --noise level
"""

import argparse
import os

import numpy as np
from scipy import ndimage

from volio import write_vol

INTENSITY = {1: 0.25, 2: 0.55, 3: 0.85}  # CSF, GM, WM in a T1-like contrast
BRAINWEB_SHAPE = (181, 217, 181)  # (z, y, x) for the 1 mm volumes


def synthetic_phantom(nx, ny, nz, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, nz), np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij"
    )
    r = np.sqrt((x / 0.85) ** 2 + (y / 0.95) ** 2 + (z / 0.8) ** 2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.hypot(x, y))
    # Wavy boundaries stand in for cortical folds, so edges are not trivial.
    folds = 0.04 * np.sin(9 * theta) * np.cos(7 * phi) + 0.02 * np.sin(17 * theta + 3 * phi)

    labels = np.zeros((nz, ny, nx), np.uint8)
    labels[r < 0.95] = 1                       # CSF shell
    labels[r < 0.88 + folds] = 2               # gray matter
    labels[r < 0.62 + 1.5 * folds] = 3         # white matter
    # Ventricles: two CSF ellipsoids inside the white matter.
    for cx in (-0.18, 0.18):
        v = ((x - cx) / 0.1) ** 2 + ((y + 0.05) / 0.3) ** 2 + (z / 0.15) ** 2
        labels[v < 1] = 1
    # Small gray-matter nuclei at random positions in the white matter.
    for _ in range(6):
        c = rng.uniform(-0.35, 0.35, 3)
        s = rng.uniform(0.04, 0.08)
        b = ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) < s**2
        labels[b & (labels == 3)] = 2

    clean = np.zeros(labels.shape, np.float32)
    for k, val in INTENSITY.items():
        clean[labels == k] = val
    # Mild blur approximates partial-volume effects at tissue boundaries.
    clean = ndimage.gaussian_filter(clean, 0.6).astype(np.float32)
    return clean, labels


def load_brainweb(t1_path, crisp_path):
    t1 = np.fromfile(t1_path, np.uint8).reshape(BRAINWEB_SHAPE).astype(np.float32) / 255.0
    crisp = np.fromfile(crisp_path, np.uint8).reshape(BRAINWEB_SHAPE)
    labels = np.where(np.isin(crisp, (1, 2, 3)), crisp, 0).astype(np.uint8)
    return t1, labels


def rician(clean, percent, rng):
    sigma = percent / 100.0 * float(clean.max())
    n1 = rng.normal(0, sigma, clean.shape)
    n2 = rng.normal(0, sigma, clean.shape)
    return np.sqrt((clean + n1) ** 2 + n2**2).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--size", type=int, nargs="+", default=[128], help="N (cube) or NX NY NZ")
    b = sub.add_parser("brainweb")
    b.add_argument("--t1", required=True, help="raw byte T1 volume, pn0 rf0")
    b.add_argument("--crisp", required=True, help="raw byte discrete phantom")
    for p in (s, b):
        p.add_argument("--out", required=True)
        p.add_argument("--noise", type=float, nargs="+", default=[3, 5, 7, 9], help="percent levels")
        p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source == "synthetic":
        dims = args.size * 3 if len(args.size) == 1 else args.size
        if len(dims) != 3:
            ap.error("--size takes 1 or 3 values")
        clean, labels = synthetic_phantom(*dims, seed=args.seed)
    else:
        clean, labels = load_brainweb(args.t1, args.crisp)

    os.makedirs(args.out, exist_ok=True)
    write_vol(os.path.join(args.out, "clean.vol"), clean)
    write_vol(os.path.join(args.out, "labels.vol"), labels)
    rng = np.random.default_rng(args.seed)
    for pct in args.noise:
        name = f"noisy_{pct:g}.vol"
        write_vol(os.path.join(args.out, name), rician(clean, pct, rng))
    nz, ny, nx = clean.shape
    print(f"wrote {args.out}: {nx}x{ny}x{nz}, noise levels {args.noise}")


if __name__ == "__main__":
    main()
