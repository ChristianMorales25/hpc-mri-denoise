"""Plot the experiment results in results/final/*.csv as PNG figures in
results/figures/.

  .venv/bin/python scripts/make_figures.py
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from volio import read_vol  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
IN = "results/final"
OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

# Validated categorical palette (light mode), assigned in fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W, FULL_W = 3.45, 7.0  # narrow and wide figure widths, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

STAGES = ["bilateral", "diffusion", "nlm", "segment"]
STAGE_NAME = {"bilateral": "Bilateral", "diffusion": "Diffusion", "nlm": "Non-local means", "segment": "Segmentation"}
IMPL_NAME = {"seq": "Sequential", "omp": "OpenMP (12 thr.)", "cuda": "CUDA naive", "best": "CUDA optimized"}
BEST = {"bilateral": "cuda-smem", "diffusion": "cuda-fused", "nlm": "cuda-smem", "segment": "cuda"}

def load(name):
    path = f"{IN}/{name}.csv"
    if not os.path.exists(path):
        print(f"missing {path}, skipping")
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) and k not in ("impl", "algo", "stage", "input", "block",
                                                                      "param", "filter", "cleanup", "mode") else v
            except ValueError:
                pass
    return rows


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.png")


def label_end(ax, x, y, text, color=INK2, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=6.5, color=color)


# --------------------------------------------------------------------------
def fig_baseline():
    rows = load("baseline")
    if not rows:
        return
    t = {}
    for r in rows:
        st = r["stage"]
        impl = r["impl"]
        key = "best" if (impl == BEST[st] and impl != "cuda") or (st == "segment" and impl == "cuda") else impl
        if st == "segment" and impl == "cuda":
            t[(st, "cuda")] = r
        t[(st, key)] = r
        if st == "nlm" and impl == "cuda-sep":
            t[(st, "sep")] = r
    table = []
    for st in STAGES:
        seq = t[(st, "seq")]["time_ms"]
        row = {"stage": st, "seq_ms": seq}
        for k in ("omp", "cuda", "best"):
            if (st, k) in t:
                r = t[(st, k)]
                row[f"{k}_ms"] = r["time_ms"]
                row[f"{k}_speedup"] = seq / r["time_ms"]
                if k != "omp":
                    row[f"{k}_kernel_ms"] = r.get("kernel_ms") or (r["time_ms"] - r["h2d_ms"] - r["d2h_ms"])
                    row[f"{k}_h2d_ms"] = r["h2d_ms"]
                    row[f"{k}_d2h_ms"] = r["d2h_ms"]
                    row[f"{k}_impl"] = r["impl"]
        if (st, "sep") in t:
            row["sep_ms"] = t[(st, "sep")]["time_ms"]
        table.append(row)

    # Speedup over sequential, log scale, one group per stage.
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    keys = ["omp", "cuda", "best"]
    n = len(STAGES)
    w = 0.26
    for j, k in enumerate(keys):
        xs, ys = [], []
        for i, row in enumerate(table):
            if f"{k}_speedup" in row and not (row["stage"] == "segment" and k == "best"):
                xs.append(i + (j - 1) * w)
                ys.append(row[f"{k}_speedup"])
        bars = ax.bar(xs, ys, w * 0.92, color=C[j + 1], label=IMPL_NAME[k], edgecolor="white", linewidth=0.5)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:,.0f}×" if y >= 10 else f"{y:.1f}×", (b.get_x() + b.get_width() / 2, y),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=5.5,
                        color=INK2, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e4)
    ax.set_xticks(range(n), [STAGE_NAME[s].replace("Non-local means", "NLM") for s in STAGES])
    ax.set_ylabel("Speedup over sequential")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    save(fig, "fig_baseline_speedup")

    # Where GPU time goes: transfers vs kernel for the optimized versions.
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    parts = [("best_h2d_ms", "Host→device"), ("best_kernel_ms", "Kernels"), ("best_d2h_ms", "Device→host")]
    names = [STAGE_NAME[r["stage"]].replace("Non-local means", "NLM") for r in table]
    left = np.zeros(len(table))
    for j, (k, lab) in enumerate(parts):
        tot = np.array([r["best_h2d_ms"] + r["best_kernel_ms"] + r["best_d2h_ms"] for r in table])
        frac = np.array([r[k] for r in table]) / tot * 100
        ax.barh(names, frac, left=left, color=C[j], label=lab, edgecolor="white", linewidth=1.0, height=0.6)
        left += frac
    for i, r in enumerate(table):
        tot = r["best_h2d_ms"] + r["best_kernel_ms"] + r["best_d2h_ms"]
        ax.annotate(f"{tot:.0f} ms" if tot >= 10 else f"{tot:.1f} ms", (100, i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.5, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of GPU time (%)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.3))
    save(fig, "fig_transfers")


# --------------------------------------------------------------------------
def fig_strong():
    rows = load("strong")
    if not rows:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W * 0.62, 2.1))
    for i, st in enumerate(STAGES):
        rs = sorted([r for r in rows if r["stage"] == st], key=lambda r: r["threads"])
        if not rs:
            continue
        p = np.array([r["threads"] for r in rs])
        tm = np.array([r["time_ms"] for r in rs])
        s = tm[0] / tm
        a1.plot(p, s, marker="o", color=C[i], label=STAGE_NAME[st])
        a2.plot(p, s / p * 100, marker="o", color=C[i], label=STAGE_NAME[st])
    for a in (a1, a2):
        a.axvline(6, color=INK2, lw=0.8, ls=":")
        a.set_xlabel("OpenMP threads")
        a.set_xticks([1, 2, 4, 6, 8, 10, 12])
    a1.plot([1, 12], [1, 12], color=INK2, lw=0.8, ls="--")
    label_end(a1, 12, 12, "ideal", dx=-14, dy=-6)
    a1.annotate("6 physical cores", (6, 0.5), xytext=(3, 0), textcoords="offset points", fontsize=6, color=INK2)
    a1.set_ylabel("Speedup")
    a1.set_ylim(0, 12.5)
    a2.set_ylabel("Parallel efficiency (%)")
    a2.set_ylim(0, 110)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h[:4], l[:4], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_strong")


def fig_weak():
    rows = load("weak")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for i, st in enumerate(STAGES[:3]):
        rs = sorted([r for r in rows if r["stage"] == st], key=lambda r: r["threads"])
        p = np.array([r["threads"] for r in rs])
        vox = np.array([r["nx"] * r["ny"] * r["nz"] for r in rs])
        tm = np.array([r["time_ms"] for r in rs])
        eff = (vox / tm) / (p * vox[0] / tm[0]) * 100  # per-thread throughput relative to 1 thread
        ax.plot(p, eff, marker="o", color=C[i], label=STAGE_NAME[st])
    ax.axvline(6, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("OpenMP threads (volume grows with threads)")
    ax.set_ylabel("Weak-scaling efficiency (%)")
    ax.set_xticks([1, 2, 4, 6, 8, 12])
    ax.set_ylim(0, 110)
    ax.legend(loc="lower left")
    save(fig, "fig_weak")


def fig_size():
    rows = load("size")
    if not rows:
        return
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.95), sharey=False)
    for a, st in zip(axes, STAGES):
        rs = [r for r in rows if r["stage"] == st]
        impls = ["seq", "omp", "cuda", BEST[st]] if st != "segment" else ["seq", "omp", "cuda"]
        for j, impl in enumerate(impls):
            pts = sorted([(r["voxels"], r["time_ms"]) for r in rs if r["impl"] == impl and "brainweb" not in r.get(
                "input", "") and abs(round(r["voxels"] ** (1 / 3)) ** 3 - r["voxels"]) < 1])
            if not pts:
                continue
            v = np.array([p[0] for p in pts])
            tm = np.array([p[1] for p in pts])
            key = "best" if (impl == BEST[st] and impl != "cuda") else impl
            color = C[["seq", "omp", "cuda", "best"].index(key)]
            a.plot(v / 1e6, v / tm / 1e3, marker="o", color=color, label=IMPL_NAME[key].replace(" (12 thr.)", ""))
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_title(STAGE_NAME[st])
        a.set_xlabel("Megavoxels")
    axes[0].set_ylabel("Throughput (Mvoxel/s)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_size")


def fig_nlm_ladder():
    rows = load("nlm_variants")
    if not rows:
        return
    combos = list(dict.fromkeys((int(r["search"]), int(r["patch"])) for r in rows))
    impls = [("cuda", "Naive"), ("cuda-smem", "Shared-memory tile"), ("cuda-sep", "Tile + separable distance")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    w = 0.26
    for j, (impl, lab) in enumerate(impls):
        ys = [next(r["kernel_ms"] for r in rows if (int(r["search"]), int(r["patch"])) == c and r["impl"] == impl)
              for c in combos]
        ax.bar(np.arange(len(combos)) + (j - 1) * w, ys, w * 0.92, color=C[[2, 3, 0][j]], label=lab,
               edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(combos)), [f"S={s}\nP={p}" for s, p in combos])
    ax.set_ylabel("Kernel time (ms)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=1)
    save(fig, "fig_nlm_ladder")


def fig_diffusion_sizes():
    rows = load("diffusion_sizes")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["voxels"])
    v = np.array([r["voxels"] for r in rows]) / 1e6
    naive = np.array([r["naive"] for r in rows])
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    for j, (k, lab) in enumerate((("smem", "3D shared tile"), ("fused_K1", "2.5D, K=1"), ("fused_K2", "2.5D, K=2"),
                                  ("fused_K3", "2.5D, K=3"))):
        s = naive / np.array([r[k] for r in rows])
        ax.plot(v, s, marker="o", color=C[[1, 2, 3, 0][j]], label=lab)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.axvspan(0, 3.0, color=GRID, alpha=0.5, lw=0)
    ax.annotate("both buffers fit in L2", (0.27, 1.40), fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Megavoxels")
    ax.set_ylabel("Speedup over naive")
    ax.set_ylim(0.4, 1.5)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), columnspacing=1.0, handlelength=1.5)
    save(fig, "fig_diffusion_sizes")


def fig_quality():
    q = load("quality")
    d = load("downstream")
    if not q or not d:
        return
    noise = sorted({int(r["noise"]) for r in q})
    series = ["noisy", "bilateral", "diffusion", "nlm"]
    names = {"noisy": "Noisy input", "bilateral": "Bilateral", "diffusion": "Diffusion", "nlm": "Non-local means",
             "clean": "Noise-free"}
    best = {}
    for r in q:
        k = (r["filter"], int(r["noise"]))
        if k not in best or r["psnr_tissue"] > best[k]["psnr_tissue"]:
            best[k] = r
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W * 0.62, 2.0))
    for j, s in enumerate(series):
        color = INK2 if s == "noisy" else C[j - 1]
        ls = "--" if s == "noisy" else "-"
        a1.plot(noise, [best[(s, n)]["psnr_tissue"] for n in noise], marker="o", color=color, ls=ls, label=names[s])
        a2.plot(noise, [next(r["dice_mean"] for r in d if r["input"] == s and int(r["noise"]) == n
                             and r["cleanup"] == "small10") for n in noise], marker="o", color=color, ls=ls)
    clean_dice = next(r["dice_mean"] for r in d if r["input"] == "clean" and r["cleanup"] == "small10")
    a2.axhline(clean_dice, color=INK2, lw=0.8, ls=":")
    a2.annotate("noise-free input", (noise[-1], clean_dice), xytext=(0, -7), textcoords="offset points", fontsize=6,
                color=INK2, ha="right")
    a1.set_ylabel("Tissue PSNR (dB)")
    a2.set_ylabel("Mean Dice (CSF, GM, WM)")
    for a in (a1, a2):
        a.set_xlabel("Rician noise (% of max. intensity)")
        a.set_xticks(noise)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_quality")


def fig_slices():
    paths = {"clean": "data/brainweb/clean.vol", "noisy": "data/brainweb/noisy_5.vol",
             "nlm": "results/final/denoised/nlm_5.vol", "seg_noisy": "results/final/denoised/seg_noisy_5.vol",
             "seg_nlm": "results/final/denoised/seg_nlm_5.vol", "truth": "data/brainweb/labels.vol"}
    if not all(os.path.exists(p) for p in paths.values()):
        print("missing volumes for slices, skipping")
        return
    v = {k: read_vol(p) for k, p in paths.items()}
    z = v["clean"].shape[0] // 2 + 10
    # Crop to the head.
    m = v["truth"][z] > 0
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min() - 4, ys.max() + 5, xs.min() - 4, xs.max() + 5
    from matplotlib.colors import ListedColormap
    seg_cmap = ListedColormap(["#000000", "#4a90d9", "#9a9a9a", "#f2f2f2"])
    panels = [("clean", "(a) Noise-free", "gray"), ("noisy", "(b) 5 % Rician noise", "gray"),
              ("nlm", "(c) Non-local means", "gray"), ("seg_noisy", "(d) Segmentation of (b)", seg_cmap),
              ("seg_nlm", "(e) Segmentation of (c)", seg_cmap), ("truth", "(f) Ground truth", seg_cmap)]
    fig, axes = plt.subplots(1, 6, figsize=(FULL_W, 1.55))
    for a, (k, title, cmap) in zip(axes, panels):
        img = v[k][z, y0:y1, x0:x1]
        if k == "truth":
            img = np.where(v["truth"][z, y0:y1, x0:x1] > 0, v["truth"][z, y0:y1, x0:x1], 0)
        a.imshow(np.flipud(img), cmap=cmap, vmin=0, vmax=1 if cmap == "gray" else 3, interpolation="nearest")
        a.set_title(title, fontsize=6.5)
        a.axis("off")
    fig.tight_layout(w_pad=0.2)
    save(fig, "fig_slices")



if __name__ == "__main__":
    for f in (fig_baseline, fig_strong, fig_weak, fig_size, fig_nlm_ladder, fig_diffusion_sizes, fig_quality,
              fig_slices):
        try:
            f()
        except Exception as e:  # keep going so one missing CSV does not block the rest
            print(f"{f.__name__} failed: {type(e).__name__}: {e}")
            raise
