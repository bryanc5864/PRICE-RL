# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""Build the cross-method synthesis figure: PRICE-RL position vs every
baseline on every available benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _palette  # noqa: E402,F401


def load_summary(p, method_col, value_col):
    df = pd.read_csv(p)
    out = {}
    for m in df[method_col].unique():
        out[m] = df[df[method_col] == m][value_col].mean()
    return out


def main():
    benchmarks = []
    # GB1 series
    for label, p in [("GB1-500q", "experiments/E1_gb1/summary.csv"),
                      ("GB1-4000q", "experiments/E1L_gb1_long/summary.csv"),
                      ("GB1-8000q", "experiments/E1mega_gb1/summary.csv")]:
        f = ROOT / p
        if not f.exists():
            continue
        d = load_summary(f, "method", "top_1pct")
        benchmarks.append((label, d))
    # Trap-5
    f = ROOT / "experiments/E5_cross_domain/summary.csv"
    if f.exists():
        df = pd.read_csv(f)
        benchmarks.append(("Trap-5 (N=30)", {
            "random": df["best_random"].mean(),
            "adalead": df["best_adalead"].mean(),
            "price_rl": df["best_price_rl"].mean(),
            "price_rl_fixed": df["best_price_fixed"].mean(),
        }))
    # E1b multi-DMS (small budget, picks up 2 datasets)
    f = ROOT / "experiments/E1b_multi_dms/summary.csv"
    if f.exists():
        df = pd.read_csv(f)
        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds]
            d = {m: sub[sub["method"] == m]["top_1pct"].mean() for m in sub["method"].unique()}
            benchmarks.append((f"{ds[:8]}-500q", d))
    # E1-wt multi-DMS
    f = ROOT / "experiments/E1wt_multi_dms/summary.csv"
    if f.exists():
        df = pd.read_csv(f)
        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds]
            d = {m: sub[sub["method"] == m]["top_1pct"].mean() for m in sub["method"].unique()}
            benchmarks.append((f"{ds[:8]}-WT", d))
    # E16 mega multi-DMS
    f = ROOT / "experiments/E16_mega_multi_dms/summary.csv"
    if f.exists():
        df = pd.read_csv(f)
        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds]
            d = {m: sub[sub["method"] == m]["top_1pct"].mean() for m in sub["method"].unique()}
            benchmarks.append((f"{ds[:8]}-mega", d))
    # E8 AAV-GPU
    f = ROOT / "experiments/E8_aav_gpu/summary.csv"
    if f.exists():
        df = pd.read_csv(f)
        d = {m: df[df["method"] == m]["top_1pct"].mean() for m in df["method"].unique()}
        benchmarks.append(("AAV-GPU", d))

    print(f"\n{len(benchmarks)} benchmarks loaded.\n")
    print("Cross-method synthesis (PRICE-RL position vs each baseline):")
    print(f"{'benchmark':<22s} | {'method':<25s} | mean score")
    print("-" * 70)
    for label, methods in benchmarks:
        for m, v in sorted(methods.items(), key=lambda x: -x[1]):
            print(f"{label:<22s} | {m:<25s} | {v:.4f}")
        print()

    # Build figure: heatmap of (benchmark x method) — normalised per-row.
    # First collect all methods seen.
    all_methods = sorted({m for _, ms in benchmarks for m in ms.keys()})
    M = np.full((len(benchmarks), len(all_methods)), np.nan)
    for i, (lab, ms) in enumerate(benchmarks):
        row_max = max(ms.values()) if ms else 1.0
        for j, mn in enumerate(all_methods):
            if mn in ms:
                M[i, j] = ms[mn] / max(row_max, 1e-9)
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(benchmarks) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="blue_green_div", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_methods)))
    ax.set_xticklabels(all_methods, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels([b[0] for b in benchmarks], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="k" if M[i, j] > 0.5 else "w", fontsize=6)
    plt.colorbar(im, ax=ax, label="score / row max")
    ax.set_title("Cross-method synthesis: PRICE-RL position across all benchmarks")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_synthesis_heatmap.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_synthesis_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
