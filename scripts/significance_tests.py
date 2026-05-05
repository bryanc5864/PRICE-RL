# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""Mann-Whitney U tests + per-seed scatter plots over all benchmarks.
Produces a single LaTeX-ready table + an "all-benchmarks" scatter figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _palette  # noqa: E402,F401  -- unify colour scheme to blue/green
OUT = ROOT / "figures"


def save_fig(fig, stem: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def utest(a, b, alt="greater"):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    u, p = mannwhitneyu(a, b, alternative=alt)
    return float(u), float(p)


def collect():
    """Return a long-format DataFrame: dataset, method, seed, score."""
    frames = []
    paths = {
        "GB1_500": (ROOT / "experiments/E1_gb1/summary.csv", "method", "top_1pct"),
        "GB1_4000": (ROOT / "experiments/E1L_gb1_long/summary.csv", "method", "top_1pct"),
        "GB1_8000": (ROOT / "experiments/E1mega_gb1/summary.csv", "method", "top_1pct"),
        "TEM-1_5r": (ROOT / "experiments/E1b_multi_dms/summary.csv", "method", "top_1pct"),
        "Trap5_30": (ROOT / "experiments/E5_cross_domain/summary.csv", None, None),
        "Trap-K_scale": (ROOT / "experiments/E5L_trap_scaling/summary.csv", None, None),
        "NK_K10_HP": (ROOT / "experiments/E6_hp_robustness/summary.csv", None, None),
    }
    for name, (p, mc, sc) in paths.items():
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["benchmark"] = name
        frames.append(df)
    return frames


def main():
    print("\n=== Mann-Whitney U-tests (PRICE-RL beats baseline) ===")
    # E1, E1-L, E1-mega
    for label, p in [("GB1 500-q", "experiments/E1_gb1/summary.csv"),
                     ("GB1 4000-q", "experiments/E1L_gb1_long/summary.csv"),
                     ("GB1 8000-q", "experiments/E1mega_gb1/summary.csv")]:
        f = ROOT / p
        if not f.exists(): continue
        df = pd.read_csv(f)
        if "method" not in df.columns: continue
        pr = df[df.method == "price_rl"]["top_1pct"].values
        for opp in ["random", "pex", "adalead", "gflownet_al"]:
            if opp in df["method"].unique():
                op = df[df.method == opp]["top_1pct"].values
                _, p_g = utest(pr, op, "greater")
                _, p_t = utest(pr, op, "two-sided")
                cmp = "PRICE>"+opp
                print(f"  [{label}] {cmp:>20s}  one-sided p={p_g:.4f}  two-sided p={p_t:.4f}")

    # WT-init
    fwt = ROOT / "experiments/E1wt_multi_dms/summary.csv"
    if fwt.exists():
        print("\n=== WT-aware vs uniform PRICE-RL (one-sided U-test, WT > uniform) ===")
        df = pd.read_csv(fwt)
        for ds in df["dataset"].unique():
            d = df[df.dataset == ds]
            wt = d[d.method == "price_rl_wt"]["top_1pct"].values
            un = d[d.method == "price_rl_uniform"]["top_1pct"].values
            _, p_g = utest(wt, un, "greater")
            print(f"  {ds:18s} wt={wt.mean():.3f} uniform={un.mean():.3f}  one-sided p={p_g:.4f}")

    # Trap-5
    ft5 = ROOT / "experiments/E5_cross_domain/summary.csv"
    if ft5.exists():
        df = pd.read_csv(ft5)
        pr = df["best_price_rl"].values
        for col in ["best_random", "best_adalead"]:
            _, p_g = utest(pr, df[col].values, "greater")
            print(f"  [Trap5_30]  PRICE>{col[5:]:14s}  one-sided p={p_g:.4f}")

    # E5-L scaling — paired Wilcoxon across (N, K) configs
    f5l = ROOT / "experiments/E5L_trap_scaling/summary.csv"
    if f5l.exists():
        df = pd.read_csv(f5l)
        agg = df.groupby(["N", "K"])[["best_random", "best_adalead", "best_price_rl"]].mean().reset_index()
        # Wilcoxon paired across (N, K)
        try:
            w_a = wilcoxon(agg["best_price_rl"], agg["best_adalead"], alternative="greater")
            print(f"\n=== Trap-K scaling — paired Wilcoxon ===")
            print(f"  PRICE > AdaLead across all (N, K): p={w_a.pvalue:.4f}")
        except Exception as e:
            print(f"  Wilcoxon failed: {e}")

    # All-benchmarks per-seed scatter
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    pairs = [
        ("E1_gb1/summary.csv", "GB1 500-q", "method", "top_1pct"),
        ("E1L_gb1_long/summary.csv", "GB1 4000-q", "method", "top_1pct"),
        ("E1mega_gb1/summary.csv", "GB1 8000-q", "method", "top_1pct"),
        ("E5_cross_domain/summary.csv", "Trap-5 N=30", None, None),
    ]
    for ax, (p, lab, mc, sc) in zip(axes, pairs):
        f = ROOT / "experiments" / p
        if not f.exists(): ax.set_visible(False); continue
        df = pd.read_csv(f)
        if mc is None:
            # Trap-5 wide format
            for i, col in enumerate(["best_random", "best_adalead", "best_price_rl"]):
                ys = df[col].values
                xs = np.full(len(ys), i)
                ax.scatter(xs + np.random.uniform(-0.1, 0.1, len(xs)), ys,
                           s=18, alpha=0.7, color=["C7", "C0", "C1"][i])
            ax.set_xticks(range(3)); ax.set_xticklabels(["random", "adalead", "PRICE-RL"], rotation=20)
        else:
            order = ["random", "pex", "adalead", "gflownet_al",
                     "price_rl_fixed", "price_rl"]
            order = [m for m in order if m in df["method"].unique()]
            colors = {"random": "C7", "pex": "C5", "adalead": "C0",
                      "gflownet_al": "C2", "price_rl_fixed": "C3", "price_rl": "C1"}
            for i, m in enumerate(order):
                ys = df[df["method"] == m]["top_1pct"].values
                xs = np.full(len(ys), i)
                ax.scatter(xs + np.random.uniform(-0.1, 0.1, len(xs)), ys,
                           s=18, alpha=0.7, color=colors.get(m, "k"))
            ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, fontsize=7)
        ax.set_title(lab)
        ax.grid(alpha=0.25, axis="y")
    save_fig(fig, "fig_all_benchmarks_scatter")


if __name__ == "__main__":
    main()
