# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""Aggregate the large-scale CSVs (E3-L, E1-L, E2-L, E5-L, E6) into
publication tables + figures with bootstrap confidence intervals."""
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
OUT = ROOT / "figures"


def save_fig(fig, stem: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    rng = np.random.default_rng(0)
    means = []
    arr = np.asarray(values)
    for _ in range(n_boot):
        means.append(rng.choice(arr, size=len(arr), replace=True).mean())
    lo, hi = np.percentile(means, [(1-ci)/2*100, (1+ci)/2*100])
    return float(arr.mean()), float(lo), float(hi)


def analyze_E3L():
    p = ROOT / "experiments/E3L_nk_large/summary.csv"
    pr = ROOT / "experiments/E3L_nk_large/per_round.csv"
    if not p.exists():
        print("E3-L not yet available")
        return
    summ = pd.read_csv(p); pr_df = pd.read_csv(pr)
    print("\n========== E3-L: Large NK sweep ==========")
    print(f"N=40, K-grid×10 seeds×16 rounds = {len(pr_df)} round-seeds")

    # T1 cosine
    cos_min = pr_df["cos_T1"].min()
    cos_pct_at_1 = (np.abs(pr_df["cos_T1"] - 1.0) < 1e-3).mean() * 100
    print(f"T1 cos: min={cos_min:.4f}, {cos_pct_at_1:.1f}% rounds at 1.0000")

    # T3 with bootstrap CI on Pearson
    L_ref = 20.0
    summ["rho_star"] = L_ref / (summ["L_analytic"] + L_ref)
    g = summ.groupby("K").agg({"mean_rho": "mean", "rho_star": "mean"}).reset_index()
    r = float(np.corrcoef(g["mean_rho"], g["rho_star"])[0, 1])
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(5000):
        idx = rng.integers(0, len(g), size=len(g))
        rs.append(float(np.corrcoef(g["mean_rho"].iloc[idx], g["rho_star"].iloc[idx])[0, 1]))
    rs = np.array([x for x in rs if not np.isnan(x)])
    print(f"T3 Pearson r = {r:.3f}  (boot 95% CI = [{np.percentile(rs,2.5):.2f}, {np.percentile(rs,97.5):.2f}])")

    # Figure: best fitness vs K with bootstrap CI ribbons
    fig, ax = plt.subplots(figsize=(7, 3.6))
    methods = [("best_random", "Random", "C7"),
               ("best_adalead", "AdaLead", "C0"),
               ("best_fixed", "PRICE-RL (αS=αT)", "C3"),
               ("best_price_rl", "PRICE-RL (full)", "C1")]
    for col, lab, c in methods:
        means, los, his = [], [], []
        for K in sorted(summ.K.unique()):
            vals = summ[summ.K == K][col].values
            m, lo, hi = bootstrap_ci(vals)
            means.append(m); los.append(lo); his.append(hi)
        Ks = sorted(summ.K.unique())
        ax.plot(Ks, means, "o-", color=c, label=lab, lw=1.6)
        ax.fill_between(Ks, los, his, color=c, alpha=0.15)
    ax.set_xlabel("Ruggedness K (NK landscape, N=40)")
    ax.set_ylabel("Best fitness reached (10 seeds, 95% boot CI)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    save_fig(fig, "fig_E3L_best_vs_K_boot")

    # Figure: ρ_t spaghetti per K
    Ks = sorted(pr_df.K.unique())
    fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharey=True)
    for ax, K in zip(axes.flat, Ks):
        d = pr_df[pr_df.K == K]
        for s in sorted(d.seed.unique()):
            ds = d[d.seed == s].sort_values("round")
            ax.plot(ds["round"], ds["rho"], "C1-", alpha=0.4)
            ax.plot(ds["round"], ds["rho_star"], "C0--", alpha=0.4)
        ax.set_title(f"K={K}")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.25)
    fig.text(0.5, -0.01, "Round", ha="center")
    fig.text(0.0, 0.5, "ρ (orange) and ρ* (blue dashed)", va="center", rotation="vertical")
    save_fig(fig, "fig_E3L_rho_traj_grid")


def analyze_E1L():
    p = ROOT / "experiments/E1L_gb1_long/summary.csv"
    tr = ROOT / "experiments/E1L_gb1_long/trajectories.csv"
    if not p.exists():
        print("E1-L not yet available"); return
    summ = pd.read_csv(p); traj = pd.read_csv(tr)
    print("\n========== E1-L: Long-horizon GB1 ==========")
    for m in ["random", "pex", "adalead", "price_rl_fixed", "price_rl"]:
        d = summ[summ.method == m]["top_1pct"].values
        mean, lo, hi = bootstrap_ci(d)
        print(f"  {m:18s} top1%={mean:.3f}  95% CI [{lo:.3f},{hi:.3f}]  (n={len(d)})")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = {"random": "C7", "pex": "C5", "adalead": "C0",
              "price_rl_fixed": "C3", "price_rl": "C1"}
    for m in ["random", "pex", "adalead", "price_rl_fixed", "price_rl"]:
        d = traj[traj.method == m]
        rounds = sorted(d["round"].unique())
        means, los, his = [], [], []
        for t in rounds:
            vals = d[d["round"] == t]["best_so_far"].values
            mn, lo, hi = bootstrap_ci(vals)
            means.append(mn); los.append(lo); his.append(hi)
        ax.plot(rounds, means, "-", color=colors[m], label=m, lw=1.6)
        ax.fill_between(rounds, los, his, color=colors[m], alpha=0.15)
    ax.set_xlabel("Round"); ax.set_ylabel("Best fitness so far (10 seeds, 95% CI)")
    ax.set_title("E1-L: GB1 active learning over 20 rounds × 200 queries")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    save_fig(fig, "fig_E1L_trajectory")


def analyze_E2L():
    tr = ROOT / "experiments/E2L_reward_hacking_long/trace.csv"
    if not tr.exists():
        print("E2-L not yet available"); return
    df = pd.read_csv(tr)
    print("\n========== E2-L: Long-horizon reward hacking ==========")
    leads = []
    for s in df.seed.unique():
        d = df[df.seed == s].sort_values("round")
        rho_cross = d[d["rho"] >= 0.9]["round"].min() if (d["rho"] >= 0.9).any() else None
        unc_cross = d[d["proxy_unc_mean"] >= 0.95 * d["proxy_unc_mean"].iloc[-1]]["round"].min()
        if rho_cross is not None and not np.isnan(unc_cross):
            leads.append(int(unc_cross) - int(rho_cross))
    leads = np.array(leads)
    if len(leads):
        m, lo, hi = bootstrap_ci(leads)
        pct2 = float((leads >= 2).mean())
        print(f"  Lead distribution: mean={m:.1f} 95% CI [{lo:.1f},{hi:.1f}], "
              f"pct≥2 rounds={pct2*100:.1f}%, n={len(leads)} seeds")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for s in df.seed.unique():
        d = df[df.seed == s].sort_values("round")
        ax.plot(d["round"], d["rho"], "C1-", alpha=0.3)
        u = d["proxy_unc_mean"] / d["proxy_unc_mean"].max()
        ax.plot(d["round"], u, "C0-", alpha=0.3)
    ax.plot([], [], "C1-", label=r"$\rho_t$ (Price ratio)")
    ax.plot([], [], "C0-", label="proxy uncertainty (norm.)")
    ax.axhline(0.9, color="C1", ls=":", label=r"$\rho_t = 0.9$ alarm")
    ax.set_xlabel("Round"); ax.set_ylabel("Signal value")
    ax.set_title("E2-L: 30 rounds × 16 seeds × 5-model ensemble")
    ax.legend(); ax.grid(alpha=0.25)
    save_fig(fig, "fig_E2L_long_trajectory")


def analyze_E5L():
    p = ROOT / "experiments/E5L_trap_scaling/summary.csv"
    if not p.exists():
        print("E5-L not yet available"); return
    df = pd.read_csv(p)
    print("\n========== E5-L: Trap-K scaling ==========")
    for (N, K), d in df.groupby(["N", "K"]):
        for c in ["best_random", "best_adalead", "best_price_rl"]:
            mean, lo, hi = bootstrap_ci(d[c].values)
            print(f"  N={N:>3d} K={K} {c:18s} {mean:.3f}  CI [{lo:.3f},{hi:.3f}]")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    Ns = sorted(df.N.unique())
    width = 0.22
    for i, m in enumerate(["best_random", "best_adalead", "best_price_rl"]):
        means, errs = [], []
        labels = []
        for N in Ns:
            for K in sorted(df[df.N == N].K.unique()):
                d = df[(df.N == N) & (df.K == K)][m].values
                mn, lo, hi = bootstrap_ci(d)
                means.append(mn); errs.append((mn - lo, hi - mn))
                labels.append(f"N{N}\nK{K}")
        x = np.arange(len(means))
        errs = np.array(errs).T
        ax.bar(x + i*width, means, width=width,
               label=m.replace("best_", ""),
               yerr=errs, capsize=2,
               color=["C7", "C0", "C1"][i])
    ax.set_xticks(np.arange(len(labels)) + width)
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(1.0, color="k", ls="--", alpha=0.3, label="optimum")
    ax.set_ylabel("Best fitness reached"); ax.legend(fontsize=8)
    ax.set_title("E5-L: Trap-K scaling (5 seeds, 95% CI)")
    save_fig(fig, "fig_E5L_trap_scaling")


def analyze_E6():
    p = ROOT / "experiments/E6_hp_robustness/summary.csv"
    if not p.exists():
        print("E6 not yet available"); return
    df = pd.read_csv(p)
    print("\n========== E6: Hyperparameter robustness ==========")
    for task in df.task.unique():
        d = df[df.task == task]
        ranked = d.sort_values("best_mean", ascending=False)
        print(f"\n[{task}]  best 5 / worst 5 of {len(d)} configs")
        for _, r in ranked.head(5).iterrows():
            print(f"  lr={r.base_lr} inner={r.inner_steps} q={r.support_q}  "
                  f"score={r.best_mean:.3f}±{r.best_std:.3f}")
        print("  ...")
        for _, r in ranked.tail(5).iterrows():
            print(f"  lr={r.base_lr} inner={r.inner_steps} q={r.support_q}  "
                  f"score={r.best_mean:.3f}±{r.best_std:.3f}")
        # Heatmap aggregated over q
        agg = d.groupby(["base_lr", "inner_steps"])["best_mean"].mean().unstack()
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(agg.values, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(agg.shape[1])); ax.set_xticklabels(agg.columns)
        ax.set_yticks(range(agg.shape[0])); ax.set_yticklabels(agg.index)
        ax.set_xlabel("inner_steps"); ax.set_ylabel("base_lr")
        ax.set_title(f"{task}: PRICE-RL score (mean over q)")
        for i in range(agg.shape[0]):
            for j in range(agg.shape[1]):
                ax.text(j, i, f"{agg.values[i,j]:.2f}", ha="center", va="center",
                        color="w" if agg.values[i,j] < (agg.values.min()+agg.values.max())/2 else "k",
                        fontsize=7)
        plt.colorbar(im, ax=ax)
        save_fig(fig, f"fig_E6_heatmap_{task}")


def main():
    analyze_E3L()
    analyze_E1L()
    analyze_E2L()
    analyze_E5L()
    analyze_E6()


if __name__ == "__main__":
    main()
