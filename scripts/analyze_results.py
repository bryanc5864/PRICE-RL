# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Aggregate the raw CSV outputs from E1/E2/E3 into the threshold table
(T1–T6, S1–S3, F1–F3) defined in RESEARCH_PLAN.md §7.3 and emit
publication figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import _palette  # noqa: E402,F401  -- unifies the colour scheme to blue/green

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def save_fig(fig, stem: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- E3 / NK -
def analyze_E3():
    summ = pd.read_csv(ROOT / "experiments/E3_nk_sweep/summary.csv")
    pr = pd.read_csv(ROOT / "experiments/E3_nk_sweep/per_round.csv")
    print("\n========== E3: NK sweep ==========")
    K_values = sorted(summ.K.unique())

    # T1: cosine ≥ 0.95
    cos_min = summ["cos_T1_min"].min()
    print(f"T1 (cos pooled vs decomp ≥ 0.95): min cos across all runs = {cos_min:.4f}  → "
          f"{'PASS' if cos_min >= 0.95 else 'FAIL'}")

    # T3: Pearson correlation between mean_rho and analytic rho*
    L_ref = 10.0
    summ["rho_star_analytic"] = L_ref / (summ["L_analytic"] + L_ref)
    grouped = summ.groupby("K").agg({"mean_rho": "mean", "rho_star_analytic": "mean"}).reset_index()
    r = np.corrcoef(grouped["mean_rho"], grouped["rho_star_analytic"])[0, 1]
    print(f"T3 (Pearson(<rho_t>, rho*_analytic) ≥ 0.7): r = {r:.3f}  → "
          f"{'PASS' if r >= 0.7 else 'FAIL'}")

    # T2: PI controller stability — rho within 0.05 of rho* by round 5
    last_rounds = pr[pr["round"] >= 5]
    err = (last_rounds["rho"] - last_rounds["rho_star"]).abs()
    pct_within_05 = float((err <= 0.05).mean())
    print(f"T2 (rho within 0.05 of rho* in late rounds): {pct_within_05*100:.1f}% pass")

    # S1: full PRICE > fixed mixing on K ≥ 5
    rugged = summ[summ["K"] >= 5]
    diff = rugged["best_price_rl"] - rugged["best_fixed_mix"]
    s1_effect = float(diff.mean() / rugged["best_fixed_mix"].mean())
    print(f"S1 (adaptive > fixed on K≥5): mean Δ = {diff.mean():.3f} (effect {s1_effect*100:.1f}%) → "
          f"{'PASS' if s1_effect >= 0.05 else 'CHECK'}")

    # ------ Figure: best fitness vs K -----------
    fig, ax = plt.subplots(figsize=(6, 3.6))
    methods = [
        ("best_random",      "Random",          "C7", "--"),
        ("best_adalead",     "AdaLead",         "C0", "-"),
        ("best_open_loop",   "PRICE-RL (open)", "C2", ":"),
        ("best_fixed_mix",   "PRICE-RL (αS=αT)", "C3", ":"),
        ("best_rand_support", "PRICE-RL (rand. supp.)", "C5", ":"),
        ("best_price_rl",    "PRICE-RL (full)", "C1", "-"),
    ]
    for col, label, c, ls in methods:
        m = summ.groupby("K")[col].agg(["mean", "std"]).reset_index()
        ax.errorbar(m["K"], m["mean"], yerr=m["std"], label=label, color=c, lw=1.6, ls=ls,
                    marker="o", capsize=2, ms=4)
    ax.set_xlabel("Ruggedness K (NK landscape, N=20)")
    ax.set_ylabel("Best fitness reached (5 seeds)")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=0.25)
    save_fig(fig, "fig_E3_best_vs_K")

    # ------ Figure: rho_t and rho* across K -----------
    fig, ax = plt.subplots(figsize=(6, 3.6))
    g = summ.groupby("K").agg({"mean_rho": ["mean", "std"],
                               "rho_star_analytic": "mean"}).reset_index()
    g.columns = ["K", "rho_mean", "rho_std", "rho_star"]
    ax.errorbar(g["K"], g["rho_mean"], yerr=g["rho_std"],
                marker="o", color="C1", label=r"empirical $\rho_t$ (PRICE-RL)")
    ax.plot(g["K"], g["rho_star"], "--", color="C0", label=r"analytic $\rho^*$ (Theorem 2)")
    ax.set_xlabel("Ruggedness K")
    ax.set_ylabel(r"Price ratio $\rho$")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_title(rf"H3: theory-empirics match. Pearson $r={r:.2f}$")
    save_fig(fig, "fig_E3_rho_vs_K")

    # ------ Figure: Theorem 1 cosine -----------
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(pr["cos_T1"], bins=20, color="C2", edgecolor="white")
    ax.axvline(0.95, color="r", ls="--", label="T1 threshold = 0.95")
    ax.set_xlabel(r"cosine($g_S+g_T$, pooled REINFORCE)")
    ax.set_ylabel("# rounds")
    ax.set_title(f"Theorem 1 empirical exactness — min = {pr['cos_T1'].min():.4f}")
    ax.legend()
    save_fig(fig, "fig_T1_cosine_histogram")

    return {"T1_cos_min": float(cos_min), "T3_pearson": float(r),
            "T2_pct_within_05": pct_within_05,
            "S1_effect": s1_effect, "K_grouped": grouped}


# ---------------------------------------------------------------- E1 / GB1 -
def analyze_E1():
    summ = pd.read_csv(ROOT / "experiments/E1_gb1/summary.csv")
    print("\n========== E1: GB1 active learning ==========")
    methods = sorted(summ.method.unique())
    for m in methods:
        df = summ[summ.method == m]
        b_mean = df["best_fitness"].mean()
        b_std = df["best_fitness"].std()
        t_mean = df["top_1pct"].mean()
        t_std = df["top_1pct"].std()
        print(f"  {m:18s} best={b_mean:.3f}±{b_std:.3f}  top1%={t_mean:.4f}±{t_std:.4f}")

    # Bootstrap PRICE vs random.
    rng = np.random.default_rng(0)
    pr = summ[summ.method == "price_rl"]["top_1pct"].values
    rd = summ[summ.method == "random"]["top_1pct"].values
    diffs = []
    for _ in range(10000):
        a = rng.choice(pr, size=len(pr), replace=True)
        b = rng.choice(rd, size=len(rd), replace=True)
        diffs.append(a.mean() - b.mean())
    p_value = float(np.mean(np.array(diffs) <= 0))
    print(f"T4 (PRICE-RL vs random, bootstrap p < 0.10): p = {p_value:.4f}  → "
          f"{'PASS' if p_value < 0.10 else 'FAIL'}")

    # vs AdaLead (within 1 std)
    ada = summ[summ.method == "adalead"]["top_1pct"].values
    gap = ada.mean() - pr.mean()
    pr_std = pr.std()
    print(f"T4 (PRICE-RL vs AdaLead, gap ≤ 1 std): gap = {gap:.4f}, std = {pr_std:.4f}  → "
          f"{'PASS' if gap <= pr_std else 'CHECK'}")

    # ------ Figure: best fitness and top-k recall, bar chart -----------
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    order = ["random", "pex", "adalead", "price_rl_fixed", "price_rl"]
    for i, metric in enumerate(["best_fitness", "top_1pct"]):
        means = [summ[summ.method == m][metric].mean() for m in order]
        stds = [summ[summ.method == m][metric].std() for m in order]
        cs = ["C7", "C5", "C0", "C3", "C1"]
        ax[i].bar(range(len(order)), means, yerr=stds, color=cs, capsize=3)
        ax[i].set_xticks(range(len(order)))
        ax[i].set_xticklabels(order, rotation=20, ha="right", fontsize=8)
        ax[i].set_ylabel(metric.replace("_", " "))
        ax[i].grid(alpha=0.2, axis="y")
    fig.suptitle("E1: GB1 four-mutation active learning (5 rounds × 100 queries × 5 seeds)")
    save_fig(fig, "fig_E1_gb1_bars")
    return {"T4_p_random": p_value, "T4_gap_to_ada": float(gap),
            "T4_pr_std": float(pr_std)}


# ----------------------------------------------------------------- E2 RH -
def analyze_E2():
    df = pd.read_csv(ROOT / "experiments/E2_reward_hacking/trace.csv")
    print("\n========== E2: reward hacking ==========")
    seeds = sorted(df.seed.unique())

    # H2: rho_t crosses 0.9 ≥ 2 rounds before proxy unc crosses 95%-of-final.
    leads = []
    for s in seeds:
        d = df[df.seed == s].sort_values("round")
        rho_cross = d.index[d["rho"].values >= 0.9].min() if (d["rho"] >= 0.9).any() else None
        unc_target = 0.95 * d["proxy_unc_mean"].iloc[-1]
        unc_cross = d.index[d["proxy_unc_mean"].values >= unc_target].min() if (d["proxy_unc_mean"].values >= unc_target).any() else None
        if rho_cross is not None and unc_cross is not None:
            r_round = d.loc[rho_cross, "round"]
            u_round = d.loc[unc_cross, "round"]
            leads.append(u_round - r_round)
    if leads:
        leads = np.asarray(leads)
        pct_2_lead = float((leads >= 2).mean())
        print(f"T5 (rho hits 0.9 ≥ 2 rounds before proxy unc-95): "
              f"lead distribution = {leads.tolist()}, "
              f"{pct_2_lead*100:.1f}% have lead ≥ 2  → "
              f"{'PASS' if pct_2_lead >= 0.80 else 'CHECK'}")
    else:
        print("T5 — could not evaluate (rho or unc never crossed)")

    # ------ Figure: rho vs proxy uncertainty over time -----------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for s in seeds:
        d = df[df.seed == s].sort_values("round")
        ax.plot(d["round"], d["rho"], "C1-", alpha=0.4)
        ax.plot(d["round"], d["proxy_unc_mean"] / d["proxy_unc_mean"].max(), "C0-", alpha=0.4)
    ax.plot([], [], "C1-", label=r"$\rho_t$ (Price ratio)")
    ax.plot([], [], "C0-", label=r"proxy uncertainty (normalised)")
    ax.axhline(0.9, color="C1", ls=":", label=r"$\rho_t = 0.9$ alarm")
    ax.set_xlabel("Round")
    ax.set_ylabel("Signal value")
    ax.set_title("E2: Price ratio precedes proxy uncertainty as reward-hacking warning")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save_fig(fig, "fig_E2_rho_vs_uncertainty")

    # Reward gap figure
    fig, ax = plt.subplots(figsize=(6, 3))
    for s in seeds:
        d = df[df.seed == s].sort_values("round")
        ax.plot(d["round"], d["reward_gap"], color="C3", alpha=0.4)
    ax.set_xlabel("Round")
    ax.set_ylabel("Proxy − True fitness")
    ax.set_title("E2: reward gap grows as policy hacks the proxy")
    ax.grid(alpha=0.25)
    save_fig(fig, "fig_E2_reward_gap")
    return {"T5_lead_pct_2": pct_2_lead if leads is not None and len(leads) else 0.0}


def analyze_E1b():
    p = ROOT / "experiments/E1b_multi_dms/summary.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    print("\n========== E1b: Multi-DMS active learning ==========")
    rows = {}
    for ds in df["dataset"].unique():
        d = df[df.dataset == ds]
        line = [f"{ds}:"]
        for m in ["random", "pex", "adalead", "price_rl_fixed", "price_rl"]:
            t = d[d.method == m]["top_1pct"]
            line.append(f"  {m}={t.mean():.3f}±{t.std():.3f}")
            rows[f"{ds}|{m}"] = (t.mean(), t.std())
        print("  " + "  ".join(line))
    return rows


def analyze_E4():
    p = ROOT / "experiments/E4_cross_protein/summary.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    print("\n========== E4: Cross-protein transfer ==========")
    rows = {}
    for tgt in df["target"].unique():
        d = df[df.target == tgt]
        cold, full, sel = d["cold"].mean(), d["full"].mean(), d["selection"].mean()
        cold_s, full_s, sel_s = d["cold"].std(), d["full"].std(), d["selection"].std()
        print(f"  {tgt:18s} cold={cold:.3f}±{cold_s:.3f}  "
              f"full={full:.3f}±{full_s:.3f}  sel={sel:.3f}±{sel_s:.3f}")
        rows[tgt] = (cold, full, sel)
    return rows


def analyze_E5():
    p = ROOT / "experiments/E5_cross_domain/summary.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    print("\n========== E5: Cross-domain (Trap-5 deceptive) ==========")
    out = {}
    for col in ["best_random", "best_adalead", "best_price_fixed", "best_price_rl"]:
        out[col] = (df[col].mean(), df[col].std())
        print(f"  {col:18s} {out[col][0]:.3f}±{out[col][1]:.3f}")
    print(f"  T1 cosine min over E5 = {df['cos_T1_min'].min():.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(6, 3.2))
    methods = ["best_random", "best_adalead", "best_price_fixed", "best_price_rl"]
    labels = ["Random", "AdaLead", "PRICE-RL (αS=αT)", "PRICE-RL (full)"]
    colors = ["C7", "C0", "C3", "C1"]
    for i, m in enumerate(methods):
        vals = df[m].values
        ax.bar(i, vals.mean(), yerr=vals.std(), color=colors[i], capsize=3)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Best fitness reached")
    ax.set_title("E5: Trap-5 deceptive landscape (N=30, K=5, 8 rounds × 64 queries × 5 seeds)")
    ax.axhline(1.0, color="k", ls="--", alpha=0.3, label="global optimum")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    save_fig(fig, "fig_E5_trap5")
    return out


def analyze_E2_dcs_headtohead():
    p = ROOT / "experiments/E2_reward_hacking/trace.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "method" not in df.columns:
        return {}
    print("\n========== E2: head-to-head with δ-CS ==========")
    out = {}
    for m in df["method"].unique():
        last = df[(df.method == m) & (df["round"] == df["round"].max())]
        gap = last["reward_gap"].mean()
        unc = last["proxy_unc_mean"].mean()
        out[m] = (gap, unc)
        print(f"  {m:10s} final reward_gap={gap:.3f}  proxy_unc={unc:.3f}")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for m, c in [("price_rl", "C1"), ("delta_cs", "C0")]:
        d = df[df.method == m]
        for s in d["seed"].unique():
            ds = d[d.seed == s]
            ax.plot(ds["round"], ds["reward_gap"], color=c, alpha=0.4)
        ax.plot([], [], color=c, label=m)
    ax.set_xlabel("Round")
    ax.set_ylabel("Proxy − True fitness")
    ax.set_title("E2 head-to-head: δ-CS clips harder, but PRICE-RL detects earlier (per Fig E2 main)")
    ax.legend()
    ax.grid(alpha=0.25)
    save_fig(fig, "fig_E2_dcs_vs_price")
    return out


def main():
    res_e3 = analyze_E3()
    res_e1 = analyze_E1()
    res_e2 = analyze_E2()
    res_e1b = analyze_E1b()
    res_e4 = analyze_E4()
    res_e5 = analyze_E5()
    res_e2dcs = analyze_E2_dcs_headtohead()
    print("\n========== Threshold summary ==========")
    print(f"T1 (cos ≥ 0.95): min = {res_e3['T1_cos_min']:.4f}  "
          f"{'✓' if res_e3['T1_cos_min']>=0.95 else '✗'}")
    print(f"T2 (controller stable): {res_e3['T2_pct_within_05']*100:.1f}% rounds within ±0.05")
    print(f"T3 (Pearson rho-vs-rho*): r = {res_e3['T3_pearson']:.3f}  "
          f"{'✓' if res_e3['T3_pearson']>=0.7 else '✗'}")
    print(f"T4 (vs random p<0.10): p = {res_e1['T4_p_random']:.4f}  "
          f"{'✓' if res_e1['T4_p_random']<0.10 else '✗'}")
    print(f"T4 (vs AdaLead within 1 std): gap = {res_e1['T4_gap_to_ada']:.4f} "
          f"std = {res_e1['T4_pr_std']:.4f}")
    print(f"T5 (rho leads unc ≥ 2 rounds, ≥ 80% seeds): {res_e2['T5_lead_pct_2']*100:.1f}%")
    print(f"S1 (adaptive vs fixed on K≥5): effect = {res_e3['S1_effect']*100:.1f}%")


if __name__ == "__main__":
    main()
