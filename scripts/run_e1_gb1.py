# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
Experiment E1 — Low-data active-learning on GB1 four-mutation landscape
(Wu et al. 2016). Tests Hypothesis H1, hard threshold T4.

Protocol: 5 rounds × B = 100 queries × 5 seeds.
Methods: random, AdaLead, PEX, PRICE-RL (full), PRICE-RL (no adaptive).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dms_loaders import load_gb1_wu2016  # noqa: E402
from src.training.baselines import adalead, pex, random_sampler  # noqa: E402
from src.training.oracle import TableOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402
from src.evaluation.metrics import top_k_recovery  # noqa: E402


def baseline_run(name: str, alphabet: int, length: int, oracle, rounds: int,
                 batch: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    history_X: list[np.ndarray] = []
    history_R: list[float] = []
    for t in range(rounds):
        if name == "random" or len(history_R) == 0:
            X = random_sampler(alphabet, length, batch, rng)
        elif name == "adalead":
            hX = np.concatenate(history_X, 0)
            hR = np.asarray(history_R)
            X = adalead(hX, hR, alphabet, length, batch, rng)
        elif name == "pex":
            hX = np.concatenate(history_X, 0)
            hR = np.asarray(history_R)
            X = pex(hX, hR, alphabet, length, batch, rng)
        else:
            raise ValueError(name)
        R = oracle.query(X)
        history_X.append(X)
        history_R.extend(R.tolist())
    return history_R


def main():
    OUT = ROOT / "experiments" / "E1_gb1"
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_gb1_wu2016()
    seqs, fit = ds["sequences"], ds["fitness"]
    L = seqs.shape[1]
    A = len(ds["alphabet"])
    print(f"GB1 N={seqs.shape[0]} L={L} A={A}")

    SEEDS = [0, 1, 2, 3, 4]
    ROUNDS = 5
    BATCH = 100

    rows = []
    t0 = time.time()
    for seed in SEEDS:
        seed_everything(seed)
        oracle = TableOracle(seqs, fit)
        # Baselines
        R_rand = baseline_run("random", A, L, oracle, ROUNDS, BATCH, seed)
        R_ada = baseline_run("adalead", A, L, TableOracle(seqs, fit), ROUNDS, BATCH, seed)
        R_pex = baseline_run("pex", A, L, TableOracle(seqs, fit), ROUNDS, BATCH, seed)

        # PRICE-RL full
        cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed)
        algo = PriceRL(L, A, TableOracle(seqs, fit), cfg)
        algo.run()
        R_price = algo.history_R
        cos_min = float(min(l.cos_pooled_vs_decomp for l in algo.logs))

        # Ablation: fixed mixing (no adaptive)
        cfg_f = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed,
                              fix_alpha_S=0.5, fix_alpha_T=0.5)
        algo_f = PriceRL(L, A, TableOracle(seqs, fit), cfg_f)
        algo_f.run()
        R_price_fix = algo_f.history_R

        for label, R in [("random", R_rand), ("adalead", R_ada), ("pex", R_pex),
                         ("price_rl", R_price), ("price_rl_fixed", R_price_fix)]:
            rows.append({
                "method": label,
                "seed": seed,
                "best_fitness": float(np.max(R)),
                "mean_fitness": float(np.mean(R)),
                "top_1pct": top_k_recovery(R, fit, 0.01),
                "n_queries": len(R),
                "cos_T1_min": cos_min if "price_rl" in label else 1.0,
            })
        print(f"seed={seed} best random={max(R_rand):.3f} ada={max(R_ada):.3f} "
              f"pex={max(R_pex):.3f} price={max(R_price):.3f} price-fix={max(R_price_fix):.3f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"E1 done in {time.time()-t0:.1f}s")

    # Aggregate
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        bs = [r["best_fitness"] for r in rows if r["method"] == m]
        ts = [r["top_1pct"] for r in rows if r["method"] == m]
        print(f"  {m:18s} best={np.mean(bs):.3f}±{np.std(bs):.3f} "
              f"top-1%-recall={np.mean(ts):.3f}")


if __name__ == "__main__":
    main()
