# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E17 — Diversity-aware metrics on GB1 mega-budget.
Re-evaluates the Phase-B E1-mega trajectories for top-K-unique and
mean-pairwise-Hamming, in addition to top-1% recall.
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
from src.evaluation.diversity import (mean_pairwise_hamming,  # noqa: E402
                                      top_k_unique)
from src.evaluation.metrics import top_k_recovery  # noqa: E402


def main():
    OUT = ROOT / "experiments" / "E17_diversity"; OUT.mkdir(parents=True, exist_ok=True)
    ds = load_gb1_wu2016(); seqs, fit = ds["sequences"], ds["fitness"]
    L, A = seqs.shape[1], len(ds["alphabet"])
    SEEDS = list(range(5)); ROUNDS, BATCH = 20, 200
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        seed_everything(seed)
        for method in ["random", "pex", "adalead"]:
            rng = np.random.default_rng(seed)
            hX, hR = [], []
            for t in range(ROUNDS):
                if method == "random" or not hR:
                    X = random_sampler(A, L, BATCH, rng)
                elif method == "adalead":
                    X = adalead(np.concatenate(hX, 0), np.asarray(hR), A, L, BATCH, rng)
                elif method == "pex":
                    X = pex(np.concatenate(hX, 0), np.asarray(hR), A, L, BATCH, rng)
                R = TableOracle(seqs, fit).query(X)
                hX.append(X); hR.extend(R.tolist())
            hX_arr = np.concatenate(hX, 0)
            rows.append({
                "method": method, "seed": seed,
                "top_1pct": top_k_recovery(hR, fit, 0.01),
                "top_K_unique": top_k_unique(hX_arr, hR, fit, 0.01),
                "mean_pairwise_hamming_top10": mean_pairwise_hamming(hX_arr, hR, k=10),
            })
        # PRICE-RL
        cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed)
        algo = PriceRL(L, A, TableOracle(seqs, fit), cfg); algo.run()
        hX_arr = np.concatenate(algo.history_X, 0)
        rows.append({
            "method": "price_rl", "seed": seed,
            "top_1pct": top_k_recovery(algo.history_R, fit, 0.01),
            "top_K_unique": top_k_unique(hX_arr, algo.history_R, fit, 0.01),
            "mean_pairwise_hamming_top10": mean_pairwise_hamming(hX_arr, algo.history_R, k=10),
        })
        print(f"seed={seed} done")

    with open(OUT / "summary.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nE17 done in {time.time()-t0:.1f}s")
    for m in ["random", "pex", "adalead", "price_rl"]:
        d = [r for r in rows if r["method"] == m]
        if d:
            top1 = np.mean([r["top_1pct"] for r in d])
            tku = np.mean([r["top_K_unique"] for r in d])
            mph = np.mean([r["mean_pairwise_hamming_top10"] for r in d])
            print(f"  {m:18s} top-1%={top1:.3f}  top-K-unique={tku:.1f}  pair-Hamming-top10={mph:.2f}")


if __name__ == "__main__":
    main()
