# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
Experiment E1b — multi-DMS active learning. Extends E1 (GB1) to GFP,
AAV, TEM-1, BCL11A enhancer. 5 rounds × B=100 × 5 seeds.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dms_loaders import (load_aav_sinai2021,  # noqa: E402
                                  load_enhancer_bcl11a,
                                  load_gfp_sarkisyan2016,
                                  load_tem1_stiffler2015)
from src.training.baselines import adalead, pex, random_sampler  # noqa: E402
from src.training.oracle import TableOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402
from src.evaluation.metrics import top_k_recovery  # noqa: E402


def baseline_run(name, alphabet, length, oracle, rounds, batch, seed):
    rng = np.random.default_rng(seed)
    history_X, history_R = [], []
    for t in range(rounds):
        if name == "random" or len(history_R) == 0:
            X = random_sampler(alphabet, length, batch, rng)
        elif name == "adalead":
            X = adalead(np.concatenate(history_X, 0), np.asarray(history_R),
                        alphabet, length, batch, rng)
        elif name == "pex":
            X = pex(np.concatenate(history_X, 0), np.asarray(history_R),
                    alphabet, length, batch, rng)
        R = oracle.query(X)
        history_X.append(X); history_R.extend(R.tolist())
    return history_R


def main():
    OUT = ROOT / "experiments" / "E1b_multi_dms"
    OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = [0, 1, 2, 3, 4]
    ROUNDS, BATCH = 5, 100
    datasets = {
        "TEM1_Stiffler2015": load_tem1_stiffler2015(),
        "Enhancer_BCL11A":   load_enhancer_bcl11a(),
    }
    rows = []
    t0 = time.time()
    for ds_name, ds in datasets.items():
        seqs, fit = ds["sequences"], ds["fitness"]
        L, A = seqs.shape[1], len(ds["alphabet"])
        print(f"\n[{ds_name}] N={seqs.shape[0]} L={L} A={A}")
        for seed in SEEDS:
            seed_everything(seed)
            for method in ["random", "adalead", "pex"]:
                R = baseline_run(method, A, L, TableOracle(seqs, fit),
                                 ROUNDS, BATCH, seed)
                rows.append({"dataset": ds_name, "method": method, "seed": seed,
                             "best": float(np.max(R)),
                             "top_1pct": top_k_recovery(R, fit, 0.01)})
            cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed)
            algo = PriceRL(L, A, TableOracle(seqs, fit), cfg)
            algo.run()
            cos_min = float(min(L_.cos_pooled_vs_decomp for L_ in algo.logs))
            rows.append({"dataset": ds_name, "method": "price_rl", "seed": seed,
                         "best": float(np.max(algo.history_R)),
                         "top_1pct": top_k_recovery(algo.history_R, fit, 0.01),
                         "cos_T1_min": cos_min})
            cfg_f = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed,
                                  fix_alpha_S=0.5, fix_alpha_T=0.5)
            algo_f = PriceRL(L, A, TableOracle(seqs, fit), cfg_f)
            algo_f.run()
            rows.append({"dataset": ds_name, "method": "price_rl_fixed", "seed": seed,
                         "best": float(np.max(algo_f.history_R)),
                         "top_1pct": top_k_recovery(algo_f.history_R, fit, 0.01)})
        # per-dataset summary
        for m in ["random", "adalead", "pex", "price_rl_fixed", "price_rl"]:
            tops = [r["top_1pct"] for r in rows
                    if r["dataset"] == ds_name and r["method"] == m]
            print(f"  {m:16s} top1%={np.mean(tops):.4f}±{np.std(tops):.4f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "seed", "best",
                                          "top_1pct", "cos_T1_min"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nE1b done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
