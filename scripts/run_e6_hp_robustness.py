# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""E6 — Hyperparameter robustness study.
Grid over base_lr × inner_steps × support_quantile on NK + GB1.
"""
from __future__ import annotations

import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dms_loaders import load_gb1_wu2016  # noqa: E402
from src.data.nk_landscape import NKConfig, NKLandscape  # noqa: E402
from src.training.oracle import NKOracle, TableOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402
from src.evaluation.metrics import top_k_recovery  # noqa: E402


def main():
    OUT = ROOT / "experiments" / "E6_hp_robustness"; OUT.mkdir(parents=True, exist_ok=True)
    base_lrs = [0.5, 1.0, 2.0, 4.0]
    inner_stepss = [4, 8, 16]
    qs = [0.02, 0.05, 0.10, 0.20]
    SEEDS = [0, 1, 2, 3, 4]
    rows = []
    t0 = time.time()
    print("== NK(N=20,K=10) ==")
    for base_lr, inner, q in itertools.product(base_lrs, inner_stepss, qs):
        bests, cossigm = [], []
        for seed in SEEDS:
            seed_everything(seed)
            nk = NKLandscape(NKConfig(N=20, K=10, alphabet=4, seed=seed))
            cfg = PriceRLConfig(rounds=8, batch=64, inner_steps=inner, seed=seed,
                                base_lr=base_lr, support_quantile=q)
            algo = PriceRL(20, 4, NKOracle(nk), cfg); algo.run()
            bests.append(max(L.best_so_far for L in algo.logs))
            cossigm.append(min(L.cos_pooled_vs_decomp for L in algo.logs))
        rows.append({"task": "NK_K10", "base_lr": base_lr, "inner_steps": inner,
                     "support_q": q, "best_mean": float(np.mean(bests)),
                     "best_std": float(np.std(bests)),
                     "cos_T1_min": float(np.min(cossigm))})
        print(f"  lr={base_lr} inner={inner} q={q} best={np.mean(bests):.3f}±{np.std(bests):.3f} cos≥{np.min(cossigm):.4f}")

    print("== GB1 (5 rounds × 100) ==")
    ds = load_gb1_wu2016(); seqs, fit = ds["sequences"], ds["fitness"]
    L_, A = seqs.shape[1], len(ds["alphabet"])
    for base_lr, inner, q in itertools.product(base_lrs, inner_stepss, qs):
        recalls = []
        for seed in SEEDS:
            seed_everything(seed)
            cfg = PriceRLConfig(rounds=5, batch=100, inner_steps=inner, seed=seed,
                                base_lr=base_lr, support_quantile=q)
            algo = PriceRL(L_, A, TableOracle(seqs, fit), cfg); algo.run()
            recalls.append(top_k_recovery(algo.history_R, fit, 0.01))
        rows.append({"task": "GB1", "base_lr": base_lr, "inner_steps": inner,
                     "support_q": q, "best_mean": float(np.mean(recalls)),
                     "best_std": float(np.std(recalls)),
                     "cos_T1_min": np.nan})
        print(f"  lr={base_lr} inner={inner} q={q} top1%={np.mean(recalls):.3f}±{np.std(recalls):.3f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\nE6 done in {time.time()-t0:.1f}s ({len(rows)} configs)")


if __name__ == "__main__":
    main()
