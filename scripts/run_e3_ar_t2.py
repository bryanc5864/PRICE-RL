# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E3-AR — Validate AR policy on NK as a test for whether the richer policy
class closes the T2 (controller tight tracking) gap."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nk_landscape import NKConfig, NKLandscape  # noqa: E402
from src.training.oracle import NKOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def main():
    OUT = ROOT / "experiments" / "E3AR_t2_test"; OUT.mkdir(parents=True, exist_ok=True)
    K_VALUES = [0, 3, 10, 19]
    SEEDS = list(range(5))
    N, A, ROUNDS, BATCH = 20, 4, 12, 64
    rows = []
    t0 = time.time()
    for policy_class in ["factorised", "ar"]:
        for K in K_VALUES:
            for seed in SEEDS:
                seed_everything(seed)
                nk = NKLandscape(NKConfig(N=N, K=K, alphabet=A, seed=seed))
                cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH,
                                    inner_steps=4, seed=seed,
                                    policy_class=policy_class,
                                    base_lr=1.0 if policy_class == "ar" else 2.0,
                                    ar_hidden=32)
                algo = PriceRL(N, A, NKOracle(nk), cfg); algo.run()
                # T2 tracking error
                err = np.array([abs(L_.rho - L_.rho_star) for L_ in algo.logs[-6:]])
                tight = float((err <= 0.05).mean())
                cos_min = float(min(L_.cos_pooled_vs_decomp for L_ in algo.logs))
                rows.append({"policy_class": policy_class, "K": K, "seed": seed,
                             "tight_pct": tight,
                             "best": max(L_.best_so_far for L_ in algo.logs),
                             "cos_T1_min": cos_min,
                             "mean_rho": float(np.mean([L_.rho for L_ in algo.logs[1:]])),
                             "mean_err_late": float(err.mean())})
                print(f"{policy_class:>10s} K={K:>2d} seed={seed} "
                      f"tight={tight:.2f} mean_err={err.mean():.3f} cos≥{cos_min:.4f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nE3-AR done in {time.time()-t0:.1f}s")
    # Aggregate
    import pandas as pd
    df = pd.read_csv(OUT / "summary.csv")
    for pc in df.policy_class.unique():
        d = df[df.policy_class == pc]
        print(f"  {pc:>12s}  tight%={d['tight_pct'].mean()*100:.1f}  mean_err={d['mean_err_late'].mean():.3f}")


if __name__ == "__main__":
    main()
