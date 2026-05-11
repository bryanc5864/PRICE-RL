# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""E7 — T2 closure via entropy injection.
Full sweep: factorised vs AR × inject vs no-inject × K ∈ {0,3,5,10,15,19} × 5 seeds.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nk_landscape import NKConfig, NKLandscape  # noqa: E402
from src.training.oracle import NKOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def main():
    OUT = ROOT / "experiments" / "E7_t2_closure"; OUT.mkdir(parents=True, exist_ok=True)
    K_VALUES = [0, 3, 5, 10, 15, 19]
    SEEDS = list(range(5))
    N, A, ROUNDS, BATCH = 20, 4, 12, 64
    rows = []
    t0 = time.time()
    configs = [
        ("factorised",  False, 0.0),
        ("factorised",  True,  0.4),
        ("ar",          False, 0.0),
        ("ar",          True,  0.4),
    ]
    for policy_class, inject, gain in configs:
        for K in K_VALUES:
            for seed in SEEDS:
                seed_everything(seed)
                nk = NKLandscape(NKConfig(N=N, K=K, alphabet=A, seed=seed))
                cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=4,
                                    seed=seed, policy_class=policy_class,
                                    base_lr=1.0 if policy_class == "ar" else 2.0,
                                    ar_hidden=32,
                                    entropy_inject=inject and policy_class == "factorised",
                                    entropy_inject_gain=gain)
                algo = PriceRL(N, A, NKOracle(nk), cfg); algo.run()
                err = np.array([abs(L.rho - L.rho_star) for L in algo.logs[-6:]])
                tight = (err <= 0.05).mean()
                tight03 = (err <= 0.10).mean()
                cos_min = float(min(L.cos_pooled_vs_decomp for L in algo.logs))
                rows.append({
                    "policy_class": policy_class, "inject": inject, "gain": gain,
                    "K": K, "seed": seed,
                    "tight_05": float(tight), "tight_10": float(tight03),
                    "mean_err_late": float(err.mean()),
                    "best": max(L.best_so_far for L in algo.logs),
                    "cos_T1_min": cos_min,
                    "mean_rho": float(np.mean([L.rho for L in algo.logs[1:]])),
                })
                print(f"{policy_class:>10s} inject={int(inject)} gain={gain} "
                      f"K={K:>2d} seed={seed} tight5={tight:.2f} tight10={tight03:.2f} "
                      f"err={err.mean():.3f} cos≥{cos_min:.4f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nE7 done in {time.time()-t0:.1f}s")
    df = pd.read_csv(OUT / "summary.csv")
    print("\n=== E7 aggregate ===")
    for (pc, inj, g), d in df.groupby(["policy_class", "inject", "gain"]):
        print(f"  {pc:>10s} inject={inj} gain={g}  "
              f"tight5%={d['tight_05'].mean()*100:>5.1f}  "
              f"tight10%={d['tight_10'].mean()*100:>5.1f}  "
              f"err={d['mean_err_late'].mean():.3f}  "
              f"best={d['best'].mean():.3f}  "
              f"cos≥{d['cos_T1_min'].min():.4f}")


if __name__ == "__main__":
    main()
