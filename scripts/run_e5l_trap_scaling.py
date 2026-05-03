# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""E5-L — Trap-K scaling. N∈{30,60,120}, K∈{3,5,7,10}, 5 seeds, 16 rounds."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.trap_landscape import TrapKLandscape  # noqa: E402
from src.training.baselines import adalead  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


class TrapOracle:
    def __init__(self, lm):
        self.lm = lm; self.calls = 0; self.misses = 0

    def query(self, X):
        self.calls += X.shape[0]
        return self.lm.fitness_batch(X)


def main():
    OUT = ROOT / "experiments" / "E5L_trap_scaling"; OUT.mkdir(parents=True, exist_ok=True)
    Ns = [30, 60, 120]
    Ks = [3, 5, 7, 10]
    SEEDS = [0, 1, 2, 3, 4]
    ROUNDS, BATCH = 16, 64
    rows = []
    t0 = time.time()
    for N in Ns:
        for K in Ks:
            if N % K != 0:
                continue
            for seed in SEEDS:
                seed_everything(seed)
                lm = TrapKLandscape(N=N, K=K)
                rng = np.random.default_rng(seed)
                hX, hR = [], []
                for t in range(ROUNDS):
                    if not hR:
                        X = rng.integers(0, 2, size=(BATCH, N), dtype=np.int64)
                    else:
                        X = adalead(np.concatenate(hX, 0), np.asarray(hR), 2, N,
                                    BATCH, rng, kappa=0.1, parents=8)
                    R = TrapOracle(lm).query(X)
                    hX.append(X); hR.extend(R.tolist())
                best_ada = float(max(hR))
                rng = np.random.default_rng(seed)
                hRr = []
                for t in range(ROUNDS):
                    X = rng.integers(0, 2, size=(BATCH, N), dtype=np.int64)
                    hRr.extend(TrapOracle(lm).query(X).tolist())
                best_rd = float(max(hRr))
                cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed)
                algo = PriceRL(N, 2, TrapOracle(lm), cfg); algo.run()
                best_pr = max(L.best_so_far for L in algo.logs)
                cos_min = float(min(L.cos_pooled_vs_decomp for L in algo.logs))
                rows.append({"N": N, "K": K, "seed": seed,
                             "best_random": best_rd, "best_adalead": best_ada,
                             "best_price_rl": best_pr, "cos_T1_min": cos_min})
                print(f"N={N:>3d} K={K} seed={seed} rd={best_rd:.3f} ada={best_ada:.3f} "
                      f"price={best_pr:.3f} cos≥{cos_min:.4f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\nE5-L done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
