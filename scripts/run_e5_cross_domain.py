# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Experiment E5 — cross-domain validation (RESEARCH_PLAN.md §7.1, item 5).

Apply PRICE-RL to a non-biology combinatorial optimisation benchmark:
the Trap-5 deceptive landscape (Goldberg 1989). The global optimum is
the all-ones string but the average gradient points to all zeros —
exactly the regime where naive policy gradient fails. Tests whether
PRICE-RL's decomposition recovers known structure on a non-biology
problem.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.trap_landscape import TrapKLandscape  # noqa: E402
from src.training.baselines import adalead, random_sampler  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


class TrapOracle:
    def __init__(self, lm: TrapKLandscape):
        self.lm = lm
        self.calls = 0
        self.misses = 0

    def query(self, X: np.ndarray) -> np.ndarray:
        self.calls += X.shape[0]
        return self.lm.fitness_batch(X)


def main():
    OUT = ROOT / "experiments" / "E5_cross_domain"
    OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = [0, 1, 2, 3, 4]
    K_BLOCK = 5
    N = 30
    ROUNDS, BATCH = 8, 64
    rows = []
    t0 = time.time()
    lm = TrapKLandscape(N=N, K=K_BLOCK)
    print(f"[Trap-{K_BLOCK}] N={N} A=2 (binary); global optimum f=1.0 (all-ones)")

    for seed in SEEDS:
        seed_everything(seed)
        # Random
        rng = np.random.default_rng(seed)
        history_R = []
        for _ in range(ROUNDS):
            X = rng.integers(0, 2, size=(BATCH, N), dtype=np.int64)
            history_R.extend(TrapOracle(lm).query(X).tolist())
        best_rand = float(np.max(history_R))

        # AdaLead
        rng = np.random.default_rng(seed)
        hX, hR = [], []
        for t_ in range(ROUNDS):
            if not hR:
                X = rng.integers(0, 2, size=(BATCH, N), dtype=np.int64)
            else:
                X = adalead(np.concatenate(hX, 0), np.asarray(hR), 2, N, BATCH, rng,
                            kappa=0.1, parents=8)
            R = TrapOracle(lm).query(X)
            hX.append(X); hR.extend(R.tolist())
        best_ada = float(np.max(hR))

        # PRICE-RL full
        cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed)
        algo = PriceRL(N, 2, TrapOracle(lm), cfg)
        algo.run()
        best_price = max(L.best_so_far for L in algo.logs)
        cos_min = float(min(L.cos_pooled_vs_decomp for L in algo.logs))
        mean_rho = float(np.mean([L.rho for L in algo.logs[1:]]))

        # PRICE-RL (no adaptive)
        cfg_f = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed,
                              fix_alpha_S=0.5, fix_alpha_T=0.5)
        algo_f = PriceRL(N, 2, TrapOracle(lm), cfg_f)
        algo_f.run()
        best_fixed = max(L.best_so_far for L in algo_f.logs)

        rows.append({"seed": seed, "best_random": best_rand,
                     "best_adalead": best_ada, "best_price_rl": best_price,
                     "best_price_fixed": best_fixed, "cos_T1_min": cos_min,
                     "mean_rho": mean_rho})
        print(f"seed={seed} rand={best_rand:.3f} ada={best_ada:.3f} "
              f"price={best_price:.3f} price-fix={best_fixed:.3f} "
              f"cosT1≥{cos_min:.4f} meanρ={mean_rho:.3f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nE5 done in {time.time()-t0:.1f}s")
    for col in ["best_random", "best_adalead", "best_price_fixed", "best_price_rl"]:
        vals = [r[col] for r in rows]
        print(f"  {col:18s} {np.mean(vals):.3f}±{np.std(vals):.3f}")


if __name__ == "__main__":
    main()
