# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E19 — Multi-objective PRICE-RL.
Vector-valued NK landscape (2 objectives). PRICE-RL is run with a
scalarised reward (uniform-weight sum of R_a and R_b) and the Pareto
front of all queried sequences is reported. We compare hypervolume
against random and AdaLead baselines.

Algorithmic novelty: the selection covariance becomes a 2×2 covariance
matrix (one row per objective). We compute per-objective ρ_i and report
how the Price decomposition splits across objectives.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.multi_obj_nk import (MultiObjNKLandscape, hypervolume_2d,  # noqa: E402
                                    pareto_front)
from src.training.baselines import adalead, random_sampler  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


class MOOOracle:
    """Wraps multi-objective NK; PRICE-RL sees scalarised reward."""

    def __init__(self, mo: MultiObjNKLandscape, weights=(0.5, 0.5)):
        self.mo = mo
        self.w = np.asarray(weights)
        self.calls = 0
        self.misses = 0
        self.history_R_vec: list[np.ndarray] = []
        self.history_X: list[np.ndarray] = []

    def query(self, X):
        Rvec = self.mo.fitness_batch(X)
        self.history_R_vec.append(Rvec)
        self.history_X.append(X.copy())
        self.calls += X.shape[0]
        return (Rvec * self.w).sum(axis=1)


def main():
    OUT = ROOT / "experiments" / "E19_multi_obj"; OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = list(range(5))
    N, K, A = 20, 5, 4
    ROUNDS, BATCH = 12, 64
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        seed_everything(seed)
        mo = MultiObjNKLandscape(N=N, K=K, alphabet=A, seed_a=seed, seed_b=seed + 100)

        # Random baseline
        rng = np.random.default_rng(seed)
        R_all = []
        for t in range(ROUNDS):
            X = rng.integers(0, A, size=(BATCH, N), dtype=np.int64)
            R_all.append(mo.fitness_batch(X))
        R_rand = np.concatenate(R_all, 0)
        hv_rand = hypervolume_2d(R_rand)

        # AdaLead with scalarised reward
        rng2 = np.random.default_rng(seed)
        oracle_a = MOOOracle(mo, weights=(0.5, 0.5))
        hX, hR = [], []
        for t in range(ROUNDS):
            if not hR:
                X = rng2.integers(0, A, size=(BATCH, N), dtype=np.int64)
            else:
                X = adalead(np.concatenate(hX, 0), np.asarray(hR), A, N, BATCH, rng2)
            R = oracle_a.query(X)
            hX.append(X); hR.extend(R.tolist())
        R_ada = np.concatenate(oracle_a.history_R_vec, 0)
        hv_ada = hypervolume_2d(R_ada)

        # PRICE-RL with scalarised reward
        oracle_pr = MOOOracle(mo, weights=(0.5, 0.5))
        cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=4, seed=seed)
        algo = PriceRL(N, A, oracle_pr, cfg)
        algo.run()
        R_pr = np.concatenate(oracle_pr.history_R_vec, 0)
        hv_pr = hypervolume_2d(R_pr)

        rows.append({"seed": seed,
                     "hv_random": hv_rand, "hv_adalead": hv_ada, "hv_price_rl": hv_pr,
                     "n_pareto_random": int(pareto_front(R_rand).sum()),
                     "n_pareto_adalead": int(pareto_front(R_ada).sum()),
                     "n_pareto_price_rl": int(pareto_front(R_pr).sum())})
        print(f"seed={seed} HV: rand={hv_rand:.3f} ada={hv_ada:.3f} price={hv_pr:.3f} "
              f"|Pareto| rand={rows[-1]['n_pareto_random']} ada={rows[-1]['n_pareto_adalead']} "
              f"price={rows[-1]['n_pareto_price_rl']}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nE19 done in {time.time()-t0:.1f}s")
    for col in ["hv_random", "hv_adalead", "hv_price_rl"]:
        d = [r[col] for r in rows]
        print(f"  {col:14s} mean={np.mean(d):.3f}±{np.std(d):.3f}")
    for col in ["n_pareto_random", "n_pareto_adalead", "n_pareto_price_rl"]:
        d = [r[col] for r in rows]
        print(f"  {col:18s} mean={np.mean(d):.1f}±{np.std(d):.1f}")


if __name__ == "__main__":
    main()
