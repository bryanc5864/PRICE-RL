# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E2-combined — Hybrid PRICE-RL + δ-CS-style uncertainty clipping.
Tests the discussion-section claim that the two are complementary.
Compare three policies on the same surrogate / true landscape:
  PRICE-RL, δ-CS-style, PRICE+δ-CS hybrid."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nk_landscape import NKConfig, NKLandscape  # noqa: E402
from src.training.delta_cs_baseline import delta_cs_step  # noqa: E402
from src.training.price_dcs_combined import PriceDCSHybrid  # noqa: E402
from src.training.surrogate import (proxy_score, proxy_uncertainty,  # noqa: E402
                                    train_surrogate)
from src.utils.seeding import seed_everything  # noqa: E402


def main():
    OUT = ROOT / "experiments" / "E2combined"; OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = list(range(8))
    ROUNDS, BATCH = 24, 64
    all_rows = []
    t0 = time.time()
    for seed in SEEDS:
        seed_everything(seed)
        nk = NKLandscape(NKConfig(N=20, K=10, alphabet=4, seed=seed))
        rng = np.random.default_rng(seed)
        train_X = rng.integers(0, 4, size=(200, 20), dtype=np.int64)
        train_y = nk.fitness_batch(train_X).astype(np.float32)
        ensemble = [
            train_surrogate(train_X,
                            train_y + rng.normal(0, 0.02, size=200).astype(np.float32),
                            length=20, alphabet=4, epochs=200, lr=1e-2, device="cpu")
            for _ in range(3)
        ]
        surr = ensemble[0]
        oracle_true = lambda X: nk.fitness_batch(X)

        # Hybrid PRICE+δ-CS
        algo = PriceDCSHybrid(20, 4, surr, ensemble, oracle_true,
                              batch=BATCH, rounds=ROUNDS, seed=seed,
                              clip_strength=1.0)
        algo.run()
        for L_ in algo.logs:
            all_rows.append({"method": "price_dcs", "seed": seed, "round": L_.round,
                             "rho": L_.rho, "proxy_R": L_.proxy_R, "true_R": L_.true_R,
                             "reward_gap": L_.reward_gap,
                             "proxy_unc_mean": L_.proxy_unc_mean,
                             "alpha_scale": L_.alpha_scale})
        # δ-CS pure
        rng2 = np.random.default_rng(seed)
        hX, hR = [], []
        for t in range(ROUNDS):
            if not hR:
                X = rng2.integers(0, 4, size=(BATCH, 20), dtype=np.int64)
            else:
                X = delta_cs_step(np.concatenate(hX, 0), np.asarray(hR),
                                  4, 20, BATCH, rng2,
                                  uncertainty_fn=lambda S: proxy_uncertainty(ensemble, S))
            pR = proxy_score(surr, X); tR = oracle_true(X)
            unc = proxy_uncertainty(ensemble, X)
            hX.append(X); hR.extend(pR.tolist())
            all_rows.append({"method": "delta_cs", "seed": seed, "round": t,
                             "rho": float("nan"), "proxy_R": float(pR.mean()),
                             "true_R": float(tR.mean()),
                             "reward_gap": float(pR.mean() - tR.mean()),
                             "proxy_unc_mean": float(unc.mean()),
                             "alpha_scale": float("nan")})
        # PRICE-RL pure (for headtohead)
        algo2 = PriceDCSHybrid(20, 4, surr, ensemble, oracle_true,
                               batch=BATCH, rounds=ROUNDS, seed=seed,
                               clip_strength=0.0)
        algo2.run()
        for L_ in algo2.logs:
            all_rows.append({"method": "price_rl", "seed": seed, "round": L_.round,
                             "rho": L_.rho, "proxy_R": L_.proxy_R,
                             "true_R": L_.true_R,
                             "reward_gap": L_.reward_gap,
                             "proxy_unc_mean": L_.proxy_unc_mean,
                             "alpha_scale": 1.0})

        last = lambda m: [r for r in all_rows if r["method"] == m and r["seed"] == seed][-1]
        print(f"seed={seed} | "
              f"PRICE gap={last('price_rl')['reward_gap']:.3f} "
              f"δCS gap={last('delta_cs')['reward_gap']:.3f} "
              f"hybrid gap={last('price_dcs')['reward_gap']:.3f}")

    with open(OUT / "trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader(); w.writerows(all_rows)
    print(f"\nE2-combined done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
