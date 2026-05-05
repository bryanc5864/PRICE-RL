# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E3-L — Large-scale NK sweep with bootstrap CIs.
N=40, K∈{0,1,3,5,8,12,18,25,32,39}, 10 seeds, 16 rounds, batch 96.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nk_landscape import NKConfig, NKLandscape  # noqa: E402
from src.training.baselines import adalead, random_sampler  # noqa: E402
from src.training.oracle import NKOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def baseline(name, A, L, oracle, rounds, batch, seed):
    rng = np.random.default_rng(seed)
    hX, hR = [], []
    for t in range(rounds):
        if name == "random" or len(hR) == 0:
            X = random_sampler(A, L, batch, rng)
        else:
            X = adalead(np.concatenate(hX, 0), np.asarray(hR), A, L, batch, rng)
        R = oracle.query(X)
        hX.append(X); hR.extend(R.tolist())
    return hR


def main():
    OUT = ROOT / "experiments" / "E3L_nk_large"
    OUT.mkdir(parents=True, exist_ok=True)
    K_VALUES = [0, 1, 3, 5, 8, 12, 18, 25, 32, 39]
    SEEDS = list(range(10))
    N, A, ROUNDS, BATCH = 40, 4, 16, 96
    rows, per_round = [], []
    t0 = time.time()
    for K in K_VALUES:
        for seed in SEEDS:
            seed_everything(seed)
            nk = NKLandscape(NKConfig(N=N, K=K, alphabet=A, seed=seed))
            L_an = nk.autocorr_length_analytic()
            cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=4, seed=seed)
            algo = PriceRL(N, A, NKOracle(nk), cfg)
            algo.run()
            best_pr = max(L.best_so_far for L in algo.logs)
            cos_min = float(min(L.cos_pooled_vs_decomp for L in algo.logs))
            mean_rho = float(np.mean([L.rho for L in algo.logs[1:]]))
            for L_ in algo.logs:
                per_round.append({"K": K, "seed": seed, "round": L_.round,
                                  "best": L_.best_so_far, "rho": L_.rho,
                                  "rho_star": L_.rho_star, "L_hat": L_.L_hat,
                                  "L_analytic": L_an, "cos_T1": L_.cos_pooled_vs_decomp})

            R_rd = baseline("random", A, N, NKOracle(NKLandscape(NKConfig(N=N, K=K, alphabet=A, seed=seed))),
                            ROUNDS, BATCH, seed)
            R_ad = baseline("adalead", A, N, NKOracle(NKLandscape(NKConfig(N=N, K=K, alphabet=A, seed=seed))),
                            ROUNDS, BATCH, seed)
            cfg_f = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=4, seed=seed,
                                  fix_alpha_S=0.5, fix_alpha_T=0.5)
            algo_f = PriceRL(N, A, NKOracle(NKLandscape(NKConfig(N=N, K=K, alphabet=A, seed=seed))), cfg_f)
            algo_f.run()
            best_fix = max(L.best_so_far for L in algo_f.logs)

            rows.append({"K": K, "seed": seed, "L_analytic": L_an,
                         "best_price_rl": best_pr, "best_random": float(max(R_rd)),
                         "best_adalead": float(max(R_ad)), "best_fixed": best_fix,
                         "mean_rho": mean_rho, "cos_T1_min": cos_min})
            print(f"K={K:>2d} seed={seed:>2d} price={best_pr:.3f} ada={max(R_ad):.3f} "
                  f"rd={max(R_rd):.3f} fix={best_fix:.3f} cos≥{cos_min:.4f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with open(OUT / "per_round.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_round[0].keys()); w.writeheader(); w.writerows(per_round)
    print(f"\nE3-L done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
