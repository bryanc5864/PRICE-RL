# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Experiment E3 — NK landscape sweep (RESEARCH_PLAN.md §7.1).
Tests Theorem 1 (cosine), Theorem 2 (regret-vs-L), Hypothesis H3 (theory-
empirics match), thresholds T1, T2, T3 and soft-threshold S2.
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


def run_baseline(name: str, alphabet: int, length: int, oracle, rounds: int,
                 batch: int, seed: int) -> tuple[list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    history_X: list[np.ndarray] = []
    history_R: list[float] = []
    for t in range(rounds):
        if name == "random" or len(history_R) == 0:
            X = random_sampler(alphabet, length, batch, rng)
        elif name == "adalead":
            hX = np.concatenate(history_X, 0)
            hR = np.asarray(history_R)
            X = adalead(hX, hR, alphabet, length, batch, rng)
        R = oracle.query(X)
        history_X.append(X)
        history_R.extend(R.tolist())
    return history_R, history_X


def main():
    OUT = ROOT / "experiments" / "E3_nk_sweep"
    OUT.mkdir(parents=True, exist_ok=True)

    K_VALUES = [0, 1, 3, 5, 10, 15, 19]
    SEEDS = [0, 1, 2, 3, 4]
    N = 20
    ALPHA = 4
    ROUNDS = 8
    BATCH = 64

    rows = []
    per_round = []
    t0 = time.time()
    for K in K_VALUES:
        for seed in SEEDS:
            seed_everything(seed)
            nk = NKLandscape(NKConfig(N=N, K=K, alphabet=ALPHA, seed=seed))
            L_analytic = nk.autocorr_length_analytic()
            oracle = NKOracle(nk)

            # ----- PRICE-RL -------------------------------------------
            cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=2,
                                seed=seed)
            algo = PriceRL(N, ALPHA, oracle, cfg)
            logs = algo.run()
            best_price = max(l.best_so_far for l in logs)
            mean_rho = float(np.mean([l.rho for l in logs[1:]]))   # skip warm-up
            mean_rho_star = float(np.mean([l.rho_star for l in logs[1:]]))
            cos_min = float(min(l.cos_pooled_vs_decomp for l in logs))
            for l in logs:
                per_round.append({
                    "K": K, "seed": seed, "round": l.round,
                    "best": l.best_so_far, "mean_R": l.mean_R,
                    "rho": l.rho, "rho_star": l.rho_star,
                    "L_hat": l.L_hat, "L_analytic": L_analytic,
                    "alpha_S": l.alpha_S, "alpha_T": l.alpha_T,
                    "cos_T1": l.cos_pooled_vs_decomp,
                    "in_supp_frac": l.in_support_frac,
                })

            # ----- baselines ------------------------------------------
            o2 = NKOracle(NKLandscape(NKConfig(N=N, K=K, alphabet=ALPHA, seed=seed)))
            R_rand, _ = run_baseline("random", ALPHA, N, o2, ROUNDS, BATCH, seed)
            best_rand = float(max(R_rand))

            o3 = NKOracle(NKLandscape(NKConfig(N=N, K=K, alphabet=ALPHA, seed=seed)))
            R_ada, _ = run_baseline("adalead", ALPHA, N, o3, ROUNDS, BATCH, seed)
            best_ada = float(max(R_ada))

            # ----- ablations: fixed mixing, open-loop, random-support --
            cfg_fixed = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=2,
                                      seed=seed, fix_alpha_S=0.5, fix_alpha_T=0.5)
            o4 = NKOracle(NKLandscape(NKConfig(N=N, K=K, alphabet=ALPHA, seed=seed)))
            algo_fix = PriceRL(N, ALPHA, o4, cfg_fixed)
            algo_fix.run()
            best_fixed = max(l.best_so_far for l in algo_fix.logs)

            cfg_open = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=2,
                                     seed=seed, rho_loop_open=True)
            o5 = NKOracle(NKLandscape(NKConfig(N=N, K=K, alphabet=ALPHA, seed=seed)))
            algo_open = PriceRL(N, ALPHA, o5, cfg_open)
            algo_open.run()
            best_open = max(l.best_so_far for l in algo_open.logs)

            cfg_rand = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=2,
                                     seed=seed, random_support=True)
            o6 = NKOracle(NKLandscape(NKConfig(N=N, K=K, alphabet=ALPHA, seed=seed)))
            algo_rs = PriceRL(N, ALPHA, o6, cfg_rand)
            algo_rs.run()
            best_rs = max(l.best_so_far for l in algo_rs.logs)

            rows.append({
                "K": K, "seed": seed,
                "L_analytic": L_analytic,
                "best_price_rl": best_price,
                "best_random": best_rand,
                "best_adalead": best_ada,
                "best_fixed_mix": best_fixed,
                "best_open_loop": best_open,
                "best_rand_support": best_rs,
                "mean_rho": mean_rho,
                "mean_rho_star": mean_rho_star,
                "cos_T1_min": cos_min,
            })
            print(f"K={K:>2d} seed={seed} best PRICE={best_price:.3f} "
                  f"rand={best_rand:.3f} ada={best_ada:.3f} cosT1≥{cos_min:.4f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    with open(OUT / "per_round.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_round[0].keys())
        w.writeheader()
        w.writerows(per_round)

    print(f"\nE3 done in {time.time()-t0:.1f}s. "
          f"Outputs: {OUT/'summary.csv'} & per_round.csv")


if __name__ == "__main__":
    main()
