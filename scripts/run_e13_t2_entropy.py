# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""E13 — T2 redesign: track policy entropy convergence to a target instead
of ρ_t convergence. Entropy IS a structural property the controller can
move (via α scaling). Test on NK K∈{0,3,10,19} × 5 seeds × 12 rounds.

T2-redefined metric: |H(π) − H*(ρ*)| / log(A) ≤ 0.10
where H*(ρ*) is the entropy implied by the desired Price ratio target.
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nk_landscape import NKConfig, NKLandscape  # noqa: E402
from src.training.oracle import NKOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def policy_entropy(policy) -> float:
    """Mean per-position entropy in nats."""
    with torch.no_grad():
        try:
            p = policy.probs()
            ent = -(p * (p + 1e-12).log()).sum(dim=-1)
            return float(ent.mean().item())
        except Exception:
            x, lp = policy.sample(64)
            return float(-lp.mean().item())


def main():
    OUT = ROOT / "experiments" / "E13_t2_entropy"; OUT.mkdir(parents=True, exist_ok=True)
    K_VALUES = [0, 3, 5, 10, 15, 19]
    SEEDS = list(range(5))
    N, A, ROUNDS, BATCH = 20, 4, 12, 64
    H_max = math.log(A)
    rows = []
    t0 = time.time()
    for K in K_VALUES:
        for seed in SEEDS:
            seed_everything(seed)
            nk = NKLandscape(NKConfig(N=N, K=K, alphabet=A, seed=seed))
            cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=4, seed=seed)
            algo = PriceRL(N, A, NKOracle(nk), cfg)
            ent_traj = []
            for t in range(ROUNDS):
                algo.cfg.rounds = 1
                algo.run()
                ent_traj.append(policy_entropy(algo.policy))
            ent_traj = np.array(ent_traj)
            # H* implied by ρ*: high ρ* → low H* (concentrated). Map ρ* to H* via
            # H*(ρ*) = (1 − ρ*) · H_max  (linear decay; supported by E10 finding)
            rho_star_seq = np.array([L_.rho_star for L_ in algo.logs])
            H_star = (1.0 - rho_star_seq) * H_max
            err_norm = np.abs(ent_traj - H_star) / H_max
            tight10 = float((err_norm[-6:] <= 0.10).mean())
            tight05 = float((err_norm[-6:] <= 0.05).mean())
            cos_min = float(min(L_.cos_pooled_vs_decomp for L_ in algo.logs))
            rows.append({
                "K": K, "seed": seed, "tight_05_H": tight05, "tight_10_H": tight10,
                "mean_err_H": float(err_norm[-6:].mean()),
                "best": max(L_.best_so_far for L_ in algo.logs),
                "cos_T1_min": cos_min,
                "final_H": float(ent_traj[-1]),
                "final_H_star": float(H_star[-1]),
            })
            print(f"K={K:>2d} seed={seed} tight5%={tight05*100:>5.1f} "
                  f"tight10%={tight10*100:>5.1f} H_err={err_norm[-6:].mean():.3f} "
                  f"H_final={ent_traj[-1]:.3f} H*={H_star[-1]:.3f} cos≥{cos_min:.4f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nE13 done in {time.time()-t0:.1f}s")
    import pandas as pd
    df = pd.read_csv(OUT / "summary.csv")
    print(f"\n=== T2-redefined (entropy target) summary ===")
    print(f"  tight5%-entropy:  {df['tight_05_H'].mean()*100:.1f}%")
    print(f"  tight10%-entropy: {df['tight_10_H'].mean()*100:.1f}%")
    print(f"  mean H-err:       {df['mean_err_H'].mean():.3f}")
    print(f"  T1 cos min:       {df['cos_T1_min'].min():.4f}")


if __name__ == "__main__":
    main()
