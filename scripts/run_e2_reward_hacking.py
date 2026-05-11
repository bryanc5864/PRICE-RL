# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Experiment E2 — Reward-hacking diagnostic via Price ratio (RESEARCH_PLAN.md §7.1).
Tests Hypothesis H2 / hard threshold T5.

Setup: NK landscape (N=20, K=10) is the ground-truth oracle.
A small MLP surrogate is trained on a *deliberately small* sample (1% of
the space, 200 labels) → it has good in-distribution accuracy but
generalises poorly OOD. We run PRICE-RL using the surrogate as reward,
periodically check the *true* fitness, and track:
    rho_t (PRICE-RL Price ratio)
    proxy_uncertainty_t (ensemble σ)
    true_minus_proxy_t (the 'reward gap')

H2 says rho_t saturates ≥ 2 rounds before proxy uncertainty signals trouble.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nk_landscape import NKConfig, NKLandscape  # noqa: E402
from src.training.controller import PriceController, PIConfig  # noqa: E402
from src.training.decomposed_gradient import (compute_decomposed_gradient,  # noqa: E402
                                              support_mask, support_threshold)
from src.training.delta_cs_baseline import delta_cs_step  # noqa: E402
from src.training.surrogate import (MLPRegressor, proxy_score,  # noqa: E402
                                    proxy_uncertainty, train_surrogate)
from src.models.policy import FactorisedCategoricalPolicy  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def run_delta_cs(seed: int, rounds: int, batch: int, ensemble, surr,
                 nk: NKLandscape) -> list[dict]:
    """Run δ-CS-style policy as a head-to-head reward-hacking baseline."""
    rng = np.random.default_rng(seed)
    history_X: list[np.ndarray] = []
    history_R: list[float] = []
    rows = []
    for t in range(rounds):
        if not history_R:
            X = rng.integers(0, 4, size=(batch, 20), dtype=np.int64)
        else:
            X = delta_cs_step(np.concatenate(history_X, 0),
                              np.asarray(history_R), 4, 20, batch, rng,
                              uncertainty_fn=lambda S: proxy_uncertainty(ensemble, S),
                              delta_max=0.4, parents=8)
        proxy_R = proxy_score(surr, X)
        true_R = nk.fitness_batch(X)
        unc = proxy_uncertainty(ensemble, X)
        history_X.append(X)
        history_R.extend(proxy_R.tolist())
        rows.append({"method": "delta_cs", "seed": seed, "round": t,
                     "proxy_R": float(proxy_R.mean()),
                     "true_R": float(true_R.mean()),
                     "reward_gap": float(proxy_R.mean() - true_R.mean()),
                     "proxy_unc_mean": float(unc.mean()),
                     "rho": float("nan")})
    return rows


def run_one(seed: int, rounds: int = 12, batch: int = 64,
            train_n: int = 200) -> list[dict]:
    seed_everything(seed)
    nk = NKLandscape(NKConfig(N=20, K=10, alphabet=4, seed=seed))

    # Sample a small training set for the surrogate.
    rng = np.random.default_rng(seed)
    train_X = rng.integers(0, 4, size=(train_n, 20), dtype=np.int64)
    train_y = nk.fitness_batch(train_X).astype(np.float32)

    ensemble = [
        train_surrogate(train_X, train_y + rng.normal(0, 0.02, size=train_n).astype(np.float32),
                        length=20, alphabet=4, epochs=200, lr=1e-2, device="cpu")
        for _ in range(3)
    ]
    surr = ensemble[0]

    policy = FactorisedCategoricalPolicy(20, 4)
    ctrl = PriceController(PIConfig(base_lr=2.0))

    rows = []
    for t in range(rounds):
        # Save policy_old for support membership.
        import copy
        policy_old = copy.deepcopy(policy).eval()

        samples_t, _ = policy.sample(batch)
        samples_np = samples_t.detach().cpu().numpy()

        # Reward = surrogate prediction (the 'proxy reward').
        proxy_R = proxy_score(surr, samples_np)
        true_R = nk.fitness_batch(samples_np)
        unc = proxy_uncertainty(ensemble, samples_np)

        R_t = torch.from_numpy(proxy_R.astype(np.float32))

        with torch.no_grad():
            logp_old = policy_old.log_prob(samples_t)
            ref, _ = policy_old.sample(batch)
            logp_ref = policy_old.log_prob(ref)
            thr = support_threshold(logp_ref, q=0.05)
            in_mask = support_mask(logp_old, thr)
        policy.zero_grad()
        decomp = compute_decomposed_gradient(policy, samples_t, R_t, logp_old, in_mask)

        # Use a fixed L_hat = N for clarity (not the focus here).
        rho_star = 0.5
        alpha_S, alpha_T = ctrl.step(decomp.rho, rho_star)

        with torch.no_grad():
            for p, gS, gT in zip(policy.parameters(), decomp.g_S, decomp.g_T):
                p.data -= alpha_S * gS + alpha_T * gT

        rows.append({
            "method": "price_rl",
            "seed": seed,
            "round": t,
            "rho": float(decomp.rho),
            "proxy_R": float(proxy_R.mean()),
            "true_R": float(true_R.mean()),
            "reward_gap": float(proxy_R.mean() - true_R.mean()),
            "proxy_unc_mean": float(unc.mean()),
            "proxy_unc_max": float(unc.max()),
            "alpha_S": float(alpha_S),
            "alpha_T": float(alpha_T),
        })
    return rows, ensemble, surr, nk


def main():
    OUT = ROOT / "experiments" / "E2_reward_hacking"
    OUT.mkdir(parents=True, exist_ok=True)

    SEEDS = list(range(8))
    all_rows = []
    t0 = time.time()
    for seed in SEEDS:
        rows, ensemble, surr, nk = run_one(seed)
        all_rows.extend(rows)
        # Head-to-head δ-CS baseline on the same surrogate / true landscape
        rows_dcs = run_delta_cs(seed, rounds=12, batch=64,
                                ensemble=ensemble, surr=surr, nk=nk)
        all_rows.extend(rows_dcs)
        rho = [r["rho"] for r in rows]
        unc_pr = rows[-1]["proxy_unc_mean"]
        unc_dcs = rows_dcs[-1]["proxy_unc_mean"]
        gap_pr = rows[-1]["reward_gap"]
        gap_dcs = rows_dcs[-1]["reward_gap"]
        print(f"seed={seed} | PRICE: ρ={rho[-1]:.3f} unc={unc_pr:.3f} gap={gap_pr:.3f}"
              f"  | δ-CS: unc={unc_dcs:.3f} gap={gap_dcs:.3f}")

    with open(OUT / "trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"E2 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
