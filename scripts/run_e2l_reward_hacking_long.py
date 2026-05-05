# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E2-L — Long-horizon reward-hacking diagnostic.
30 rounds × 16 seeds × 5-model ensemble. Statistical CIs on lead time."""
from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nk_landscape import NKConfig, NKLandscape  # noqa: E402
from src.training.controller import PIConfig, PriceController  # noqa: E402
from src.training.decomposed_gradient import (compute_decomposed_gradient,  # noqa: E402
                                              support_mask, support_threshold)
from src.training.surrogate import (proxy_score, proxy_uncertainty,  # noqa: E402
                                    train_surrogate)
from src.models.policy import FactorisedCategoricalPolicy  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def run_one(seed: int, rounds: int = 30, batch: int = 64,
            train_n: int = 200, n_ensemble: int = 5) -> list[dict]:
    seed_everything(seed)
    nk = NKLandscape(NKConfig(N=20, K=10, alphabet=4, seed=seed))
    rng = np.random.default_rng(seed)
    train_X = rng.integers(0, 4, size=(train_n, 20), dtype=np.int64)
    train_y = nk.fitness_batch(train_X).astype(np.float32)
    ensemble = [
        train_surrogate(train_X,
                        train_y + rng.normal(0, 0.02, size=train_n).astype(np.float32),
                        length=20, alphabet=4, epochs=200, lr=1e-2, device="cpu")
        for _ in range(n_ensemble)
    ]
    surr = ensemble[0]
    policy = FactorisedCategoricalPolicy(20, 4)
    ctrl = PriceController(PIConfig(base_lr=2.0))
    rows = []
    for t in range(rounds):
        policy_old = copy.deepcopy(policy).eval()
        samples_t, _ = policy.sample(batch)
        samples_np = samples_t.detach().cpu().numpy()
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
        rho_star = 0.5
        alpha_S, alpha_T = ctrl.step(decomp.rho, rho_star)
        with torch.no_grad():
            for p, gS, gT in zip(policy.parameters(), decomp.g_S, decomp.g_T):
                p.data -= alpha_S * gS + alpha_T * gT
        rows.append({"seed": seed, "round": t, "rho": float(decomp.rho),
                     "proxy_R": float(proxy_R.mean()), "true_R": float(true_R.mean()),
                     "reward_gap": float(proxy_R.mean() - true_R.mean()),
                     "proxy_unc_mean": float(unc.mean()),
                     "proxy_unc_max": float(unc.max())})
    return rows


def main():
    OUT = ROOT / "experiments" / "E2L_reward_hacking_long"
    OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = list(range(16))
    all_rows = []
    t0 = time.time()
    for seed in SEEDS:
        rs = run_one(seed)
        all_rows.extend(rs)
        print(f"seed={seed:>2d} final ρ={rs[-1]['rho']:.3f} unc={rs[-1]['proxy_unc_mean']:.3f} "
              f"gap={rs[-1]['reward_gap']:.3f}")
    with open(OUT / "trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys()); w.writeheader(); w.writerows(all_rows)
    print(f"\nE2-L done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
