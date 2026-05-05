# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E21 — Surrogate corruption robustness.
Inject Gaussian noise of increasing std into the proxy reward; track
ρ_t alarm vs. true reward gap. Establishes the signal-vs-noise frontier
of the Price-ratio diagnostic.
"""
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
from src.training.surrogate import (proxy_score, train_surrogate)  # noqa: E402
from src.models.policy import FactorisedCategoricalPolicy  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def main():
    OUT = ROOT / "experiments" / "E21_corruption"; OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = list(range(4))
    NOISE_STDS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    ROUNDS, BATCH = 12, 64
    rows = []
    t0 = time.time()
    for noise_std in NOISE_STDS:
        for seed in SEEDS:
            seed_everything(seed)
            nk = NKLandscape(NKConfig(N=20, K=10, alphabet=4, seed=seed))
            rng = np.random.default_rng(seed)
            train_X = rng.integers(0, 4, size=(200, 20), dtype=np.int64)
            train_y = nk.fitness_batch(train_X).astype(np.float32)
            surr = train_surrogate(train_X,
                                    train_y + rng.normal(0, 0.02, 200).astype(np.float32),
                                    length=20, alphabet=4, epochs=200, lr=1e-2,
                                    device="cpu")
            policy = FactorisedCategoricalPolicy(20, 4)
            ctrl = PriceController(PIConfig(base_lr=2.0))
            seed_rho_alarm = None
            for t in range(ROUNDS):
                policy_old = copy.deepcopy(policy).eval()
                samples_t, _ = policy.sample(BATCH)
                X = samples_t.detach().cpu().numpy()
                proxy_clean = proxy_score(surr, X)
                proxy_R = proxy_clean + rng.normal(0, noise_std, BATCH)
                true_R = nk.fitness_batch(X)
                R_t = torch.from_numpy(proxy_R.astype(np.float32))
                with torch.no_grad():
                    logp_old = policy_old.log_prob(samples_t)
                    ref, _ = policy_old.sample(BATCH)
                    logp_ref = policy_old.log_prob(ref)
                    thr = support_threshold(logp_ref, q=0.05)
                    in_mask = support_mask(logp_old, thr)
                policy.zero_grad()
                decomp = compute_decomposed_gradient(policy, samples_t, R_t,
                                                      logp_old, in_mask)
                alpha_S, alpha_T = ctrl.step(decomp.rho, 0.5)
                with torch.no_grad():
                    for p, gS, gT in zip(policy.parameters(),
                                          decomp.g_S, decomp.g_T):
                        p.data -= alpha_S * gS + alpha_T * gT
                if decomp.rho >= 0.9 and seed_rho_alarm is None:
                    seed_rho_alarm = t
            rows.append({"noise_std": noise_std, "seed": seed,
                          "rho_alarm_round": seed_rho_alarm,
                          "final_rho": float(decomp.rho),
                          "final_gap": float(proxy_R.mean() - true_R.mean())})
            print(f"noise={noise_std:.2f} seed={seed} alarm={seed_rho_alarm} "
                  f"final_rho={decomp.rho:.3f} final_gap={proxy_R.mean() - true_R.mean():.3f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nE21 done in {time.time()-t0:.1f}s")
    import pandas as pd
    df = pd.read_csv(OUT / "summary.csv")
    for ns in NOISE_STDS:
        sub = df[df.noise_std == ns]
        n_alarm = sub.rho_alarm_round.notna().sum()
        gap = sub.final_gap.mean()
        print(f"  noise={ns:.2f}: alarm fired in {n_alarm}/{len(sub)} seeds; "
              f"mean final gap = {gap:.3f}")


if __name__ == "__main__":
    main()
