# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E15 — ρ-gated closed-loop AL: fix the E9 weakness.

E9 showed AdaLead beats PRICE-RL when the policy trains on a retrained
surrogate (because surrogate noise leaks into PRICE-RL's gradient).
Fix: gate PRICE-RL updates with the surrogate's ensemble-σ. When
σ̂ > threshold, freeze the gradient step (keep current policy); when
σ̂ < threshold, apply the standard PRICE-RL update.
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

from src.data.dms_loaders import load_gb1_wu2016  # noqa: E402
from src.training.baselines import adalead, random_sampler  # noqa: E402
from src.training.oracle import TableOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.training.surrogate import (proxy_score, proxy_uncertainty,  # noqa: E402
                                    train_surrogate)
from src.utils.seeding import seed_everything  # noqa: E402
from src.evaluation.metrics import top_k_recovery  # noqa: E402


class GatedSurrogateOracle:
    """Closed-loop oracle: trains an ensemble of surrogates on the
    labelled history; queries return surrogate predictions; ALSO exposes
    per-candidate σ̂ for ρ-gating."""

    def __init__(self, true_oracle: TableOracle, length: int, alphabet: int,
                 n_ensemble: int = 3, retrain_every: int = 1,
                 retrain_epochs: int = 100):
        self.true_oracle = true_oracle
        self.length = length
        self.alphabet = alphabet
        self.n_ensemble = n_ensemble
        self.retrain_every = retrain_every
        self.retrain_epochs = retrain_epochs
        self.history_X: list[np.ndarray] = []
        self.history_R: list[float] = []
        self.ensemble = None
        self._n_calls = 0

    def _retrain(self):
        if not self.history_X:
            return
        X = np.concatenate(self.history_X, axis=0)
        y = np.asarray(self.history_R, dtype=np.float32)
        rng = np.random.default_rng(42)
        self.ensemble = [
            train_surrogate(
                X, y + rng.normal(0, 0.02, len(y)).astype(np.float32),
                length=self.length, alphabet=self.alphabet,
                epochs=self.retrain_epochs, lr=1e-2, device="cpu",
            )
            for _ in range(self.n_ensemble)
        ]

    def query(self, X):
        R_true = self.true_oracle.query(X)
        self.history_X.append(X.copy())
        self.history_R.extend(R_true.tolist())
        self._n_calls += 1
        if self._n_calls % self.retrain_every == 0:
            self._retrain()
        if self.ensemble is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        return proxy_score(self.ensemble[0], X)

    def sigma(self, X):
        if self.ensemble is None:
            return np.ones(X.shape[0])
        return proxy_uncertainty(self.ensemble, X)


def main():
    OUT = ROOT / "experiments" / "E15_rho_gated"; OUT.mkdir(parents=True, exist_ok=True)
    ds = load_gb1_wu2016(); seqs, fit = ds["sequences"], ds["fitness"]
    L, A = seqs.shape[1], len(ds["alphabet"])
    SEEDS = list(range(5)); ROUNDS, BATCH = 5, 100
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        seed_everything(seed)
        srog = GatedSurrogateOracle(TableOracle(seqs, fit), L, A,
                                    n_ensemble=3, retrain_every=1,
                                    retrain_epochs=100)
        rng = np.random.default_rng(seed)
        # Bootstrap round 0
        X0 = random_sampler(A, L, BATCH, rng)
        srog.query(X0)
        # Run PRICE-RL rounds 1..R-1 with σ-gating
        cfg = PriceRLConfig(rounds=ROUNDS - 1, batch=BATCH, inner_steps=8, seed=seed)
        algo = PriceRL(L, A, srog, cfg)
        # Custom inner loop: apply σ-gating
        import copy
        from src.training.decomposed_gradient import (compute_decomposed_gradient,
                                                       support_mask, support_threshold)
        for t in range(cfg.rounds):
            policy_old = copy.deepcopy(algo.policy).eval()
            samples_t, _ = algo.policy.sample(BATCH)
            X = samples_t.detach().cpu().numpy()
            R_proxy = srog.query(X)
            sigma_arr = srog.sigma(X)
            sigma_norm_arr = sigma_arr / max(sigma_arr.max(), 1e-6)
            # Soft per-batch gate: scale = 1 − mean(σ̂_norm), bounded in [0.4, 1.0].
            scale = float(max(0.4, 1.0 - sigma_norm_arr.mean()))
            R_t = torch.from_numpy(R_proxy.astype(np.float32))
            with torch.no_grad():
                logp_old = policy_old.log_prob(samples_t)
                ref, _ = policy_old.sample(BATCH)
                logp_ref = policy_old.log_prob(ref)
                thr = support_threshold(logp_ref, q=0.05)
                in_mask = support_mask(logp_old, thr)
            algo.policy.zero_grad()
            decomp = compute_decomposed_gradient(algo.policy, samples_t, R_t,
                                                  logp_old, in_mask)
            alpha_S, alpha_T = algo.controller.step(decomp.rho, 0.5)
            alpha_S *= scale; alpha_T *= scale
            with torch.no_grad():
                for p, gS, gT in zip(algo.policy.parameters(),
                                      decomp.g_S, decomp.g_T):
                    p.data -= alpha_S * gS + alpha_T * gT
        # Evaluate on TRUE labels
        true_history = srog.history_R
        rows.append({"method": "price_rl_rho_gated", "seed": seed,
                     "best": float(np.max(true_history)),
                     "top_1pct": top_k_recovery(true_history, fit, 0.01)})
        # AdaLead baseline (closed-loop, true labels each round — for fair comparison)
        rng2 = np.random.default_rng(seed)
        X0_b = random_sampler(A, L, BATCH, rng2)
        R0_b = TableOracle(seqs, fit).query(X0_b)
        hX, hR = [X0_b], list(R0_b)
        for t in range(ROUNDS - 1):
            X = adalead(np.concatenate(hX, 0), np.asarray(hR), A, L, BATCH, rng2)
            R = TableOracle(seqs, fit).query(X)
            hX.append(X); hR.extend(R.tolist())
        rows.append({"method": "adalead_closed_loop", "seed": seed,
                     "best": float(np.max(hR)),
                     "top_1pct": top_k_recovery(hR, fit, 0.01)})
        print(f"seed={seed} PRICE-gated={rows[-2]['top_1pct']:.3f} "
              f"ada={rows[-1]['top_1pct']:.3f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nE15 done in {time.time()-t0:.1f}s")
    for m in ["price_rl_rho_gated", "adalead_closed_loop"]:
        d = [r["top_1pct"] for r in rows if r["method"] == m]
        print(f"  {m:24s} top1%={np.mean(d):.4f}±{np.std(d):.4f}")


if __name__ == "__main__":
    main()
