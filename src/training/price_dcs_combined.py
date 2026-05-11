# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Combined PRICE-RL + δ-CS-style proxy-uncertainty clipping.
The discussion section claims they're complementary; this lets us
test that claim empirically.

Mechanism: at each round PRICE-RL computes g_S, g_T as usual; before
applying the update, we look up the proxy-ensemble σ on the policy's
sampled batch and *clip* α_S, α_T proportional to (1 − σ_normalised).
High-uncertainty regions ⇒ smaller step (δ-CS-style conservatism);
low-uncertainty regions ⇒ full PRICE-RL step.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from .autocorr import estimate_autocorr_length, rho_star_from_L
from .controller import PIConfig, PriceController
from .decomposed_gradient import (compute_decomposed_gradient, cosine_grad,
                                  support_mask, support_threshold)
from ..models.policy import FactorisedCategoricalPolicy


@dataclass
class HybridLog:
    round: int
    rho: float
    proxy_R: float
    true_R: float
    reward_gap: float
    proxy_unc_mean: float
    alpha_scale: float


class PriceDCSHybrid:
    """PRICE-RL with δ-CS-style proxy-uncertainty clipping of α_S, α_T."""

    def __init__(self, length: int, alphabet: int, surrogate, ensemble,
                 oracle_true, batch: int = 64, rounds: int = 30,
                 base_lr: float = 2.0, seed: int = 0,
                 device: str = "cpu", clip_strength: float = 1.0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.policy = FactorisedCategoricalPolicy(length, alphabet).to(device)
        self.ctrl = PriceController(PIConfig(base_lr=base_lr))
        self.surr = surrogate
        self.ensemble = ensemble
        self.oracle_true = oracle_true
        self.batch = batch
        self.rounds = rounds
        self.length = length
        self.alphabet = alphabet
        self.device = device
        self.clip_strength = clip_strength
        self.logs: list[HybridLog] = []

    def run(self) -> list[HybridLog]:
        from .surrogate import proxy_score, proxy_uncertainty
        for t in range(self.rounds):
            policy_old = copy.deepcopy(self.policy).eval()
            samples_t, _ = self.policy.sample(self.batch)
            X = samples_t.detach().cpu().numpy()
            proxy_R = proxy_score(self.surr, X)
            true_R = self.oracle_true(X)
            unc = proxy_uncertainty(self.ensemble, X)
            R_t = torch.from_numpy(proxy_R.astype(np.float32))
            with torch.no_grad():
                logp_old = policy_old.log_prob(samples_t)
                ref, _ = policy_old.sample(self.batch)
                logp_ref = policy_old.log_prob(ref)
                thr = support_threshold(logp_ref, q=0.05)
                in_mask = support_mask(logp_old, thr)
            self.policy.zero_grad()
            decomp = compute_decomposed_gradient(self.policy, samples_t, R_t,
                                                 logp_old, in_mask)
            rho_star = 0.5
            alpha_S, alpha_T = self.ctrl.step(decomp.rho, rho_star)

            # δ-CS clipping: scale step by (1 - σ̂) where σ̂ ∈ [0,1]
            sigma_hat = float(unc.mean() / max(unc.max(), 1e-6))
            scale = max(0.05, 1.0 - self.clip_strength * sigma_hat)
            alpha_S *= scale
            alpha_T *= scale

            with torch.no_grad():
                for p, gS, gT in zip(self.policy.parameters(),
                                     decomp.g_S, decomp.g_T):
                    p.data -= alpha_S * gS + alpha_T * gT

            self.logs.append(HybridLog(
                round=t, rho=float(decomp.rho),
                proxy_R=float(proxy_R.mean()),
                true_R=float(true_R.mean()),
                reward_gap=float(proxy_R.mean() - true_R.mean()),
                proxy_unc_mean=float(unc.mean()),
                alpha_scale=scale,
            ))
        return self.logs
