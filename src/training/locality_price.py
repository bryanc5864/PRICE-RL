# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Locality-aware PRICE-RL (RESEARCH_PLAN extension §15).

The mega-budget multi-DMS finding (E16) showed AdaLead beats PRICE-RL-WT
1.5–2.5× on long-sequence DMS landscapes because AdaLead's mutate-from-
best operator induces locality that PRICE-RL's diffuse policy lacks. We
close the gap by re-weighting PRICE-RL's gradient with a per-candidate
locality kernel:

    w_loc(x) = exp(-d_H(x, x*) / r),

where x* is the best-so-far variant and r is the locality radius. The
gradient becomes
    g_S^local = E[ w_loc(x) (R(x) - b) ∇log π(x) | x ∈ supp_t ]
    g_T^local = E[ w_loc(x) (R(x) - b) ∇log π(x) | x ∉ supp_t ]

The selection / transmission decomposition is preserved exactly; only
the per-sample weight changes. ρ_t is computed from the locality-
weighted gradient norms, so the diagnostic still works.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from .autocorr import estimate_autocorr_length, rho_star_from_L
from .controller import PIConfig, PriceController
from .decomposed_gradient import (cosine_grad, support_mask, support_threshold,
                                  _grad_norm)
from ..models.policy import FactorisedCategoricalPolicy


@dataclass
class LocalityConfig:
    radius: float = 8.0          # Hamming radius scale
    use_top_k_centre: int = 4     # average over top-K best variants for centre


def hamming_to_centres(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Min Hamming distance from each row of X to any centre."""
    if centres.shape[0] == 0:
        return np.zeros(X.shape[0])
    d = (X[:, None, :] != centres[None, :, :]).sum(axis=-1)
    return d.min(axis=1)


def locality_weights(X: np.ndarray, centres: np.ndarray, radius: float) -> np.ndarray:
    if centres.shape[0] == 0:
        return np.ones(X.shape[0])
    d = hamming_to_centres(X, centres)
    return np.exp(-d / max(1.0, radius))


def compute_decomposed_gradient_local(policy, samples: torch.Tensor,
                                       rewards: torch.Tensor,
                                       logp_old: torch.Tensor,
                                       in_support: torch.Tensor,
                                       loc_w: torch.Tensor,
                                       baseline: float | None = None,
                                       is_clip: float = 5.0):
    """Locality-weighted decomposed gradient. ρ_t computed from norms."""
    B = samples.shape[0]
    if baseline is None:
        baseline = float(rewards.mean().item())
    centred = (rewards - baseline) * loc_w
    std = centred.std()
    if std.item() > 1e-8:
        centred = centred / (std + 1e-8)

    logp_new = policy.log_prob(samples.detach())
    in_mask = in_support
    out_mask = ~in_support
    n_S = int(in_mask.sum().item())
    n_T = int(out_mask.sum().item())

    if n_S > 0:
        loss_S = -(centred[in_mask] * logp_new[in_mask]).sum() / B
        g_S = torch.autograd.grad(loss_S, list(policy.parameters()),
                                  retain_graph=True, allow_unused=True)
        g_S = [torch.zeros_like(p) if g is None else g
               for p, g in zip(policy.parameters(), g_S)]
    else:
        g_S = [torch.zeros_like(p) for p in policy.parameters()]

    if n_T > 0:
        with torch.no_grad():
            log_ratio = logp_new[out_mask] - logp_old[out_mask]
            ratio = torch.clamp(torch.exp(log_ratio), max=is_clip)
        loss_T = -(ratio * centred[out_mask] * logp_new[out_mask]).sum() / B
        g_T = torch.autograd.grad(loss_T, list(policy.parameters()),
                                  retain_graph=True, allow_unused=True)
        g_T = [torch.zeros_like(p) if g is None else g
               for p, g in zip(policy.parameters(), g_T)]
    else:
        g_T = [torch.zeros_like(p) for p in policy.parameters()]

    loss_pool = -(centred * logp_new).sum() / B
    g_pool = torch.autograd.grad(loss_pool, list(policy.parameters()),
                                 retain_graph=False, allow_unused=True)
    g_pool = [torch.zeros_like(p) if g is None else g
              for p, g in zip(policy.parameters(), g_pool)]

    nS, nT = _grad_norm(g_S), _grad_norm(g_T)
    rho = nS / (nS + nT) if (nS + nT) > 0 else 0.5
    return g_S, g_T, g_pool, rho, n_S, n_T


class LocalityPriceRL:
    """PRICE-RL with locality-weighted gradients."""

    def __init__(self, length: int, alphabet: int, oracle, cfg, locality_cfg: LocalityConfig):
        self.cfg = cfg
        self.lcfg = locality_cfg
        self.oracle = oracle
        self.length = length
        self.alphabet = alphabet
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.policy = FactorisedCategoricalPolicy(
            length, alphabet, wildtype=cfg.wildtype, wt_strength=cfg.wt_strength,
        ).to(cfg.device)
        self.controller = PriceController(PIConfig(base_lr=cfg.base_lr))
        self.history_X: list[np.ndarray] = []
        self.history_R: list[float] = []
        self.logs = []

    def _centres(self) -> np.ndarray:
        if not self.history_R:
            return np.zeros((0, self.length), dtype=np.int64)
        all_X = np.concatenate(self.history_X, axis=0)
        all_R = np.asarray(self.history_R)
        k = min(self.lcfg.use_top_k_centre, len(all_R))
        top_idx = np.argsort(all_R)[::-1][:k]
        return all_X[top_idx]

    def run(self):
        for t in range(self.cfg.rounds):
            policy_old = copy.deepcopy(self.policy).eval()
            samples_t, _ = self.policy.sample(self.cfg.batch)
            X = samples_t.detach().cpu().numpy()
            R = self.oracle.query(X)
            self.history_X.append(X.copy())
            self.history_R.extend(R.tolist())

            centres = self._centres()
            loc = locality_weights(X, centres, self.lcfg.radius)
            loc_t = torch.from_numpy(loc.astype(np.float32))
            R_t = torch.from_numpy(R.astype(np.float32))

            with torch.no_grad():
                logp_old = policy_old.log_prob(samples_t)
                ref, _ = policy_old.sample(self.cfg.batch)
                logp_ref = policy_old.log_prob(ref)
                thr = support_threshold(logp_ref, q=self.cfg.support_quantile)
                in_mask = support_mask(logp_old, thr)

            self.policy.zero_grad()
            g_S, g_T, g_pool, rho, n_S, n_T = compute_decomposed_gradient_local(
                self.policy, samples_t, R_t, logp_old, in_mask, loc_t,
                is_clip=self.cfg.is_clip,
            )
            cos_pool = cosine_grad([gs + gt for gs, gt in zip(g_S, g_T)], g_pool)

            seq_hist = np.concatenate(self.history_X, axis=0)
            R_hist = np.asarray(self.history_R)
            L_hat = estimate_autocorr_length(seq_hist, R_hist, d_max=self.length)
            rho_star = rho_star_from_L(L_hat, self.length)
            alpha_S, alpha_T = self.controller.step(rho, rho_star)

            with torch.no_grad():
                for p, gS, gT in zip(self.policy.parameters(), g_S, g_T):
                    p.data -= alpha_S * gS + alpha_T * gT

            self.logs.append({
                "round": t,
                "best_so_far": float(np.max(self.history_R)),
                "mean_R": float(R.mean()),
                "rho": float(rho), "rho_star": float(rho_star),
                "L_hat": float(L_hat), "cos_T1": float(cos_pool),
                "n_S": n_S, "n_T": n_T,
                "mean_locality": float(loc.mean()),
            })
        return self.logs
