# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
The PRICE-RL active-learning loop.
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
from ..models.ar_policy import ARCategoricalPolicy


@dataclass
class PriceRLConfig:
    rounds: int = 5
    batch: int = 100
    inner_steps: int = 8
    base_lr: float = 2.0
    support_quantile: float = 0.05
    is_clip: float = 5.0
    entropy_coeff: float = 0.01
    seed: int = 0
    device: str = "cpu"
    rho_target_override: float | None = None
    fix_alpha_S: float | None = None      # ablation: set both to fixed value
    fix_alpha_T: float | None = None
    rho_loop_open: bool = False           # ablation: skip PI controller
    random_support: bool = False          # ablation: random partition
    policy_class: str = "factorised"      # "factorised" or "ar"
    ar_hidden: int = 64
    wildtype: list[int] | None = None      # optional WT-aware init
    wt_strength: float = 3.0
    entropy_inject: bool = False          # support-shaping perturbation (T2 fix)
    entropy_inject_gain: float = 0.4
    entropy_target_mode: bool = False     # controller targets policy entropy (T2 redesign)
    rho_gate_threshold: float = 0.95      # disable updates when ρ saturates above this
    surrogate_uncertainty_gate: float = 0.0  # if > 0, scale α by (1 - σ̂)


@dataclass
class RoundLog:
    round: int
    best_so_far: float
    mean_R: float
    rho: float
    rho_star: float
    L_hat: float
    alpha_S: float
    alpha_T: float
    n_S: int
    n_T: int
    cos_pooled_vs_decomp: float
    in_support_frac: float


class PriceRL:
    def __init__(self, length: int, alphabet: int, oracle, cfg: PriceRLConfig):
        self.cfg = cfg
        self.oracle = oracle
        self.length = length
        self.alphabet = alphabet
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        if cfg.policy_class == "factorised":
            self.policy = FactorisedCategoricalPolicy(
                length, alphabet,
                wildtype=cfg.wildtype,
                wt_strength=cfg.wt_strength,
            ).to(cfg.device)
        elif cfg.policy_class == "ar":
            self.policy = ARCategoricalPolicy(length, alphabet, hidden=cfg.ar_hidden).to(cfg.device)
        else:
            raise ValueError(cfg.policy_class)
        self.controller = PriceController(PIConfig(base_lr=cfg.base_lr))
        self.history_X: list[np.ndarray] = []
        self.history_R: list[float] = []
        self.logs: list[RoundLog] = []

    def _sample(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        s, lp = self.policy.sample(B)
        return s.to(self.cfg.device), lp.to(self.cfg.device)

    def _support_mask_for_batch(self, samples: torch.Tensor,
                                policy_old: FactorisedCategoricalPolicy) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (in_support_bool, logp_old)."""
        with torch.no_grad():
            logp_old = policy_old.log_prob(samples)
            ref_samples, _ = policy_old.sample(max(64, samples.shape[0]))
            logp_ref = policy_old.log_prob(ref_samples)
            thr = support_threshold(logp_ref, q=self.cfg.support_quantile)
            in_mask = support_mask(logp_old, thr)
        if self.cfg.random_support:
            # Ablation: random partition with same in-support fraction.
            frac = float(in_mask.float().mean().item()) if in_mask.numel() > 0 else 0.5
            in_mask = torch.from_numpy(self.rng.random(samples.shape[0]) < frac).to(samples.device)
        return in_mask, logp_old

    def _apply(self, grads_S: list[torch.Tensor], grads_T: list[torch.Tensor],
               alpha_S: float, alpha_T: float):
        """θ ← θ - α_S·g_S - α_T·g_T (we negate because the *loss* gradient
        was returned by autograd; we want gradient *ascent* on the policy
        objective, so subtract the loss-gradient in parameter space).
        Entropy bonus is handled separately."""
        with torch.no_grad():
            for p, gS, gT in zip(self.policy.parameters(), grads_S, grads_T):
                p.data -= alpha_S * gS + alpha_T * gT

    def _entropy_step(self, coeff: float):
        if coeff <= 0:
            return
        # AR entropy is no-grad (Monte Carlo); skip the explicit step.
        if self.cfg.policy_class != "factorised":
            return
        self.policy.zero_grad()
        H = self.policy.entropy()
        (-coeff * H).backward()
        with torch.no_grad():
            for p in self.policy.parameters():
                if p.grad is not None:
                    p.data -= self.cfg.base_lr * p.grad

    def run(self) -> list[RoundLog]:
        for t in range(self.cfg.rounds):
            policy_old = copy.deepcopy(self.policy).eval()

            # ----- Acquire candidates and labels ------------------------
            samples_t, _ = self._sample(self.cfg.batch)
            samples_np = samples_t.detach().cpu().numpy()
            R = self.oracle.query(samples_np)
            self.history_X.append(samples_np)
            self.history_R.extend(R.tolist())
            best_so_far = float(np.max(self.history_R))

            # ----- Decomposed gradient ---------------------------------
            R_t = torch.from_numpy(R.astype(np.float32)).to(self.cfg.device)
            in_mask, logp_old = self._support_mask_for_batch(samples_t, policy_old)
            self.policy.zero_grad()
            decomp = compute_decomposed_gradient(
                self.policy, samples_t, R_t, logp_old, in_mask, is_clip=self.cfg.is_clip,
            )
            cos_check = cosine_grad([gs + gt for gs, gt in zip(decomp.g_S, decomp.g_T)],
                                    decomp.g_pooled)

            # ----- Estimate L̂_t and ρ*_t -------------------------------
            seq_hist = np.concatenate(self.history_X, axis=0)
            R_hist = np.asarray(self.history_R)
            L_hat = estimate_autocorr_length(seq_hist, R_hist, d_max=self.length)
            rho_star = (self.cfg.rho_target_override
                        if self.cfg.rho_target_override is not None
                        else rho_star_from_L(L_hat, self.length))

            # ----- Controller → α_S, α_T -------------------------------
            if self.cfg.fix_alpha_S is not None and self.cfg.fix_alpha_T is not None:
                alpha_S, alpha_T = self.cfg.fix_alpha_S, self.cfg.fix_alpha_T
            elif self.cfg.rho_loop_open:
                alpha_S = self.cfg.base_lr * (rho_star / 0.5)
                alpha_T = self.cfg.base_lr * ((1 - rho_star) / 0.5)
            else:
                alpha_S, alpha_T = self.controller.step(decomp.rho, rho_star)

            # ----- Apply update ----------------------------------------
            self._apply(decomp.g_S, decomp.g_T, alpha_S, alpha_T)

            # Optional inner repeats for richer training (kept off by default
            # for low-data regime).
            for _ in range(self.cfg.inner_steps - 1):
                # Recompute on the same batch (no new oracle calls).
                self.policy.zero_grad()
                d2 = compute_decomposed_gradient(
                    self.policy, samples_t, R_t, logp_old, in_mask, is_clip=self.cfg.is_clip,
                )
                self._apply(d2.g_S, d2.g_T, alpha_S, alpha_T)

            self._entropy_step(self.cfg.entropy_coeff)

            # ----- Support-shaping perturbation (T2 fix) ----------------
            # Interpolate logits toward their per-position mean with
            # strength λ · max(0, ρ_t − ρ*_t). This widens the support
            # without zeroing out structure: l ← l − λ·err·(l − mean(l)).
            # Mean-zeroed shift, so it cannot push ρ below the structural
            # floor — only restore it when ρ has saturated above target.
            if self.cfg.entropy_inject and self.cfg.policy_class == "factorised":
                err_pos = max(0.0, decomp.rho - rho_star)
                if err_pos > 0:
                    lam = min(0.9, self.cfg.entropy_inject_gain * err_pos)
                    with torch.no_grad():
                        l = self.policy.logits.data
                        l.sub_(lam * (l - l.mean(dim=-1, keepdim=True)))

            self.logs.append(RoundLog(
                round=t,
                best_so_far=best_so_far,
                mean_R=float(R.mean()),
                rho=float(decomp.rho),
                rho_star=float(rho_star),
                L_hat=float(L_hat),
                alpha_S=float(alpha_S),
                alpha_T=float(alpha_T),
                n_S=int(decomp.n_S),
                n_T=int(decomp.n_T),
                cos_pooled_vs_decomp=float(cos_check),
                in_support_frac=float(decomp.in_support_frac),
            ))
        return self.logs
