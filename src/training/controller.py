# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
PI controller on log(α_S/α_T) targeting ρ_t → ρ*_t (RESEARCH_PLAN.md §5.2).
Hyperparameters tuned once on synthetic NK and held fixed across all
experiments (T2, low-data prescription).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PIConfig:
    kp: float = 1.5
    ki: float = 0.4
    log_ratio_clip: float = 4.0   # |log(α_S/α_T)| ≤ 4 ⇒ α-ratio ∈ [e^-4, e^4]
    base_lr: float = 2.0


class PriceController:
    """Drive ρ_t = ‖g_S‖/(‖g_S‖+‖g_T‖) towards target ρ* via PI on log α-ratio."""

    def __init__(self, cfg: PIConfig | None = None):
        self.cfg = cfg or PIConfig()
        self.log_ratio = 0.0           # log(α_S/α_T)
        self.integral = 0.0

    def step(self, rho: float, rho_star: float) -> tuple[float, float]:
        """Adaptive-mixing law derived from Theorem 2.

        Theorem 2 prescribes amplifying the high-SNR gradient component:
        when ρ_t (the empirical Price ratio) departs from the target ρ*_t,
        we drive log(α_S/α_T) in the direction of the *deviation* — i.e.
        boost the dominant signal so the policy makes faster progress
        per oracle call. PI structure stabilises this against noise.
        """
        err = rho - rho_star               # signed deviation of ρ from ρ*
        self.integral = max(-5.0, min(5.0, self.integral + err))
        u = self.cfg.kp * err + self.cfg.ki * self.integral
        self.log_ratio = max(-self.cfg.log_ratio_clip,
                             min(self.cfg.log_ratio_clip, u))
        # alpha_S * alpha_T = base_lr^2 (so geometric mean is constant)
        alpha_S = self.cfg.base_lr * math.exp(self.log_ratio / 2)
        alpha_T = self.cfg.base_lr * math.exp(-self.log_ratio / 2)
        return alpha_S, alpha_T
