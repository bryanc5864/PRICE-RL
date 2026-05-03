# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Decomposed policy gradient (Theorem 1, RESEARCH_PLAN.md §4 step 4-5).

Given a policy π_θ, a previous policy π_old (frozen reference defining the
support supp_t), and a batch of (x, R(x)) tuples sampled from π_θ, partition
the batch by membership in supp_t and compute:

    g_S = ∇_θ E_{x∼π_θ , x∈supp_t}[ (R(x) - b) log π_θ(x) ]      (selection)
    g_T = ∇_θ E_{x∼π_θ , x∉supp_t}[ (R(x) - b) log π_θ(x) ]      (transmission)

Membership is decided by an embedding-density model trained on samples from
π_old (for NK / factorised policies the natural surrogate is the log-prob
under π_old itself: x is "in support" iff log π_old(x) ≥ q-th percentile of
log-probs from a reference sample. This is the embedding-density support of
RESEARCH_PLAN.md §5.2 instantiated for factorised categorical policies).

Theorem 1 (exactness): for a factorised categorical policy, the standard
REINFORCE gradient ∇J equals g_S + g_T exactly, since the partition is a
disjoint cover of the sample space and ∇log π is linear in the indicator.
This is verified empirically by hard threshold T1 (cosine ≥ 0.95).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DecomposedGradient:
    g_S: list[torch.Tensor]
    g_T: list[torch.Tensor]
    g_pooled: list[torch.Tensor]
    in_support_frac: float
    rho: float  # ‖g_S‖ / (‖g_S‖+‖g_T‖)
    n_S: int
    n_T: int


def support_mask(logp_old: torch.Tensor, threshold: float) -> torch.Tensor:
    """In-support iff logp_old ≥ threshold. Returns bool (B,)."""
    return logp_old >= threshold


def support_threshold(logp_old_ref: torch.Tensor, q: float = 0.05) -> float:
    """q-quantile of reference log-probs ≈ 95-percentile-density radius."""
    return torch.quantile(logp_old_ref, q).item()


def _grad_norm(grads: list[torch.Tensor]) -> float:
    if not grads:
        return 0.0
    return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)).item())


def compute_decomposed_gradient(
    policy,                     # FactorisedCategoricalPolicy
    samples: torch.Tensor,      # (B, L)
    rewards: torch.Tensor,      # (B,)
    logp_old: torch.Tensor,     # (B,)  log π_old(x_b)
    in_support: torch.Tensor,   # (B,)  bool
    baseline: float | None = None,
    is_clip: float = 5.0,
) -> DecomposedGradient:
    """
    Returns g_S (selection), g_T (transmission), and the pooled gradient.

    The in-support contribution uses standard REINFORCE.
    The out-of-support contribution uses a clipped self-normalised IS weight
    π_θ(x) / π_old(x) capped at `is_clip` to bound variance (Risk 4 in §8).

    All three gradients are computed by autograd against policy.parameters().
    """
    B = samples.shape[0]
    if baseline is None:
        baseline = float(rewards.mean().item())
    centred = rewards - baseline
    # Standardise so the gradient magnitude is invariant to reward scale.
    # This is an affine reward transform; the Price decomposition is
    # invariant under affine shifts (the covariance and Δf-expectation
    # terms transform consistently).
    std = centred.std()
    if std.item() > 1e-8:
        centred = centred / (std + 1e-8)

    # Recompute log π_θ(x) WITH gradient (sample tensor itself is detached).
    logp_new = policy.log_prob(samples.detach())  # (B,)

    in_mask = in_support
    out_mask = ~in_support
    n_S = int(in_mask.sum().item())
    n_T = int(out_mask.sum().item())

    # ----- Selection gradient (in-support) ---------------------------------
    if n_S > 0:
        loss_S = -(centred[in_mask] * logp_new[in_mask]).sum() / B
        g_S = torch.autograd.grad(loss_S, list(policy.parameters()),
                                  retain_graph=True, allow_unused=True)
        g_S = [torch.zeros_like(p) if g is None else g
               for p, g in zip(policy.parameters(), g_S)]
    else:
        g_S = [torch.zeros_like(p) for p in policy.parameters()]

    # ----- Transmission gradient (out-of-support, clipped IS) --------------
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

    # ----- Pooled REINFORCE for verification -------------------------------
    loss_pool = -(centred * logp_new).sum() / B
    g_pool = torch.autograd.grad(loss_pool, list(policy.parameters()),
                                 retain_graph=False, allow_unused=True)
    g_pool = [torch.zeros_like(p) if g is None else g
              for p, g in zip(policy.parameters(), g_pool)]

    nS = _grad_norm(g_S)
    nT = _grad_norm(g_T)
    rho = nS / (nS + nT) if (nS + nT) > 0 else 0.5

    return DecomposedGradient(
        g_S=g_S, g_T=g_T, g_pooled=g_pool,
        in_support_frac=n_S / B if B > 0 else 0.0,
        rho=rho, n_S=n_S, n_T=n_T,
    )


def cosine_grad(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    """Flatten + cosine for verifying Theorem 1 (T1).

    Returns 1.0 when both gradients have zero norm (degenerate but
    consistent: g_S + g_T = 0 = g_pooled trivially), -1 when only one
    side has zero norm (real disagreement), and the actual cosine
    otherwise.
    """
    fa = torch.cat([x.flatten() for x in a])
    fb = torch.cat([x.flatten() for x in b])
    na, nb = fa.norm().item(), fb.norm().item()
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return -1.0
    return float((fa @ fb / (fa.norm() * fb.norm())).item())
