# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
A faithful δ-CS-style baseline (Kim et al., ICML 2025).
The current state of the art for active-learning RL in biology uses
proxy-uncertainty to clip the per-step exploration radius.

We implement a stripped-down version: at each round, query the proxy
ensemble to estimate per-candidate uncertainty σ. Use σ to set a
*conservative* edit budget — high-uncertainty regions allow fewer edits
than low-uncertainty ones. This is the same mechanism δ-CS uses, just
without the discrete-diffusion backbone (we apply it to a token-edit
search comparable to AdaLead).

This lets us compare PRICE-RL head-to-head on the reward-hacking
experiment (E2) rather than only against a passive proxy-σ tracker.
"""
from __future__ import annotations

import numpy as np


def delta_cs_step(history_X: np.ndarray, history_R: np.ndarray,
                  alphabet: int, length: int, batch: int,
                  rng: np.random.Generator,
                  uncertainty_fn,                # f(seqs) -> per-seq σ
                  delta_max: float = 0.5,
                  parents: int = 8) -> np.ndarray:
    """δ-CS-style proposal: edit `parents` best candidates with a per-edit
    rate δ that *shrinks* as proxy uncertainty grows.
    """
    if history_X.shape[0] == 0:
        return rng.integers(0, alphabet, size=(batch, length), dtype=np.int64)
    top = np.argsort(history_R)[::-1][:parents]
    parents_X = history_X[top]
    out = np.zeros((batch, length), dtype=np.int64)
    # Estimate uncertainty around each parent.
    unc = uncertainty_fn(parents_X)
    unc_norm = (unc - unc.min()) / (unc.max() - unc.min() + 1e-9)
    for i in range(batch):
        j = rng.integers(0, len(parents_X))
        p = parents_X[j].copy()
        # δ shrinks linearly with uncertainty.
        delta = delta_max * (1.0 - unc_norm[j])
        delta = max(0.01, delta)
        mask = rng.random(length) < delta
        n_mut = int(mask.sum())
        if n_mut > 0:
            p[mask] = rng.integers(0, alphabet, size=n_mut, dtype=np.int64)
        out[i] = p
    return out
