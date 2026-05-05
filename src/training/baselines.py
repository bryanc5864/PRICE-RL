# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
Reproducible baselines for active learning over discrete sequences:
- random: uniform over the alphabet
- AdaLead (Sinai et al. 2020): rollout from current best with edit kappa
- PEX (Ren et al. 2022): proximal exploration with edit-distance regulariser

These implementations are intentionally minimal — exactly what is needed
to compete on the small fixed-budget protocol.
"""
from __future__ import annotations

import numpy as np


def random_sampler(alphabet: int, length: int, batch: int, rng: np.random.Generator):
    return rng.integers(0, alphabet, size=(batch, length), dtype=np.int64)


def adalead(history_X: np.ndarray, history_R: np.ndarray, alphabet: int,
            length: int, batch: int, rng: np.random.Generator,
            kappa: float = 0.05, parents: int = 8) -> np.ndarray:
    """Greedy evolutionary baseline: pick best `parents`, mutate each with
    rate `kappa` per position, return `batch` candidates."""
    if history_X.shape[0] == 0:
        return random_sampler(alphabet, length, batch, rng)
    top = np.argsort(history_R)[::-1][:parents]
    parents_X = history_X[top]
    out = np.zeros((batch, length), dtype=np.int64)
    for i in range(batch):
        p = parents_X[rng.integers(0, len(parents_X))].copy()
        mask = rng.random(length) < kappa
        n_mut = int(mask.sum())
        if n_mut > 0:
            p[mask] = rng.integers(0, alphabet, size=n_mut, dtype=np.int64)
        out[i] = p
    return out


def pex(history_X: np.ndarray, history_R: np.ndarray, alphabet: int,
        length: int, batch: int, rng: np.random.Generator,
        proximal_radius: int = 3, parents: int = 8) -> np.ndarray:
    """Proximal exploration: edits restricted to a small Hamming neighbourhood.

    A faithful 'spirit-of-PEX' implementation: among proposed edits, retain
    only those within the proximal_radius of any visited parent.
    """
    if history_X.shape[0] == 0:
        return random_sampler(alphabet, length, batch, rng)
    top = np.argsort(history_R)[::-1][:parents]
    parents_X = history_X[top]
    out = np.zeros((batch, length), dtype=np.int64)
    for i in range(batch):
        p = parents_X[rng.integers(0, len(parents_X))].copy()
        n_edits = rng.integers(1, proximal_radius + 1)
        positions = rng.choice(length, size=n_edits, replace=False)
        p[positions] = rng.integers(0, alphabet, size=n_edits, dtype=np.int64)
        out[i] = p
    return out
