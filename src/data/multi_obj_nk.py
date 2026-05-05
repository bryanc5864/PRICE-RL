# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""Multi-objective NK landscape.
Two NK landscapes with different seeds give vector-valued fitness
(R_1, R_2) ∈ [0, 1]^2; the Pareto front is the set of non-dominated
(R_1, R_2) pairs."""
from __future__ import annotations

import numpy as np

from .nk_landscape import NKConfig, NKLandscape


class MultiObjNKLandscape:
    def __init__(self, N: int = 20, K: int = 5, alphabet: int = 4,
                 seed_a: int = 0, seed_b: int = 1):
        self.nk_a = NKLandscape(NKConfig(N=N, K=K, alphabet=alphabet, seed=seed_a))
        self.nk_b = NKLandscape(NKConfig(N=N, K=K, alphabet=alphabet, seed=seed_b))
        self.N = N
        self.K = K
        self.alphabet = alphabet

    def fitness_batch(self, X: np.ndarray) -> np.ndarray:
        Ra = self.nk_a.fitness_batch(X)
        Rb = self.nk_b.fitness_batch(X)
        return np.stack([Ra, Rb], axis=-1)


def pareto_front(R: np.ndarray) -> np.ndarray:
    """R: (N, D) reward matrix. Returns boolean mask of non-dominated rows."""
    N = R.shape[0]
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        # j dominates i if all R[j] >= R[i] and any strictly greater
        dominated = ((R >= R[i]).all(axis=1) & (R > R[i]).any(axis=1))
        if dominated.any():
            keep[i] = False
    return keep


def hypervolume_2d(R: np.ndarray, ref: np.ndarray = np.array([0.0, 0.0])) -> float:
    """2D hypervolume relative to reference point."""
    keep = pareto_front(R)
    P = R[keep]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(-P[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for x, y in P:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return float(max(0.0, hv))
