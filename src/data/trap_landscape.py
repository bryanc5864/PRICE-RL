# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Trap-5 deceptive landscape (Goldberg, Genetic Algorithms 1989).

Sequences x ∈ {0,1}^N split into N/5 blocks of 5 bits. Per-block fitness:
    block of 5 ones  → 1.0
    j ones (j<5)     → (4 - j)/4    (deceptive ramp away from optimum)

This is the canonical *non-biology* combinatorial optimisation
benchmark: the global optimum is the all-ones string but the gradient
of the average fitness points the *opposite* direction. It is the
benchmark on which any general RL algorithm has to demonstrate it can
escape deception. We use it as Exp 5 (cross-domain validation): the
Price decomposition should still apply, the diagnostic ρ_t should be
informative, and the regret bound should hold qualitatively.
"""
from __future__ import annotations

import numpy as np


class TrapKLandscape:
    def __init__(self, N: int = 30, K: int = 5):
        assert N % K == 0
        self.N = N
        self.K = K

    def fitness_block(self, j_ones: int) -> float:
        if j_ones == self.K:
            return 1.0
        return (self.K - 1 - j_ones) / max(1, self.K - 1)

    def fitness(self, x: np.ndarray) -> float:
        blocks = x.reshape(-1, self.K)
        return float(np.mean([self.fitness_block(int(b.sum())) for b in blocks]))

    def fitness_batch(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([self.fitness(x) for x in X], dtype=np.float64)

    def autocorr_length(self) -> float:
        """Heuristic — block size sets correlation length on Trap-K."""
        return float(self.K)
