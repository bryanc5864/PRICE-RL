# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
NK landscape simulator (Kauffman & Weinberger 1989).
Implements the paper (NK datasets) and §7.1 (Experiment E3).

A sequence x ∈ {0,...,A-1}^N has fitness
    f(x) = (1/N) sum_i  table_i[ x_i ; x_{neighbours_i(1)} ; ... ; x_{neighbours_i(K)} ].
Each per-locus contribution is sampled iid Uniform[0,1] keyed by
(seed, locus, joint state), giving a deterministic but rugged landscape.

Autocorrelation length on NK is known analytically:
    rho(d) = (1 - d/N)^(K+1)         (Stadler & Happel 1999).
Equivalently exp(-d/L) with L ≈ N / (K+1) for d/N small. Used as
ground-truth ρ*_t target in Theorem 2 / Experiment E3.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _table_value(seed: int, locus: int, joint_state: tuple[int, ...]) -> float:
    """Deterministic uniform[0,1) keyed by (seed, locus, joint state)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(seed.to_bytes(8, "little", signed=False))
    h.update(locus.to_bytes(4, "little", signed=False))
    for s in joint_state:
        h.update(int(s).to_bytes(2, "little", signed=False))
    n = int.from_bytes(h.digest(), "little")
    return (n & ((1 << 53) - 1)) / float(1 << 53)


@dataclass(frozen=True)
class NKConfig:
    N: int = 20
    K: int = 3
    alphabet: int = 4
    seed: int = 0


class NKLandscape:
    """Cached NK landscape with O(1) fitness queries on sequences."""

    def __init__(self, cfg: NKConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.neighbours = np.zeros((cfg.N, cfg.K), dtype=np.int64)
        for i in range(cfg.N):
            choices = list(range(cfg.N))
            choices.remove(i)
            self.neighbours[i] = rng.choice(choices, size=cfg.K, replace=False)
        self._cache: dict[tuple, float] = {}

    def fitness(self, x: np.ndarray) -> float:
        """Fitness of a single 1-D sequence x (length N, ints in [0, A))."""
        x = np.asarray(x, dtype=np.int64)
        assert x.shape == (self.cfg.N,), x.shape
        N, K = self.cfg.N, self.cfg.K
        total = 0.0
        for i in range(N):
            joint = (int(x[i]),) + tuple(int(x[j]) for j in self.neighbours[i])
            key = (i, joint)
            v = self._cache.get(key)
            if v is None:
                v = _table_value(self.cfg.seed, i, joint)
                self._cache[key] = v
            total += v
        return total / N

    def fitness_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised fitness over a (B,N) array of sequences."""
        return np.asarray([self.fitness(x) for x in X], dtype=np.float64)

    def autocorr_length_analytic(self) -> float:
        """L such that rho(d) ~ exp(-d/L), from Stadler & Happel 1999.

        rho(d) = (1 - d/N)^(K+1).  Define L by linearising at d=0:
        log rho(d) = (K+1) log(1 - d/N) ≈ -(K+1) d / N
        ⇒ L = N / (K+1).
        """
        return self.cfg.N / (self.cfg.K + 1)

    def autocorr_curve(self, d: np.ndarray) -> np.ndarray:
        return (1.0 - d / self.cfg.N) ** (self.cfg.K + 1)
