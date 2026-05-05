# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
Active-learning oracle wrapper. Takes a fitness-table dataset (cached DMS or
NK) and answers queries on int-encoded sequences. For DMS we use a hash-map
lookup; queries on unmeasured variants fall back to the nearest-neighbour
mean (declared explicitly in the loop log so it is never silent).
"""
from __future__ import annotations

import numpy as np


class TableOracle:
    """Lookup oracle from a fitness-labelled table of sequences."""

    def __init__(self, sequences: np.ndarray, fitness: np.ndarray):
        assert sequences.shape[0] == fitness.shape[0]
        self.sequences = sequences
        self.fitness = fitness
        self._index: dict[bytes, float] = {
            s.tobytes(): float(f) for s, f in zip(sequences, fitness)
        }
        self.calls = 0
        self.misses = 0

    def query_one(self, x: np.ndarray) -> float:
        self.calls += 1
        v = self._index.get(x.tobytes())
        if v is None:
            self.misses += 1
            # Nearest-neighbour fallback by Hamming distance.
            d = (self.sequences != x[None, :]).sum(axis=-1)
            j = int(np.argmin(d))
            return float(self.fitness[j])
        return v

    def query(self, X: np.ndarray) -> np.ndarray:
        """Vectorised batch query. First does hash lookup; for misses,
        runs ONE batched (B, N, L) Hamming-distance computation rather
        than per-row scans. Big speedup on long-sequence datasets."""
        B, L = X.shape
        out = np.empty(B, dtype=np.float64)
        miss_idx: list[int] = []
        for i, x in enumerate(X):
            v = self._index.get(x.tobytes())
            if v is None:
                miss_idx.append(i)
            else:
                out[i] = v
        self.calls += B
        if miss_idx:
            self.misses += len(miss_idx)
            Xm = X[miss_idx]                                  # (M, L)
            # Use a chunked broadcast to bound peak memory.
            chunk = max(1, 4_000_000 // max(1, self.sequences.shape[0]))
            best_j = np.empty(len(miss_idx), dtype=np.int64)
            for s in range(0, len(miss_idx), chunk):
                e = min(s + chunk, len(miss_idx))
                d = (self.sequences[None, :, :] != Xm[s:e, None, :]).sum(axis=-1)
                best_j[s:e] = d.argmin(axis=1)
            out[miss_idx] = self.fitness[best_j]
        return out


class NKOracle:
    """Wrap an NKLandscape so it shares the .query interface."""

    def __init__(self, nk):
        self.nk = nk
        self.calls = 0
        self.misses = 0

    def query_one(self, x: np.ndarray) -> float:
        self.calls += 1
        return float(self.nk.fitness(x))

    def query(self, X: np.ndarray) -> np.ndarray:
        return self.nk.fitness_batch(X)
