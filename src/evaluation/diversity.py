# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""Diversity-aware metrics.

Biology rarely cares about a single best variant — it cares about a
DIVERSE set of high-fitness candidates that can be screened in parallel.
We add two metrics on top of top-1% recall:

  - top-K-unique: number of distinct sequences in the campaign that lie
    in the top-1% of the labelled landscape (caps recall double-counts).
  - mean-pairwise-Hamming on top-K: average Hamming distance between
    the K highest-fitness candidates found.

Both are used to ablate Phase-A and Phase-B benchmarks for diversity.
"""
from __future__ import annotations

import numpy as np


def top_k_unique(history_X: np.ndarray, history_R: list[float],
                 all_R: np.ndarray, k_frac: float = 0.01) -> int:
    """Number of distinct rows in history_X whose label is in the top-k%."""
    if len(history_R) == 0:
        return 0
    k = max(1, int(k_frac * all_R.size))
    threshold = np.partition(all_R, -k)[-k]
    rows_above = np.asarray(history_R) >= threshold
    if not rows_above.any():
        return 0
    above = history_X[np.where(rows_above)[0]]
    bset = {row.tobytes() for row in above}
    return len(bset)


def mean_pairwise_hamming(history_X: np.ndarray, history_R: list[float],
                          k: int = 10) -> float:
    """Mean Hamming distance among the K highest-fitness sequences."""
    if len(history_R) == 0:
        return 0.0
    R = np.asarray(history_R)
    top_idx = np.argsort(R)[::-1][:k]
    top_X = history_X[top_idx]
    if len(top_X) < 2:
        return 0.0
    n = len(top_X)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float((top_X[i] != top_X[j]).sum())
            count += 1
    return total / max(1, count)
