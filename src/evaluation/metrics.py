# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""Active-learning metrics (RESEARCH_PLAN.md §7.1)."""
from __future__ import annotations

import numpy as np


def best_fitness(history_R: list[float]) -> float:
    return float(np.max(history_R)) if history_R else float("-inf")


def top_k_recovery(history_R: list[float], all_R: np.ndarray, k_frac: float = 0.01) -> float:
    """Fraction of the top-k% sequences (by fitness) that the campaign found."""
    if all_R.size == 0:
        return 0.0
    k = max(1, int(k_frac * all_R.size))
    threshold = np.partition(all_R, -k)[-k]
    return float(np.sum(np.asarray(history_R) >= threshold) / k)


def cumulative_regret(history_R: list[float], optimum: float) -> np.ndarray:
    arr = np.asarray(history_R, dtype=np.float64)
    return np.cumsum(optimum - arr)
