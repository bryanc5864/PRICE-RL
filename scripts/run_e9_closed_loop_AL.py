# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""E9 — Closed-loop active learning with retrained surrogate on GB1.

Replaces the cached-table oracle with a small MLP surrogate that is
retrained each round on labelled history. This matches the protocol of
all 2025 baselines (δ-CS, μSearch, ALDE) more faithfully than direct
oracle access. PRICE-RL uses the surrogate as proxy reward; the true
oracle is consulted only to evaluate the candidates that PRICE-RL
proposes (matching the wet-lab AL loop).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dms_loaders import load_gb1_wu2016  # noqa: E402
from src.training.baselines import adalead, random_sampler  # noqa: E402
from src.training.oracle import TableOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.training.surrogate import (proxy_score, train_surrogate)  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402
from src.evaluation.metrics import top_k_recovery  # noqa: E402


class SurrogateOracle:
    """Closed-loop AL oracle: every PRICE-RL query consults the true
    oracle to LABEL the candidates (counts toward budget), then returns
    SURROGATE predictions (used as PRICE-RL's training signal). The
    surrogate is retrained from labelled history on a schedule."""

    def __init__(self, true_oracle: TableOracle, length: int, alphabet: int,
                 retrain_every: int = 1, retrain_epochs: int = 150):
        self.true_oracle = true_oracle
        self.length = length
        self.alphabet = alphabet
        self.retrain_every = retrain_every
        self.retrain_epochs = retrain_epochs
        self.history_X: list[np.ndarray] = []
        self.history_R: list[float] = []
        self.surr = None
        self._n_calls = 0

    def _retrain(self):
        if not self.history_X:
            return
        X = np.concatenate(self.history_X, axis=0)
        y = np.asarray(self.history_R, dtype=np.float32)
        self.surr = train_surrogate(X, y, length=self.length,
                                    alphabet=self.alphabet,
                                    epochs=self.retrain_epochs, lr=1e-2,
                                    device="cpu")

    def query(self, X):
        # 1) Consult the true oracle and append to labelled history.
        R_true = self.true_oracle.query(X)
        self.history_X.append(X.copy())
        self.history_R.extend(R_true.tolist())
        self._n_calls += 1
        # 2) Retrain the surrogate on schedule.
        if self._n_calls % self.retrain_every == 0:
            self._retrain()
        # 3) Return surrogate predictions as PRICE-RL's reward signal.
        if self.surr is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        return proxy_score(self.surr, X)


def main():
    OUT = ROOT / "experiments" / "E9_closed_loop_AL"; OUT.mkdir(parents=True, exist_ok=True)
    ds = load_gb1_wu2016(); seqs, fit = ds["sequences"], ds["fitness"]
    L, A = seqs.shape[1], len(ds["alphabet"])
    SEEDS = list(range(5)); ROUNDS, BATCH = 5, 100
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        seed_everything(seed)
        srog = SurrogateOracle(TableOracle(seqs, fit), L, A,
                                retrain_every=1, retrain_epochs=150)
        rng = np.random.default_rng(seed)
        # ---- Round 0: random bootstrap (true labels) ----
        X0 = random_sampler(A, L, BATCH, rng)
        srog.query(X0)                           # labels and trains surrogate
        # ---- Rounds 1..R-1: PRICE-RL on the surrogate-as-reward ----
        cfg = PriceRLConfig(rounds=ROUNDS - 1, batch=BATCH, inner_steps=8, seed=seed)
        algo = PriceRL(L, A, srog, cfg)
        algo.run()
        true_history = srog.history_R
        cos_min = float(min(L_.cos_pooled_vs_decomp for L_ in algo.logs))
        rows.append({"method": "price_rl_closed_loop", "seed": seed,
                     "best": float(np.max(true_history)),
                     "top_1pct": top_k_recovery(true_history, fit, 0.01),
                     "cos_T1_min": cos_min,
                     "n_queries": len(true_history)})
        # AdaLead with same bootstrap (random round 0)
        rng2 = np.random.default_rng(seed)
        X0_b = random_sampler(A, L, BATCH, rng2)
        R0_b = TableOracle(seqs, fit).query(X0_b)
        hX, hR = [X0_b], list(R0_b)
        for t_ in range(ROUNDS - 1):
            X = adalead(np.concatenate(hX, 0), np.asarray(hR), A, L, BATCH, rng2)
            R = TableOracle(seqs, fit).query(X)
            hX.append(X); hR.extend(R.tolist())
        rows.append({"method": "adalead_closed_loop", "seed": seed,
                     "best": float(np.max(hR)),
                     "top_1pct": top_k_recovery(hR, fit, 0.01),
                     "n_queries": len(hR)})
        print(f"seed={seed} PRICE={rows[-2]['top_1pct']:.3f} ada={rows[-1]['top_1pct']:.3f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nE9 done in {time.time()-t0:.1f}s")
    for m in ["price_rl_closed_loop", "adalead_closed_loop"]:
        d = [r["top_1pct"] for r in rows if r["method"] == m]
        print(f"  {m:24s} top1%={np.mean(d):.4f}±{np.std(d):.4f}")


if __name__ == "__main__":
    main()
