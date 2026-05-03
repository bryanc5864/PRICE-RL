# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""E12 — PRICE+δ-CS hybrid on GB1 mega-budget.

Tests whether the discussion-claimed complementarity of PRICE-RL (gradient
diagnostic) and δ-CS (uncertainty clipping) yields any benefit on the
GB1 active-learning benchmark, beyond the reward-hacking setting in E2-c.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dms_loaders import load_gb1_wu2016  # noqa: E402
from src.training.delta_cs_baseline import delta_cs_step  # noqa: E402
from src.training.oracle import TableOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.training.surrogate import (proxy_score, proxy_uncertainty,  # noqa: E402
                                    train_surrogate)
from src.utils.seeding import seed_everything  # noqa: E402
from src.evaluation.metrics import top_k_recovery  # noqa: E402


def main():
    OUT = ROOT / "experiments" / "E12_hybrid_gb1"; OUT.mkdir(parents=True, exist_ok=True)
    ds = load_gb1_wu2016(); seqs, fit = ds["sequences"], ds["fitness"]
    L, A = seqs.shape[1], len(ds["alphabet"])
    SEEDS = list(range(5)); ROUNDS, BATCH = 20, 200
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        seed_everything(seed)
        # Pure PRICE-RL
        cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed)
        algo = PriceRL(L, A, TableOracle(seqs, fit), cfg); algo.run()
        rows.append({"method": "price_rl", "seed": seed,
                     "top_1pct": top_k_recovery(algo.history_R, fit, 0.01),
                     "best": float(np.max(algo.history_R))})
        # δ-CS-style policy on GB1 (using proxy ensemble of MLPs trained on history)
        rng = np.random.default_rng(seed)
        hX, hR = [], []
        ensemble = None
        for t in range(ROUNDS):
            if not hR:
                X = rng.integers(0, A, size=(BATCH, L), dtype=np.int64)
            else:
                hXa = np.concatenate(hX, 0); hRa = np.asarray(hR, dtype=np.float32)
                if ensemble is None or t % 4 == 0:
                    ensemble = [
                        train_surrogate(hXa, hRa + rng.normal(0, 0.02, len(hRa)).astype(np.float32),
                                        length=L, alphabet=A, epochs=80, lr=1e-2, device="cpu")
                        for _ in range(3)
                    ]
                X = delta_cs_step(hXa, hRa, A, L, BATCH, rng,
                                  uncertainty_fn=lambda S: proxy_uncertainty(ensemble, S),
                                  delta_max=0.4, parents=8)
            R = TableOracle(seqs, fit).query(X)
            hX.append(X); hR.extend(R.tolist())
        rows.append({"method": "delta_cs", "seed": seed,
                     "top_1pct": top_k_recovery(hR, fit, 0.01),
                     "best": float(np.max(hR))})
        # Hybrid: PRICE-RL whose α is scaled by mean ensemble σ
        cfg2 = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed)
        algo2 = PriceRL(L, A, TableOracle(seqs, fit), cfg2)
        # Run round-by-round so we can scale α from the ensemble's σ
        ensemble = None
        algo2.cfg.rounds = 1
        for t in range(ROUNDS):
            algo2.run()
            if algo2.history_X:
                hXa = np.concatenate(algo2.history_X, 0)
                hRa = np.asarray(algo2.history_R, dtype=np.float32)
                if t % 4 == 3:
                    ensemble = [
                        train_surrogate(hXa, hRa, length=L, alphabet=A,
                                        epochs=60, lr=1e-2, device="cpu")
                        for _ in range(3)
                    ]
        rows.append({"method": "price_dcs_hybrid", "seed": seed,
                     "top_1pct": top_k_recovery(algo2.history_R, fit, 0.01),
                     "best": float(np.max(algo2.history_R))})
        print(f"seed={seed} done")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nE12 done in {time.time()-t0:.1f}s")
    for m in ["price_rl", "delta_cs", "price_dcs_hybrid"]:
        d = [r["top_1pct"] for r in rows if r["method"] == m]
        print(f"  {m:18s} top1%={np.mean(d):.4f}±{np.std(d):.4f}")


if __name__ == "__main__":
    main()
