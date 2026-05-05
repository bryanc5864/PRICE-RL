# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E11 — Wallclock / throughput benchmark.
Profiles per-round cost across (length L, alphabet A, batch B). Substantiates
the reproducibility claim that PRICE-RL's full campaign fits in minutes."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nk_landscape import NKConfig, NKLandscape  # noqa: E402
from src.training.oracle import NKOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def main():
    OUT = ROOT / "experiments" / "E11_throughput"; OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cfgs = [
        ("toy",    10, 4, 32),
        ("small",  20, 4, 64),
        ("med",    40, 4, 128),
        ("large",  80, 4, 256),
        ("aa20",   20, 20, 64),
        ("aa40",   40, 20, 64),
        ("dna600", 600, 4, 64),
    ]
    print("Profiling PRICE-RL per-round wallclock:")
    for name, N, A, B in cfgs:
        seed_everything(0)
        if N <= 80:
            nk = NKLandscape(NKConfig(N=N, K=min(N - 1, 5), alphabet=A, seed=0))
            oracle = NKOracle(nk)
        else:
            # Synthetic Gaussian-random oracle for L=600 (DNA-scale)
            class FakeOracle:
                calls = 0; misses = 0
                def query(self, X):
                    self.calls += X.shape[0]
                    return np.random.default_rng(0).random(X.shape[0])
            oracle = FakeOracle()
        cfg = PriceRLConfig(rounds=4, batch=B, inner_steps=4, seed=0)
        # warm-up
        PriceRL(N, A, oracle, cfg).run()
        t0 = time.time()
        for _ in range(3):
            PriceRL(N, A, oracle, cfg).run()
        wall = (time.time() - t0) / 3 / 4  # per-round seconds
        rows.append({"name": name, "N": N, "A": A, "batch": B,
                     "rounds_sec": wall})
        print(f"  {name:>8s} N={N:>4d} A={A:>2d} B={B:>4d}  per-round = {wall*1000:.1f} ms")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    total = sum(r["rounds_sec"] for r in rows)
    print(f"\nTotal per-round wall across configs: {total*1000:.1f} ms")


if __name__ == "__main__":
    main()
