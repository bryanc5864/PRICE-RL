# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""E1-mega — 40 rounds × 200 queries × 10 seeds on GB1.

Tests whether PRICE-RL's GB1 lead over AdaLead (already non-overlapping
CIs at 4,000 queries) widens at 8,000 queries.
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
from src.training.baselines import adalead, pex, random_sampler  # noqa: E402
from src.training.gflownet_al import (FlowFactorisedPolicy,  # noqa: E402
                                       gflownet_al_step)
from src.training.oracle import TableOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402
from src.evaluation.metrics import top_k_recovery  # noqa: E402


def per_round_baseline(name, alphabet, length, oracle, rounds, batch, seed):
    rng = np.random.default_rng(seed)
    hX, hR = [], []
    pr = []
    for t in range(rounds):
        if name == "random" or not hR:
            X = random_sampler(alphabet, length, batch, rng)
        elif name == "adalead":
            X = adalead(np.concatenate(hX, 0), np.asarray(hR), alphabet, length, batch, rng)
        elif name == "pex":
            X = pex(np.concatenate(hX, 0), np.asarray(hR), alphabet, length, batch, rng)
        R = oracle.query(X)
        hX.append(X); hR.extend(R.tolist())
        pr.append({"round": t, "best_so_far": float(np.max(hR)),
                   "mean_R": float(R.mean())})
    return hR, pr


def gflownet_run(alphabet, length, oracle, rounds, batch, seed):
    seed_everything(seed)
    pol = FlowFactorisedPolicy(length, alphabet)
    hX, hR = [], []
    pr = []
    for t in range(rounds):
        X = gflownet_al_step(pol, np.concatenate(hX, 0) if hX else np.zeros((0, length), dtype=np.int64),
                             np.asarray(hR) if hR else np.zeros(0),
                             batch=batch)
        R = oracle.query(X)
        hX.append(X); hR.extend(R.tolist())
        pr.append({"round": t, "best_so_far": float(np.max(hR)),
                   "mean_R": float(R.mean())})
    return hR, pr


def main():
    OUT = ROOT / "experiments" / "E1mega_gb1"; OUT.mkdir(parents=True, exist_ok=True)
    ds = load_gb1_wu2016(); seqs, fit = ds["sequences"], ds["fitness"]
    L, A = seqs.shape[1], len(ds["alphabet"])
    SEEDS = list(range(10)); ROUNDS, BATCH = 40, 200
    rows, traj = [], []
    t0 = time.time()
    for seed in SEEDS:
        seed_everything(seed)
        for method in ["random", "pex", "adalead"]:
            R, pr = per_round_baseline(method, A, L, TableOracle(seqs, fit),
                                       ROUNDS, BATCH, seed)
            rows.append({"method": method, "seed": seed,
                         "best": float(np.max(R)),
                         "top_1pct": top_k_recovery(R, fit, 0.01)})
            for d in pr: d.update(method=method, seed=seed); traj.append(d)
        # GFlowNet-AL
        R, pr = gflownet_run(A, L, TableOracle(seqs, fit), ROUNDS, BATCH, seed)
        rows.append({"method": "gflownet_al", "seed": seed,
                     "best": float(np.max(R)),
                     "top_1pct": top_k_recovery(R, fit, 0.01)})
        for d in pr: d.update(method="gflownet_al", seed=seed); traj.append(d)

        # PRICE-RL full
        cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed)
        algo = PriceRL(L, A, TableOracle(seqs, fit), cfg); algo.run()
        for L_ in algo.logs:
            traj.append({"method": "price_rl", "seed": seed, "round": L_.round,
                         "best_so_far": L_.best_so_far, "mean_R": L_.mean_R})
        rows.append({"method": "price_rl", "seed": seed,
                     "best": float(np.max(algo.history_R)),
                     "top_1pct": top_k_recovery(algo.history_R, fit, 0.01),
                     "cos_T1_min": float(min(L_.cos_pooled_vs_decomp for L_ in algo.logs))})
        cfg_f = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed,
                              fix_alpha_S=0.5, fix_alpha_T=0.5)
        algo_f = PriceRL(L, A, TableOracle(seqs, fit), cfg_f); algo_f.run()
        for L_ in algo_f.logs:
            traj.append({"method": "price_rl_fixed", "seed": seed, "round": L_.round,
                         "best_so_far": L_.best_so_far, "mean_R": L_.mean_R})
        rows.append({"method": "price_rl_fixed", "seed": seed,
                     "best": float(np.max(algo_f.history_R)),
                     "top_1pct": top_k_recovery(algo_f.history_R, fit, 0.01)})
        print(f"seed={seed} done")

    with open(OUT / "summary.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(OUT / "trajectories.csv", "w", newline="") as f:
        keys = sorted({k for r in traj for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(traj)
    print(f"\nE1-mega done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
