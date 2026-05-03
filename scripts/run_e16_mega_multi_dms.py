# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""E16 — Mega-budget multi-DMS with WT init.
All 4 DMS at 4000-query budget × 5 seeds × WT-init PRICE-RL.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dms_loaders import (AA20_INDEX, load_aav_sinai2021,  # noqa: E402
                                  load_enhancer_bcl11a, load_gfp_sarkisyan2016,
                                  load_tem1_stiffler2015)
from src.training.baselines import adalead, random_sampler  # noqa: E402
from src.training.oracle_gpu import TableOracleGPU  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402
from src.evaluation.metrics import top_k_recovery  # noqa: E402


def baseline_run(name, alphabet, length, oracle, rounds, batch, seed):
    rng = np.random.default_rng(seed)
    hX, hR = [], []
    for t in range(rounds):
        if name == "random" or not hR:
            X = random_sampler(alphabet, length, batch, rng)
        elif name == "adalead":
            X = adalead(np.concatenate(hX, 0), np.asarray(hR), alphabet, length, batch, rng)
        R = oracle.query(X)
        hX.append(X); hR.extend(R.tolist())
    return hR


def main():
    OUT = ROOT / "experiments" / "E16_mega_multi_dms"; OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = list(range(5))
    ROUNDS, BATCH = 20, 200       # 4000-query budget
    datasets = {
        "TEM1_Stiffler2015":   load_tem1_stiffler2015(),
        "Enhancer_BCL11A":     load_enhancer_bcl11a(),
        "GFP_Sarkisyan2016":   load_gfp_sarkisyan2016(),
        "AAV_Sinai2021":       load_aav_sinai2021(),
    }
    rows = []
    t0 = time.time()
    for ds_name, ds in datasets.items():
        seqs, fit = ds["sequences"], ds["fitness"]
        L, A = seqs.shape[1], len(ds["alphabet"])
        if A == 20:
            wt_idx = [AA20_INDEX.get(c, 0) for c in ds["wildtype"]]
        else:
            wt_idx = [0] * L
        print(f"\n[{ds_name}] N={seqs.shape[0]} L={L} A={A}")
        for seed in SEEDS:
            seed_everything(seed)
            for method in ["random", "adalead"]:
                R = baseline_run(method, A, L, TableOracleGPU(seqs, fit),
                                 ROUNDS, BATCH, seed)
                rows.append({"dataset": ds_name, "method": method, "seed": seed,
                             "best": float(np.max(R)),
                             "top_1pct": top_k_recovery(R, fit, 0.01)})
            cfg = PriceRLConfig(rounds=ROUNDS, batch=BATCH, inner_steps=8, seed=seed,
                                wildtype=wt_idx, wt_strength=4.0)
            algo = PriceRL(L, A, TableOracleGPU(seqs, fit), cfg); algo.run()
            rows.append({"dataset": ds_name, "method": "price_rl_wt", "seed": seed,
                         "best": float(np.max(algo.history_R)),
                         "top_1pct": top_k_recovery(algo.history_R, fit, 0.01),
                         "cos_T1_min": float(min(L_.cos_pooled_vs_decomp for L_ in algo.logs))})
        for m in ["random", "adalead", "price_rl_wt"]:
            tops = [r["top_1pct"] for r in rows
                    if r["dataset"] == ds_name and r["method"] == m]
            print(f"  {m:18s} top1%={np.mean(tops):.4f}±{np.std(tops):.4f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nE16 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
