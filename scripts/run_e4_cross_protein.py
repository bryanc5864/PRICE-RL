# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
Experiment E4 — cross-protein transfer (Theorem 3 validation).

Setup. The BLAT family in ProteinGym has four TEM-1 β-lactamase DMS
studies under different selection conditions / library protocols:
Stiffler 2015 (cefotaxime; main paper), Deng 2012 (ampicillin),
Firnberg 2014 (broad selection), and Jacquier 2013 (small library).
These share *the same protein* but differ in selection pressure and
mutational neighbourhood — exactly the variation Theorem 3 says
selection-only transfer should handle.

Protocol.
  1. Train a PRICE-RL factorised policy on TEM-1 (Stiffler) for 5
     rounds (5 seeds). Save policy logits.
  2. For each target ∈ {Deng, Firnberg, Jacquier}, run a cold campaign
     of 3 rounds × 50 queries × 5 seeds in three transfer modes:
       • cold      — fresh uniform init (control)
       • full      — copy source logits verbatim
       • selection — copy source logits centred & damped (preserves
                     per-locus rank structure but resets concentration)
  3. Compare top-1% recall: Theorem 3 predicts selection > full when
     mutational neighbourhood differs.
"""
from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dms_loaders import (load_blat_deng2012,  # noqa: E402
                                  load_blat_firnberg2014,
                                  load_blat_jacquier2013,
                                  load_tem1_stiffler2015)
from src.training.oracle import TableOracle  # noqa: E402
from src.training.price_rl import PriceRL, PriceRLConfig  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402
from src.evaluation.metrics import top_k_recovery  # noqa: E402


def make_pricerl(target_ds, seed: int, rounds: int = 3, batch: int = 50) -> PriceRL:
    cfg = PriceRLConfig(rounds=rounds, batch=batch, inner_steps=8, seed=seed)
    return PriceRL(target_ds["sequences"].shape[1], len(target_ds["alphabet"]),
                   TableOracle(target_ds["sequences"], target_ds["fitness"]), cfg)


def main():
    OUT = ROOT / "experiments" / "E4_cross_protein"
    OUT.mkdir(parents=True, exist_ok=True)

    source = load_tem1_stiffler2015()
    targets = {
        "BLAT_Deng2012":     load_blat_deng2012(),
        "BLAT_Firnberg2014": load_blat_firnberg2014(),
        "BLAT_Jacquier2013": load_blat_jacquier2013(),
    }
    SEEDS = [0, 1, 2, 3, 4]
    rows = []
    t0 = time.time()

    for seed in SEEDS:
        # --- Train on source ---
        seed_everything(seed)
        algo_src = PriceRL(source["sequences"].shape[1], len(source["alphabet"]),
                           TableOracle(source["sequences"], source["fitness"]),
                           PriceRLConfig(rounds=5, batch=100, inner_steps=8, seed=seed))
        algo_src.run()
        src_logits = algo_src.policy.logits.detach().clone()
        # rank-preserving centred + damped logits (selection-only)
        damping = 0.3
        sel_logits = (src_logits - src_logits.mean(dim=-1, keepdim=True)) * damping

        for tgt_name, tgt_ds in targets.items():
            # COLD
            algo_cold = make_pricerl(tgt_ds, seed)
            algo_cold.run()
            r_cold = top_k_recovery(algo_cold.history_R, tgt_ds["fitness"], 0.01)

            # FULL transfer
            algo_full = make_pricerl(tgt_ds, seed)
            with torch.no_grad():
                algo_full.policy.logits.copy_(src_logits)
            algo_full.run()
            r_full = top_k_recovery(algo_full.history_R, tgt_ds["fitness"], 0.01)

            # SELECTION-only transfer
            algo_sel = make_pricerl(tgt_ds, seed)
            with torch.no_grad():
                algo_sel.policy.logits.copy_(sel_logits)
            algo_sel.run()
            r_sel = top_k_recovery(algo_sel.history_R, tgt_ds["fitness"], 0.01)

            rows.append({"seed": seed, "target": tgt_name,
                         "cold": r_cold, "full": r_full, "selection": r_sel,
                         "best_cold": float(np.max(algo_cold.history_R)),
                         "best_full": float(np.max(algo_full.history_R)),
                         "best_sel":  float(np.max(algo_sel.history_R))})
            print(f"seed={seed} target={tgt_name:18s} cold={r_cold:.3f} "
                  f"full={r_full:.3f} sel={r_sel:.3f}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nE4 done in {time.time()-t0:.1f}s")

    # Aggregate
    targets_list = sorted({r["target"] for r in rows})
    for tgt in targets_list:
        rs_cold = [r["cold"] for r in rows if r["target"] == tgt]
        rs_full = [r["full"] for r in rows if r["target"] == tgt]
        rs_sel  = [r["selection"] for r in rows if r["target"] == tgt]
        print(f"  {tgt:18s} cold={np.mean(rs_cold):.3f}±{np.std(rs_cold):.3f}  "
              f"full={np.mean(rs_full):.3f}±{np.std(rs_full):.3f}  "
              f"sel={np.mean(rs_sel):.3f}±{np.std(rs_sel):.3f}")


if __name__ == "__main__":
    main()
