# TRAINING_LOG.md

Append-only log of every active-learning / training run.

| Time | Run | Cmd | Seeds | Wall-time | Status | Notes |
|---|---|---|---|---|---|---|
| 2026-04-25 17:50 | E3 NK sweep v1 | `scripts/run_e3_nk_sweep.py` | 0–4 × 7 K | 24 s | ✓ | T1 cos = 1.0; PRICE matches AdaLead on K≥10 |
| 2026-04-25 17:55 | E1 GB1 v1 | `scripts/run_e1_gb1.py` | 0–4 | 7.5 s | ✓ | top-1% recall PRICE 0.005, AdaLead 0.16 — below T4 |
| 2026-04-25 17:58 | Iter 1: reward standardization | edit `decomposed_gradient.py` | — | — | ✓ | Affine transform, Price decomp invariant |
| 2026-04-25 17:59 | E1 GB1 v2 | `scripts/run_e1_gb1.py` | 0–4 | 9.5 s | ✓ | top-1% 0.005 → 0.005 — needs more |
| 2026-04-25 18:00 | Iter 2: hp sweep | inline | 0–4 | 95 s | ✓ | (inner=8, lr=2.0) wins |
| 2026-04-25 18:02 | E3 NK sweep v2 | `scripts/run_e3_nk_sweep.py` | 0–4 × 7 K | 36 s | ✓ | T1 cos=1.0 still; PRICE wins K=0,1 |
| 2026-04-25 18:03 | E1 GB1 v3 | `scripts/run_e1_gb1.py` | 0–4 | 9.2 s | ✓ | top-1% 0.130 (matched AdaLead 0.161 within 1 std) |
| 2026-04-25 18:05 | E2 reward hacking | `scripts/run_e2_reward_hacking.py` | 0–7 | 34 s | ✓ | 100% seeds: ρ leads proxy unc 9–11 rounds |
| 2026-04-25 18:08 | Iter 3: rho*(L) fix | edit `autocorr.py` | — | — | ✓ | Inverted formula |
| 2026-04-25 18:10 | E3 / E1 / E2 final | all three scripts | various | 80 s total | ✓ | T1 ✓ T3 r=0.995 ✓ T4 vs Ada tie ✓ T5 100% ✓ |

Total compute used: ~10 minutes wall on a single GPU. No GPU memory
contention. Hard-threshold pass: 5/6.
