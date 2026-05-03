# REVIEW_REPORT_POST_RESULTS.md

> Post-results review gate (autonomous-research-engine §Phase 4 step 7).
> Carried out after experiments and before paper writing.

## Verdict: **PASS — proceed to paper writing.**

## Audit checklist

### (1) No fabricated numbers
Every number in `RESULTS.md` was traced to a CSV in `experiments/`:
- E1 numbers ↔ `experiments/E1_gb1/summary.csv` (rows 0–24, 5 methods × 5 seeds).
- E2 numbers ↔ `experiments/E2_reward_hacking/trace.csv` (96 rows: 8 seeds × 12 rounds).
- E3 numbers ↔ `experiments/E3_nk_sweep/summary.csv` and `per_round.csv`
  (35 rows in summary; 280 rows in per_round = 7 K × 5 seeds × 8 rounds).
- Threshold table ↔ output of `scripts/analyze_results.py`.

### (2) No data leakage
- TableOracle prevents access to held-out labels through any channel
  except oracle queries. Calls and misses are tracked.
- Surrogate model in E2 is trained on a *random* slice (200 of 4²⁰
  possible sequences) — independent of any subsequent active-learning
  trajectory.
- No information from the held-out evaluation crosses into model
  training in any experiment.

### (3) Threshold honesty
The PI controller stability threshold T2 was **NOT met** (13% vs 100%
required). This is reported transparently in RESULTS.md and will be
discussed as a limitation in the paper. We did not move the threshold
post-hoc to make T2 pass — see commit history of RESEARCH_PLAN.md.

### (4) Reproducibility
- All seeds are explicit (`SEEDS = [0, 1, 2, 3, 4]` for protein and NK,
  `range(8)` for E2). All RNGs (Python, NumPy, PyTorch CPU+CUDA) are
  re-seeded by `seed_everything()`.
- The autonomous improvement loop's three iterations are recorded in
  `RESULTS.md` and visible in the committed source diff.
- TRAINING_LOG.md captures every wall-time, status, and observation.

### (5) Statistical rigor
- E1 between-method comparison: 10,000-resample paired bootstrap on the
  difference of means (PRICE-RL vs random), p = 0.0000.
- E1 vs AdaLead: tested for non-inferiority within 1 std (gap −0.0004,
  std 0.0416) — within 1 std comfortably.
- E2 lead-time: counted as the difference between the round when ρ_t
  first crosses 0.9 and the round when proxy uncertainty first crosses
  95% of its final value. Per-seed leads (9, 9, 10, 10, 10, 10, 11, 11)
  reported in full.

### (6) Figures match data
All six figures (`figures/fig_*.{png,pdf}`) were generated from the same
CSV files used to compute the threshold tables. No figure was hand-edited.

### (7) Code review
Spot-checked critical path:
- `compute_decomposed_gradient` — partition is exact, IS clip is finite,
  cosine check fires every round.
- `PriceController.step` — sign convention documented in source comment.
- `estimate_autocorr_length` — log-space OLS, robust to small N.
- Seeding helpers cover all RNG sources.

## Outstanding issues (not blockers)
- **T2** open issue: tracking is loose; documented in §Discussion of paper.
- **Cross-protein transfer (Exp 4) and non-biology (Exp 5)** deferred —
  noted in Future Work.

## Sign-off
Ready to write the LaTeX paper.
