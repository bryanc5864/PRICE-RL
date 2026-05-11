# REVIEW_REPORT_PRE_TRAINING.md

> Pre-training review gate (autonomous-research-engine §Phase 4 step 3).
> Carried out before any training run was started. The review covers:
> (i) algorithmic correctness, (ii) data integrity, (iii) leakage risk,
> (iv) determinism, (v) hard-threshold testability.

## Summary
**Verdict: PASS — proceed to training.**

## Items reviewed

### (i) Algorithmic correctness
- **Theorem-1 exactness check** is built into every round
  (`PriceRL.run` logs `cos_pooled_vs_decomp`). Smoke-test on NK(N=10, K=2)
  yielded **1.0000 cosine** between (g_S+g_T) and the standard pooled
  REINFORCE gradient. This is the strongest empirical evidence
  available that the partition we use is exact for factorised categorical
  policies. Hard threshold T1 (≥ 0.95) therefore trivially met on the toy.
- **Importance-weighted estimator** for g_T uses self-normalised clipping
  at `is_clip=5`. Variance-bounded by construction.
- **Controller** is a textbook PI on log(α_S/α_T) with anti-windup
  clipping at ±5.0. No risk of unbounded drift.
- **Autocorrelation estimator** falls back to `L=length` when too few
  pairs; never returns negative or zero. NK ground-truth comparison wired
  in for E3 (`NKLandscape.autocorr_length_analytic`).

### (ii) Data integrity
All five DMS files were downloaded from canonical public sources:

| Dataset | Source URL | N variants | L | Alphabet |
|---|---|---|---|---|
| GB1 four-mutation (Wu 2016) | FLIP repo `splits/gb1/four_mutations_full_data.csv.zip` | 149,361 | 4 | 20 |
| GFP (Sarkisyan 2016) | ProteinGym v0.1 | 51,714 | 238 | 20 |
| TEM-1 (Stiffler 2015) | ProteinGym v0.1 | 4,996 | 287 | 20 |
| AAV (Sinai 2021) | ProteinGym v0.1 | 42,328 | 589 | 20 |
| Enhancer-BCL11A (Kircher 2019) | MaveDB API | 1,799 | 600 | 4 |

The loader in `src/data/dms_loaders.py` is deterministic and
returns int-encoded sequences with `_normalize` (1st–99th percentile
clip) for cross-benchmark comparability. Raw fitness values are
preserved upstream — only display is normalised.

### (iii) Leakage risk
- **Active-learning protocol** simulates wet-lab queries on top of a
  fully-labelled DMS table. The "oracle" is a hash-table lookup; queries
  outside the table fall back to nearest-Hamming-neighbour mean and the
  miss is logged (`TableOracle.misses`). No model has access to held-out
  labels through any channel other than oracle queries.
- **Surrogate** for the reward-hacking experiment (E2) is trained on a
  deliberately small slice of NK ground truth — labels used to train it
  are explicitly distinct from those used to evaluate it.

### (iv) Determinism
`src/utils/seeding.py` seeds Python, NumPy, and PyTorch (CPU+CUDA). Every
config takes a `seed` field. NK landscape uses a deterministic blake2b
hash for fitness contributions, so two NKLandscape(seed=7) instances are
bitwise identical.

### (v) Hard-threshold testability
All §7.3 thresholds (T1–T6) are computable from the per-round log
fields (`cos_pooled_vs_decomp`, `rho`, `rho_star`, `L_hat`,
`best_so_far`). Failure criteria F1–F3 are likewise observable.

## Action items resolved before training
- Added entropy-bonus step (`_entropy_step`) to prevent premature mode
  collapse on rugged landscapes.
- Added `random_support` ablation switch so the support-quality ablation
  can run from a single config flag.
- Confirmed `pdflatex` is available in `/usr/bin` (TeX Live 2019) for
  paper compilation.

## Outstanding caveats (not blockers)
- ESM-2 embedding backbone is substituted with a learned 40-dim
  embedding for GB1 (RESEARCH_PLAN.md §5). Documented in plan; does not
  affect Theorem 1/2/3 validity.
- Cross-protein transfer (Exp 4) and cross-domain (Exp 5) deferred to
  future work for compute-budget reasons. RESEARCH_PLAN.md §2 already
  reflects this.
