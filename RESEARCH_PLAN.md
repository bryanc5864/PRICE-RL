# RESEARCH_PLAN.md — PRICE-RL

> Selection–Transmission Decomposed RL for Sequential Biological Design
> Bryan Cheng · CJZ Labs · 2026
>
> This plan is the executable refinement of the user-supplied proposal. It
> preserves the proposal's claims and adds the §7.3 quantitative success
> thresholds required by the autonomous research engine.

## 1. Goal

Operationalize Frank (2025)'s Price-equation/policy-gradient equivalence
as an RL primitive: decompose the policy gradient into selection (g_S)
and transmission (g_T) components, weight them with a PI controller
targeting a regret-optimal Price ratio ρ*_t computed from the reward
landscape's autocorrelation length, and validate the resulting algorithm
theoretically (3 theorems) and empirically (a synthetic NK sweep + a
public protein DMS benchmark + a reward-hacking diagnostic experiment).

## 2. Hypotheses

- **H1 (low-data SOTA, partial).** PRICE-RL matches or beats baselines on
  standard active-learning protein/DNA benchmarks. The win margin is
  predictable from the landscape's autocorrelation length L.
- **H2 (reward-hacking).** ρ_t saturates toward 1 strictly earlier than
  proxy-variance signals when the policy is exploiting proxy artifacts.
- **H3 (theory-empirics match).** On NK landscapes, empirical ρ_t tracks
  the analytically-computable optimum ρ*_t across K∈{0..N}.

(Hypotheses 4 and 5 from the proposal — cross-protein transfer and
non-biology cross-domain — are deferred to a future submission for
compute-budget reasons; they are listed as future work in the paper.)

## 3. Theoretical contributions

- **Theorem 1.** Equivalence: ∇J = g_S + g_T under entropy-regularized
  active-learning RL with embedding-density support.
- **Theorem 2.** Adaptive-mixing regret bound parameterized by the reward
  landscape's autocorrelation length L; tight up to polylog factors.
- **Theorem 3.** Selection–transmission identifiability for cross-task
  transfer.

## 4. Method overview

Per round t:
1. Sample B candidates from π_θ.
2. Query oracle (here: cached DMS labels / NK ground truth).
3. Partition batch by membership in supp_t (95th-percentile embedding-
   density radius of π_{θ_{t-1}}).
4. Compute g_S (REINFORCE on in-support) and g_T (clipped IS-REINFORCE
   on out-of-support).
5. ρ_t = ‖g_S‖ / (‖g_S‖+‖g_T‖); estimate L̂_t from history; look up
   ρ*_t; PI-control α_S, α_T to drive ρ_t → ρ*_t.
6. Apply θ_{t+1} = θ_t + α_S g_S + α_T g_T.

## 5. Backbone choice

The proposal calls for ESM-2 150M as a frozen embedding. For the GB1
4-position benchmark we substitute a lightweight learned embedding
(40-dim) because the sequence space is small (20⁴ = 160k) and ESM-2
embedding precomputation on 160k sequences takes longer than the entire
active-learning campaign. We document this substitution explicitly. For
NK landscapes the policy is a categorical softmax over positions, so no
PLM is needed.

## 6. Datasets

- **NK landscapes (synthetic).** N=20, K∈{0,1,3,5,10,15,19}. 5 seeds per K.
- **GB1 (Wu et al. 2016).** 4-position combinatorial DMS, 149,361 variants.
  Active-learning protocol: 5 rounds × B=100 queries.

(GFP, TEM-1, AAV, enhancer DMS deferred. NK + GB1 alone exercises the
full algorithm including the autocorrelation estimator and the
embedding-density support.)

## 7. Experimental plan

### 7.1 Experiments

- **E3 (NK sweep, primary theory test).** PRICE-RL vs. random / AdaLead
  on NK with K∈{0,1,3,5,10,15,19}. 5 seeds. Track best fitness, ρ_t,
  ρ*_t (analytic), regret.
- **E2 (reward-hacking diagnostic).** Train a deliberately-weak proxy
  surrogate on 5% of NK labels; track ρ_t and proxy uncertainty as
  campaign proceeds; verify ρ_t crosses 0.9 before proxy variance flags.
- **E1 (low-data benchmark).** PRICE-RL vs. random / AdaLead / PEX on
  GB1; 5 rounds × B=100 × 5 seeds. Best fitness reached + top-1%
  recovery.
- **Ablations.** (a) PRICE-RL with α_S=α_T (no adaptive mixing).
  (b) PRICE-RL without ρ-targeted control (open-loop schedule).
  (c) PRICE-RL with random support partition (kills the decomposition).

### 7.2 Statistical protocol

- All numbers averaged over 5 seeds with std-dev bars.
- Paired bootstrap test (10k resamples) for between-method comparisons.
- p-values reported alongside effect sizes.

### 7.3 Definition of success (QUANTITATIVE THRESHOLDS — REQUIRED)

#### Hard thresholds (must meet ALL to consider project complete)

- **T1 (theorem 1, empirical).** Sum of estimated g_S + g_T must agree
  with the standard pooled REINFORCE gradient to cosine similarity ≥ 0.95
  on NK (any K). Validates Theorem 1's exactness claim.
- **T2 (controller stability).** PI controller drives ρ_t to within ±0.05
  of the target ρ*_t in ≤ 10 rounds, on 100% of seeds across the K sweep.
- **T3 (theory-empirics match, H3).** On NK, the Pearson correlation
  between empirical mean ρ_t over the campaign and the analytic ρ*_t,
  taken across the K sweep, is ≥ 0.7.
- **T4 (low-data win, H1).** PRICE-RL achieves best-fitness no worse than
  the best baseline (random / AdaLead) on GB1 (paired-bootstrap p < 0.10
  for the comparison vs. random; non-inferior vs. AdaLead within 1 std).
- **T5 (reward-hacking lead, H2).** ρ_t crosses 0.9 at least 2 rounds
  before proxy variance crosses its 95th-percentile alarm threshold, on
  ≥ 80% of reward-hacking seeds.
- **T6 (no fabrication).** Every numeric result in RESULTS.md is
  reproducible from `scripts/run_*.py` with a fixed seed.

#### Soft thresholds (should meet most)

- **S1.** Adaptive mixing (full PRICE-RL) beats α_S=α_T ablation on best
  fitness for K ≥ 5 (the rugged regime), with effect size ≥ 5 %.
- **S2.** Empirical regret on NK scales no worse than √T·polylog as
  predicted by Theorem 2 (qualitative match in log-log slope).
- **S3.** Reproducibility: every figure regenerable end-to-end from
  source CSVs in ≤ 30 minutes on a single GPU.

#### Failure criteria (revise plan if any hold)

- **F1.** g_S+g_T cosine to pooled gradient < 0.8 → Theorem 1 statement
  needs weakening (an approximation bound, not exact identity).
- **F2.** Controller diverges or oscillates with amplitude > 0.2 in
  ρ_t-space on > 1 seed/K — paper reports honest failure of stable
  control and reframes ρ_t as a diagnostic only.
- **F3.** PRICE-RL strictly underperforms random acquisition on GB1 →
  pivot framing: paper shifts from "algorithm" to "diagnostic" (per
  proposal §8 Risk 3 fall-back).

## 8. Compute envelope (this run)

- Hardware: a single GTX 1080 Ti / RTX 2080 Ti–class GPU (one of 10
  available). PRICE-RL is intentionally lightweight — full NK sweep + GB1
  campaign + ablations + reward-hacking experiment fits in ≤ 6 wall-hours.
- All experiments use deterministic seeds and ship CSV outputs.

## 9. Deliverables

- `src/` — PRICE-RL implementation, NK simulator, GB1 active-learning
  loop, baselines, controller.
- `experiments/` — config + raw outputs.
- `figures/` — PNG and PDF for paper.
- `paper/main.tex` — compiled to `paper/main.pdf`.
- `RESULTS.md`, `TRAINING_LOG.md`, review reports, `BIBLIOGRAPHY_VERIFICATION.md`.

## 10. Plan adherence rule

Every code module references the section of this plan it implements
(see file headers). Any deviation is documented in TRAINING_LOG.md.
