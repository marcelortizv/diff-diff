# Survey Data Support Roadmap

This document captures planned future work for survey data support in diff-diff.
Phases 1-3 are implemented. Phases 4-5 are deferred for future PRs.

## Implemented (Phases 1-2)

- **SurveyDesign** class with weights, strata, PSU, FPC, weight_type, nest, lonely_psu
- **Weighted OLS** via sqrt(w) transformation in `solve_ols()`
- **Weighted sandwich estimator** in `compute_robust_vcov()` (pweight/fweight/aweight)
- **Taylor Series Linearization** (TSL) variance with strata + PSU + FPC
- **Survey degrees of freedom**: n_PSU - n_strata
- **Base estimator integration**: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD
- **Weighted demeaning**: `demean_by_group()` and `within_transform()` with weights
- **SurveyMetadata** in results objects with effective n, DEFF, weight_range

## Implemented (Phase 3): OLS-Based Standalone Estimators

| Estimator | File | Survey Support | Notes |
|-----------|------|----------------|-------|
| StackedDiD | `stacked_did.py` | Full | Q-weights compose multiplicatively with survey weights; TSL vcov on composed weights |
| SunAbraham | `sun_abraham.py` | Full | Survey weights in LinearRegression + weighted within-transform; bootstrap+survey deferred |
| BaconDecomposition | `bacon.py` | Diagnostic | Weighted cell means, weighted within-transform, weighted group shares; no inference (diagnostic only) |
| TripleDifference | `triple_diff.py` | Reg only | Regression method with weighted OLS + TSL on influence functions; IPW/DR deferred (needs weighted `solve_logit()`) |
| ContinuousDiD | `continuous_did.py` | Analytical | Weighted B-spline OLS + TSL on influence functions; bootstrap+survey deferred |
| EfficientDiD | `efficient_did.py` | Analytical | Weighted means/covariances in Omega* + TSL on EIF scores; bootstrap+survey deferred |

### Phase 3 Deferred Work

The following capabilities were deferred from Phase 3 because they depend on
Phase 5 infrastructure (weighted `solve_logit()` or bootstrap+survey interaction):

| Estimator | Deferred Capability | Blocker |
|-----------|-------------------|---------|
| SunAbraham | Pairs bootstrap + survey | Phase 5: bootstrap+survey interaction |
| TripleDifference | IPW and DR methods + survey | Phase 5: weighted `solve_logit()` |
| ContinuousDiD | Multiplier bootstrap + survey | Phase 5: bootstrap+survey interaction |
| EfficientDiD | Multiplier bootstrap + survey | Phase 5: bootstrap+survey interaction |
| EfficientDiD | Covariates (DR path) + survey | DR nuisance estimation needs survey weight threading |

All blocked combinations raise `NotImplementedError` when attempted, with a
message pointing to the planned phase or describing the limitation.

## Phase 4: Complex Standalone Estimators

These require more substantial changes beyond threading weights.

| Estimator | File | Complexity | Notes |
|-----------|------|------------|-------|
| ImputationDiD | `imputation.py` | Medium | Weighted Stage 1 OLS, weighted ATT aggregation, weighted conservative variance |
| TwoStageDiD | `two_stage.py` | Medium-High | Weighted within-transformation, weighted GMM sandwich |
| CallawaySantAnna | `staggered.py` | High | Weights enter propensity score (weighted solve_logit), IPW/DR reweighting (w_survey x w_ipw), multiplier bootstrap |
| SyntheticDiD | `synthetic_did.py` | Medium | Survey-weighted treated mean in optimization, weighted placebo variance |
| TROP | `trop.py` | Medium | Survey weights in ATT aggregation and LOOCV (distinct from TROP's internal weights) |

## Phase 5: Advanced Features

### Replicate Weight Variance
Re-run WLS for each replicate weight column, compute variance from distribution
of estimates. Supports BRR, Fay's BRR, JK1, JKn, bootstrap replicate methods.
Add `replicate_weights`, `replicate_type`, `replicate_rho` fields to SurveyDesign.

### DEFF Diagnostics
Compare survey vcov to SRS vcov element-wise. Report design effect per
coefficient. Effective n = n / DEFF.

### Wild Bootstrap with Survey Weights
Add `survey_weights` parameter to `wild_bootstrap_se()`. Requires careful
interaction between bootstrap resampling and survey weight structure.

### Subpopulation Analysis
`SurveyDesign.subpopulation(data, mask)` — zero-out weights for excluded
observations while preserving the full design structure for correct variance
estimation (unlike simple subsetting, which would drop design information).

### Weighted `solve_logit()`
Add weights to the IRLS iteration in `solve_logit()`. Required by
CallawaySantAnna (propensity score estimation) and TripleDifference (IPW method).
Working weights become `w_survey * mu * (1 - mu)`.
