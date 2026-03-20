# Survey Data Support Roadmap

This document captures planned future work for survey data support in diff-diff.
Phases 1-2 (core infrastructure + base estimator integration) are implemented.
Phases 3-5 are deferred for future PRs.

## Implemented (Phases 1-2)

- **SurveyDesign** class with weights, strata, PSU, FPC, weight_type, nest, lonely_psu
- **Weighted OLS** via sqrt(w) transformation in `solve_ols()`
- **Weighted sandwich estimator** in `compute_robust_vcov()` (pweight/fweight/aweight)
- **Taylor Series Linearization** (TSL) variance with strata + PSU + FPC
- **Survey degrees of freedom**: n_PSU - n_strata
- **Base estimator integration**: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD
- **Weighted demeaning**: `demean_by_group()` and `within_transform()` with weights
- **SurveyMetadata** in results objects with effective n, DEFF, weight_range

## Phase 3: OLS-Based Standalone Estimators

These estimators use `LinearRegression` or `solve_ols()` internally, so weight
support is largely mechanical (threading `survey_design` through `fit()`).

| Estimator | File | Complexity | Notes |
|-----------|------|------------|-------|
| StackedDiD | `stacked_did.py` | Low | Q-weights compose multiplicatively with survey weights |
| SunAbraham | `sun_abraham.py` | Low | Single saturated regression via LinearRegression |
| BaconDecomposition | `bacon.py` | Medium | Weighted demeaning, weighted cell means, effective group sizes |
| TripleDifference | `triple_diff.py` | Medium-High | Three methods (reg, ipw, dr); weighted cell means, weighted logistic regression (requires Phase 5) |
| ContinuousDiD | `continuous_did.py` | Low-Medium | Uses LinearRegression internally |
| EfficientDiD | `efficient_did.py` | Low-Medium | Uses LinearRegression internally |

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
