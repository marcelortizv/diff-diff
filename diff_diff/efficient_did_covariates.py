"""
Doubly robust math for the Efficient DiD estimator (with covariates).

Implements the with-covariates path from Chen, Sant'Anna & Xie (2025):
outcome regression via OLS, propensity score ratios via logistic regression,
doubly robust generated outcomes (Eq 4.4), and the efficient influence
function for analytical standard errors.

All functions are pure (no state), operating on pre-pivoted numpy arrays.
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np

from diff_diff.linalg import (
    _check_propensity_diagnostics,
    solve_logit,
    solve_ols,
)


def estimate_outcome_regression(
    outcome_wide: np.ndarray,
    covariate_matrix: np.ndarray,
    group_mask: np.ndarray,
    t_col: int,
    tpre_col: int,
) -> np.ndarray:
    """Estimate conditional mean outcome change m_hat(X) for a comparison group.

    Regresses ``(Y_t - Y_{tpre})`` on ``X`` within the units identified by
    ``group_mask`` using OLS.  Returns predicted values ``m_hat(X_i)`` for
    **all** units (extrapolated from the within-group fit).

    This implements ``m_hat_{g',t,tpre}(X) = E[Y_t - Y_{tpre} | G=g', X]``.

    Parameters
    ----------
    outcome_wide : ndarray, shape (n_units, n_periods)
        Pivoted outcome matrix.
    covariate_matrix : ndarray, shape (n_units, n_covariates)
        Unit-level (time-invariant) covariates.
    group_mask : ndarray of bool, shape (n_units,)
        Mask selecting units in the comparison group.
    t_col, tpre_col : int
        Column indices in ``outcome_wide`` for the two time periods.

    Returns
    -------
    m_hat : ndarray, shape (n_units,)
        Predicted ``E[Y_t - Y_{tpre} | X]`` for every unit.
    """
    # Dependent variable: outcome change within the comparison group
    Y_group = outcome_wide[group_mask]
    delta_y = Y_group[:, t_col] - Y_group[:, tpre_col]

    # Design matrix with intercept for the group
    X_group = covariate_matrix[group_mask]
    X_design = np.column_stack([np.ones(len(X_group)), X_group])

    # Fit OLS — we only need coefficients, not vcov
    coef, _, _ = solve_ols(
        X_design,
        delta_y,
        return_vcov=False,
        rank_deficient_action="warn",
    )

    # Predict for all units
    X_all = np.column_stack([np.ones(len(covariate_matrix)), covariate_matrix])

    # Handle NaN coefficients from rank-deficient fits: set NaN coefs to 0
    # so prediction degrades gracefully (those terms contribute nothing)
    coef_safe = np.where(np.isfinite(coef), coef, 0.0)
    m_hat = X_all @ coef_safe

    # Guard against non-finite predictions
    non_finite = ~np.isfinite(m_hat)
    if non_finite.any():
        n_bad = int(non_finite.sum())
        warnings.warn(
            f"Outcome regression produced {n_bad} non-finite prediction(s). "
            "Setting to 0.0 (equivalent to no covariate adjustment).",
            UserWarning,
            stacklevel=2,
        )
        m_hat[non_finite] = 0.0

    return m_hat


def estimate_propensity_ratio(
    covariate_matrix: np.ndarray,
    mask_g: np.ndarray,
    mask_gp: np.ndarray,
    pscore_trim: float = 0.01,
) -> np.ndarray:
    r"""Estimate propensity score ratio r_{g,g'}(X) = p_g(X) / p_{g'}(X).

    Fits binary logistic regression on units in ``{g, g'}`` with ``D=1``
    for ``G=g`` and ``D=0`` for ``G=g'``.  The fitted probability is
    ``pscore = P(G=g | G in {g,g'}, X)``, and the ratio is computed as
    ``pscore / (1 - pscore)`` (conditional odds).

    On logit failure (convergence, separation, LinAlgError), falls back to
    the unconditional population fraction ratio ``n_g / n_{g'}``.

    Parameters
    ----------
    covariate_matrix : ndarray, shape (n_units, n_covariates)
        Unit-level covariates.
    mask_g : ndarray of bool, shape (n_units,)
        Mask for the target treatment group.
    mask_gp : ndarray of bool, shape (n_units,)
        Mask for the comparison group.
    pscore_trim : float, default 0.01
        Propensity scores are clipped to ``[pscore_trim, 1-pscore_trim]``
        before ratio computation.

    Returns
    -------
    ratio : ndarray, shape (n_units,)
        Estimated ``r_{g,g'}(X_i)`` for every unit (extrapolated from
        the fit on ``{g, g'}`` units).
    """
    n_g = int(np.sum(mask_g))
    n_gp = int(np.sum(mask_gp))
    n_units = len(covariate_matrix)

    # Short-circuit: r_{g,g}(X) = 1 for same-cohort comparisons (PT-All)
    if np.array_equal(mask_g, mask_gp):
        return np.ones(n_units)

    # Stack covariates for the two groups
    combined_mask = mask_g | mask_gp
    X_combined = covariate_matrix[combined_mask]
    # Treatment indicator: derive from mask_g so labels align with row order
    D = mask_g[combined_mask].astype(float)

    try:
        beta, pscore_combined = solve_logit(X_combined, D, rank_deficient_action="warn")
        _check_propensity_diagnostics(pscore_combined, pscore_trim)

        # Predict for all units using the logit coefficients
        X_all_with_intercept = np.column_stack([np.ones(n_units), covariate_matrix])
        # Handle NaN coefficients from rank-deficient fits
        beta_safe = np.where(np.isfinite(beta), beta, 0.0)
        z = X_all_with_intercept @ beta_safe
        z = np.clip(z, -500, 500)
        pscore_all = 1.0 / (1.0 + np.exp(-z))

    except (np.linalg.LinAlgError, ValueError):
        warnings.warn(
            "Propensity score estimation failed for a group pair. "
            "Falling back to unconditional population fraction ratio.",
            UserWarning,
            stacklevel=2,
        )
        # Fallback: constant ratio n_g / n_gp for all units
        flat_p = n_g / (n_g + n_gp) if (n_g + n_gp) > 0 else 0.5
        pscore_all = np.full(n_units, flat_p)

    # Trim propensity scores
    pscore_all = np.clip(pscore_all, pscore_trim, 1.0 - pscore_trim)

    # Ratio: pscore / (1 - pscore) = conditional odds
    ratio = pscore_all / (1.0 - pscore_all)
    return ratio


def compute_generated_outcomes_cov(
    target_g: float,
    target_t: float,
    valid_pairs: List[Tuple[float, float]],
    outcome_wide: np.ndarray,
    cohort_masks: Dict[float, np.ndarray],
    never_treated_mask: np.ndarray,
    period_to_col: Dict[float, int],
    period_1_col: int,
    cohort_fractions: Dict[float, float],
    m_hat_cache: Dict[Tuple, np.ndarray],
    r_hat_cache: Dict[Tuple[float, float], np.ndarray],
    never_treated_val: float = np.inf,
) -> np.ndarray:
    """Compute per-unit doubly robust generated outcomes (Eq 4.4).

    For each valid pair ``(g', t_pre)`` and each unit ``i``, three terms::

        Term 1 (treated):
            (G_{g,i} / pi_g) * (Y_{i,t} - Y_{i,1}
                - m_{inf,t,tpre}(X_i) - m_{g',tpre,1}(X_i))

        Term 2 (never-treated):
            -r_{g,inf}(X_i) * (G_{inf,i} / pi_g)
                * (Y_{i,t} - Y_{i,tpre} - m_{inf,t,tpre}(X_i))

        Term 3 (comparison cohort):
            -r_{g,g'}(X_i) * (G_{g',i} / pi_g)
                * (Y_{i,tpre} - Y_{i,1} - m_{g',tpre,1}(X_i))

    Parameters
    ----------
    target_g, target_t : float
        Target group-time.
    valid_pairs : list of (g', t_pre)
        Valid comparison pairs.
    outcome_wide : ndarray, shape (n_units, n_periods)
    cohort_masks : dict
        ``{cohort: bool_mask}``
    never_treated_mask : ndarray of bool
    period_to_col : dict
    period_1_col : int
        Column index of effective baseline period (Y_1).
    cohort_fractions : dict
        ``{cohort: n_cohort / n}``
    m_hat_cache : dict
        Outcome regression predictions, keyed by
        ``(comparison_group, t_col, tpre_col)``.
    r_hat_cache : dict
        Propensity score ratios, keyed by ``(target_g, comparison_g)``.
    never_treated_val : float
        Sentinel for the never-treated group.

    Returns
    -------
    gen_out : ndarray, shape (n_units, H)
        Per-unit generated outcome for each valid pair.
    """
    H = len(valid_pairs)
    n_units = outcome_wide.shape[0]
    if H == 0:
        return np.empty((n_units, 0))

    t_col = period_to_col[target_t]
    y1_col = period_1_col

    g_mask = cohort_masks[target_g]
    pi_g = cohort_fractions[target_g]

    gen_out = np.zeros((n_units, H))

    for j, (gp, tpre) in enumerate(valid_pairs):
        tpre_col = period_to_col[tpre]

        # Retrieve cached nuisance parameters
        # m_{inf, t, tpre}(X)
        m_inf_t_tpre = m_hat_cache[(never_treated_val, t_col, tpre_col)]
        # m_{g', tpre, 1}(X)
        m_gp_tpre_1 = m_hat_cache[(gp, tpre_col, y1_col)]
        # r_{g, inf}(X)
        r_g_inf = r_hat_cache[(target_g, never_treated_val)]
        # r_{g, g'}(X)
        r_g_gp = r_hat_cache[(target_g, gp)]

        # ------- Term 1: treated units (G = g) -------
        if pi_g > 0:
            Y_t_minus_Y1 = outcome_wide[g_mask, t_col] - outcome_wide[g_mask, y1_col]
            residual_treated = Y_t_minus_Y1 - m_inf_t_tpre[g_mask] - m_gp_tpre_1[g_mask]
            gen_out[g_mask, j] += (1.0 / pi_g) * residual_treated

        # ------- Term 2: never-treated units (G = inf) -------
        pi_inf = cohort_fractions.get(never_treated_val, 0.0)
        if pi_inf > 0:
            Y_t_minus_Ytpre = (
                outcome_wide[never_treated_mask, t_col] - outcome_wide[never_treated_mask, tpre_col]
            )
            residual_inf = Y_t_minus_Ytpre - m_inf_t_tpre[never_treated_mask]
            gen_out[never_treated_mask, j] -= (
                r_g_inf[never_treated_mask] * (1.0 / pi_g) * residual_inf
            )

        # ------- Term 3: comparison cohort units (G = g') -------
        if np.isinf(gp):
            gp_mask = never_treated_mask
        else:
            gp_mask = cohort_masks[gp]
        pi_gp = cohort_fractions.get(gp, 0.0)
        if pi_gp > 0:
            Y_tpre_minus_Y1 = outcome_wide[gp_mask, tpre_col] - outcome_wide[gp_mask, y1_col]
            residual_gp = Y_tpre_minus_Y1 - m_gp_tpre_1[gp_mask]
            gen_out[gp_mask, j] -= r_g_gp[gp_mask] * (1.0 / pi_g) * residual_gp

    return gen_out


def compute_omega_star_cov(
    generated_outcomes: np.ndarray,
) -> np.ndarray:
    """Unconditional sample covariance of per-unit DR generated outcomes.

    Uses ``ddof=1`` for consistency with ``_sample_cov()`` in the nocov path.

    Parameters
    ----------
    generated_outcomes : ndarray, shape (n_units, H)
        Per-unit generated outcomes from :func:`compute_generated_outcomes_cov`.

    Returns
    -------
    omega : ndarray, shape (H, H)
        Covariance matrix.
    """
    n, H = generated_outcomes.shape
    if H == 0:
        return np.empty((0, 0))
    if n < 2:
        return np.zeros((H, H))

    # Demean
    means = generated_outcomes.mean(axis=0)  # shape (H,)
    centered = generated_outcomes - means  # shape (n, H)

    # Sample covariance with ddof=1
    omega = (centered.T @ centered) / (n - 1)
    return omega


def compute_eif_cov(
    weights: np.ndarray,
    generated_outcomes: np.ndarray,
    y_hat_mean: np.ndarray,
    n_units: int,
) -> np.ndarray:
    """Per-unit efficient influence function from DR generated outcomes.

    ``EIF_i = sum_j w_j * (Y_hat_{j,i} - y_hat_j)``

    Parameters
    ----------
    weights : ndarray, shape (H,)
        Efficient combination weights.
    generated_outcomes : ndarray, shape (n_units, H)
        Per-unit generated outcomes.
    y_hat_mean : ndarray, shape (H,)
        Sample average of generated outcomes per pair.
    n_units : int
        Total number of units.

    Returns
    -------
    eif : ndarray, shape (n_units,)
        EIF value for every unit.
    """
    H = len(weights)
    if H == 0:
        return np.zeros(n_units)

    # Demeaned generated outcomes: (n_units, H)
    centered = generated_outcomes - y_hat_mean

    # Weighted sum across pairs: (n_units,)
    eif = centered @ weights
    return eif
