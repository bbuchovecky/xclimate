"""
Statistical regression utilities for xarray fields and numpy arrays.
"""

from __future__ import annotations
import numpy as np
import xarray as xr
import scipy.stats as stats
import scipy.odr as odr


def _ols_single(x, y, alpha=0.05):
    """Core function for computing ordinary least squares (OLS) regression."""
    # Coerce to 1D arrays and drop NaNs
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    # Exit if not enough data
    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Perform the regression output
    ols = stats.linregress(x, y)
    slope = ols.slope
    intercept = ols.intercept
    slope_se = ols.stderr
    intercept_se = ols.intercept_stderr

    # Estimate the degrees of freedom
    p_free = 2
    dof = max(len(x) - p_free, 1)

    # Compute the critical t-value and the confidence interval for the slope
    tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
    slope_ci_halfwidth = tcrit * slope_se

    # Wald test for H0: slope == 0
    t_stat = slope / slope_se
    slope_p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), dof))

    return slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value


def ols_single(x, y, alpha=0.05):
    """Wrapper of OLS for 1D arrays."""
    out = _ols_single(x, y, alpha=alpha)
    slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value = out
    result = dict(
        slope=slope,
        intercept=intercept,
        slope_se=slope_se,
        intercept_se=intercept_se,
        slope_ci_halfwidth=slope_ci_halfwidth,
        slope_p_value=slope_p_value,
    )
    return result


def ols_field(x_da, y_da, sample_dim, alpha=0.05):
    """Wrapper of OLS for multidimensional xarray datasets."""
    args = [x_da, y_da]
    core_dims = [[sample_dim], [sample_dim]]

    out = xr.apply_ufunc(
        _ols_single,
        *args,
        input_core_dims=core_dims,
        output_core_dims=[[], [], [], [], [], []],
        vectorize=True,
        dask="parallelized",
        join="inner",
        output_dtypes=[float, float, float, float, float, float],
        kwargs=dict(alpha=alpha),
    )

    slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value = out
    slope.name = ("slope",)
    intercept.name = "intercept"
    slope_se.name = "slope_se"
    intercept_se.name = "intercept_se"
    slope_ci_halfwidth.name = "slope_ci_halfwidth"
    slope_p_value.name = "slope_p_value"

    return slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value


def _odr_single(x, y, alpha=0.05):
    """Core function for computing orthogonal distance regression (ODR) regression."""
    # Coerce to 1D arrays and drop NaNs
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    # Exit if not enough data
    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Linear function y = B_0 * x + B_1
    def f(B, X):
        return B[0] * X + B[1]

    # Use OLS to get initial parameter values
    ols = stats.linregress(x, y)
    beta0 = np.array([ols.slope, ols.intercept], dtype=float)

    # Create the ODR model and perform the regression
    model = odr.Model(f)
    data = odr.Data(x, y)
    myodr = odr.ODR(data, model, beta0=beta0)
    out = myodr.run()

    # Store the regression output
    slope, intercept = out.beta[0], out.beta[1]
    slope_se, intercept_se = out.sd_beta[0], out.sd_beta[1]

    # Estimate the degrees of freedom
    p_free = 2
    dof = max(len(x) - p_free, 1)

    # Compute the critical t-value and the confidence interval for the slope
    tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
    slope_ci_halfwidth = tcrit * slope_se

    # Wald test for H0: slope == 0
    t_stat = slope / slope_se
    slope_p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), dof))

    return slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value


def odr_single(x, y, alpha=0.05):
    """Wrapper of ODR for 1D arrays"""
    out = _odr_single(x, y, alpha=alpha)
    slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value = out
    result = dict(
        slope=slope,
        intercept=intercept,
        slope_se=slope_se,
        intercept_se=intercept_se,
        slope_ci_halfwidth=slope_ci_halfwidth,
        slope_p_value=slope_p_value,
    )
    return result


def odr_field(x_da, y_da, sample_dim, alpha=0.05):
    """Wrapper of ODR for multidimensional xarray datasets."""
    args = [x_da, y_da]
    core_dims = [[sample_dim], [sample_dim]]

    out = xr.apply_ufunc(
        _odr_single,
        *args,
        input_core_dims=core_dims,
        output_core_dims=[[], [], [], [], [], []],
        vectorize=True,
        dask="parallelized",
        join="inner",
        output_dtypes=[float, float, float, float, float, float],
        kwargs=dict(alpha=alpha),
    )

    slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value = out
    slope.name = ("slope")
    intercept.name = "intercept"
    slope_se.name = "slope_se"
    intercept_se.name = "intercept_se"
    slope_ci_halfwidth.name = "slope_ci_halfwidth"
    slope_p_value.name = "slope_p_value"

    return slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value
