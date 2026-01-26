"""
Statistical utilities for performing multiple hypothesis tests.
"""

from __future__ import annotations
import numpy as np
import xarray as xr


def calculate_pval_fdr(
    ps: xr.DataArray | np.ndarray,
    alpha_fdr: float,
) -> float:
    """
    Calculates the adjusted p-value threshold given a
    specified false discovery rate (FDR) control level.

    Wilks, D. S. (2016). “The Stippling Shows Statistically Significant Grid Points”:
    How Research Results are Routinely Overstated and Overinterpreted, and What to Do about It.
    Bulletin of the American Meteorological Society, 97(12), 2263–2273.
    https://doi.org/10.1175/BAMS-D-15-00267.1

    Parameters:
    -----------
    ps : xarray.DataArray | numpy.ndarray
        Array of p-values from local hypothesis tests at each grid point
    alpha_fdr : float
        The specified control level for the false discovery rate

    Returns:
    --------
    pval_fdr : float
        The adjusted p-value threshold
    """

    # Convert DataArray to NumPy array if necessary
    if isinstance(ps, xr.DataArray):
        ps = ps.values

    # Sort the p-values
    ps_sorted = np.sort(ps.flatten())
    n = ps_sorted.shape[0]

    # Select p-values below the FDR control level
    ps_sorted_subset = np.where(ps_sorted <= (np.linspace(1, n, n) / n) * alpha_fdr)[0]

    # Select the maximum p-value below the FDR control level as the FDR adjusted p-value
    if ps_sorted_subset.size > 0:
        p_fdr = ps_sorted[ps_sorted_subset].max()
    else:
        print("no p-values above the FDR control level")
        p_fdr = np.nan

    return p_fdr
