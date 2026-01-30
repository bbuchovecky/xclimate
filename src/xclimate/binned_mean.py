"""Utilities to compute binned means.

This module provides functions for computing binned statistics on xarray DataArrays.
The primary use case is to bin data into quantile-based bins along specified dimensions
and compute conditional means or joint distributions.

Key Functions
-------------
get_quantile_binned_mean : Compute binned means based on quantile bins
get_quantiles : Calculate quantile edges for binning
get_bins : Assign data points to bins based on quantile edges
get_binned_mean2d : Compute 2D binned means
get_joint_hist : Compute 2D joint histogram counts
"""

from __future__ import annotations
from typing import List, Tuple, Union, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

@dataclass
class QBinnedMean:
    """Data structure for 2D quantile binned mean."""
    xb_qedge: xr.DataArray
    yb_qedge: xr.DataArray
    xb_bin: xr.DataArray
    yb_bin: xr.DataArray
    joint_hist: xr.DataArray
    binned_mean: xr.DataArray


def get_quantiles(
    da: xr.DataArray,
    nbin: int,
    qdim: Union[str, List[str]],
) -> xr.DataArray:
    """Compute quantile edges for binning.

    Parameters
    ----------
    da : xr.DataArray
        Input data array to compute quantiles from.
    nbin : int
        Number of bins to create. Will generate nbin+1 quantile edges.
    qdim : str or list of str
        Dimension(s) along which to compute quantiles.

    Returns
    -------
    xr.DataArray
        Quantile edges with dimension 'quantile' ranging from 0 to 1.
        Has length nbin+1 to define nbin bins.

    Notes
    -----
    NaN values are automatically skipped in quantile calculation.
    """
    qedges = np.linspace(0, 1, nbin + 1)
    return da.quantile(qedges, dim=qdim, skipna=True)


def get_bins(da: xr.DataArray, da_edge: xr.DataArray) -> xr.DataArray:
    """Assign data points to bins based on quantile edges.

    Parameters
    ----------
    da : xr.DataArray
        Input data array to bin.
    da_edge : xr.DataArray
        Quantile edges defining bin boundaries. Must have a 'quantile' dimension.

    Returns
    -------
    xr.DataArray
        Integer bin indices for each value in da. Values range from 0 to len(edges)-2.
        NaN values in da remain as NaN in the output.

    Notes
    -----
    Uses np.searchsorted with side='right' to determine bin membership.
    Values are clipped to the valid bin range [0, len(edges)-2].
    """
    assert "quantile" in da_edge.dims
    return xr.apply_ufunc(
        lambda x, edges: np.where(
            np.isnan(x),
            np.nan,
            # searchsorted returns insertion index; subtract 1 to get bin index
            # clip to max bin index to handle edge cases
            np.minimum(np.searchsorted(edges, x, side="right") - 1, len(edges) - 2),
        ),
        da,
        da_edge,
        input_core_dims=[[], ["quantile"]],
        vectorize=True,
        dask="parallelized",
    )


def get_binned_mean2d(
    Z: xr.DataArray,
    xb_bin: xr.DataArray,
    yb_bin: xr.DataArray,
    xnb: int,
    ynb: int,
    agg_dims: Sequence[str],
) -> xr.DataArray:
    """Compute 2D binned means.

    Computes the mean of Z for each combination of (x_bin, y_bin) indices.

    Parameters
    ----------
    Z : xr.DataArray
        Values to average within each bin.
    xb_bin : xr.DataArray
        Integer bin indices for the x-dimension.
    yb_bin : xr.DataArray
        Integer bin indices for the y-dimension.
    xnb : int
        Number of bins in the x-dimension.
    ynb : int
        Number of bins in the y-dimension.
    agg_dims : list of str
        Dimensions to aggregate over when computing binned means.

    Returns
    -------
    xr.DataArray
        2D array of binned means with dimensions ('x_bin', 'y_bin').
        Bins without data are filled with NaN.

    Notes
    -----
    Uses pandas groupby for efficient binned mean computation.
    Output dimensions are ordered as (x_bin, y_bin) with x_bin varying faster.
    """

    def _binned_mean_core(
        Z_data: np.ndarray, x_data: np.ndarray, y_data: np.ndarray
    ) -> np.ndarray:
        """Core function to compute binned means."""
        # Flatten to 1D
        Z_flat = Z_data.flatten()
        x_flat = x_data.flatten()
        y_flat = y_data.flatten()

        # Filter valid values
        valid_mask = np.isfinite(Z_flat) & np.isfinite(x_flat) & np.isfinite(y_flat)

        if not np.any(valid_mask):
            # Return NaN array if all invalid
            return np.full((xnb, ynb), np.nan)

        # Create DataFrame for grouping
        df = pd.DataFrame(
            {
                "Z": Z_flat[valid_mask],
                "xb": x_flat[valid_mask].astype(int),
                "yb": y_flat[valid_mask].astype(int),
            }
        )

        # Group and compute mean
        mean_grouped = df.groupby(["xb", "yb"])["Z"].mean()

        # Unstack to 2D (unstack xb so it becomes columns = x-axis)
        mean_2d = mean_grouped.unstack("xb", fill_value=np.nan)

        # Reindex to ensure full grid (rows=yb, columns=xb)
        mean_2d = mean_2d.reindex(
            index=np.arange(ynb), columns=np.arange(xnb), fill_value=np.nan
        )

        return mean_2d.to_numpy()

    result = xr.apply_ufunc(
        _binned_mean_core,
        Z,
        xb_bin,
        yb_bin,
        input_core_dims=[agg_dims, agg_dims, agg_dims],
        output_core_dims=[["x_bin", "y_bin"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                "x_bin": xnb,
                "y_bin": ynb,
            },
        },
    )

    result = result.assign_coords(
        {
            "x_bin": np.arange(xnb),
            "y_bin": np.arange(ynb),
        }
    )
    result.name = "binned_mean"
    result.attrs = {
        "long_name": f"binned mean {Z.name}",
        "units": Z.attrs.get("units", ""),
    }

    return result


def get_binned_mean2d_with_ci(
    Z: xr.DataArray,
    xb_bin: xr.DataArray,
    yb_bin: xr.DataArray,
    xnb: int,
    ynb: int,
    agg_dims: List[str],
    confidence: float = 0.95,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute 2D binned means with confidence intervals.

    Computes the mean and confidence interval of Z for each combination 
    of (x_bin, y_bin) indices.

    Parameters
    ----------
    Z : xr.DataArray
        Values to average within each bin.
    xb_bin : xr.DataArray
        Integer bin indices for the x-dimension.
    yb_bin : xr.DataArray
        Integer bin indices for the y-dimension.
    xnb : int
        Number of bins in the x-dimension.
    ynb : int
        Number of bins in the y-dimension.
    agg_dims : list of str
        Dimensions to aggregate over when computing binned means.
    confidence : float, optional
        Confidence level (default: 0.95 for 95% CI).

    Returns
    -------
    mean : xr.DataArray
        2D array of binned means with dimensions ('x_bin', 'y_bin').
    ci_lower : xr.DataArray
        2D array of lower confidence bounds.
    ci_upper : xr.DataArray
        2D array of upper confidence bounds.

    Notes
    -----
    Uses the t-distribution for small samples and normal approximation 
    for large samples (n > 30). Bins with fewer than 2 observations 
    have NaN confidence intervals.
    """
    from scipy import stats

    def _binned_stats_core(
        Z_data: np.ndarray, x_data: np.ndarray, y_data: np.ndarray
    ) -> np.ndarray:
        """Core function to compute binned means and confidence intervals."""
        # Flatten to 1D
        Z_flat = Z_data.flatten()
        x_flat = x_data.flatten()
        y_flat = y_data.flatten()

        # Filter valid values
        valid_mask = np.isfinite(Z_flat) & np.isfinite(x_flat) & np.isfinite(y_flat)

        if not np.any(valid_mask):
            # Return NaN arrays if all invalid
            return np.full((3, xnb, ynb), np.nan)

        # Create DataFrame for grouping
        df = pd.DataFrame(
            {
                "Z": Z_flat[valid_mask],
                "xb": x_flat[valid_mask].astype(int),
                "yb": y_flat[valid_mask].astype(int),
            }
        )

        # Group and compute statistics
        grouped = df.groupby(["xb", "yb"])["Z"]
        mean_grouped = grouped.mean()
        std_grouped = grouped.std()
        count_grouped = grouped.count()

        # Unstack to 2D
        mean_2d = mean_grouped.unstack("xb", fill_value=np.nan)
        std_2d = std_grouped.unstack("xb", fill_value=np.nan)
        count_2d = count_grouped.unstack("xb", fill_value=np.nan)

        # Reindex to ensure full grid
        mean_2d = mean_2d.reindex(
            index=np.arange(ynb), columns=np.arange(xnb), fill_value=np.nan
        )
        std_2d = std_2d.reindex(
            index=np.arange(ynb), columns=np.arange(xnb), fill_value=np.nan
        )
        count_2d = count_2d.reindex(
            index=np.arange(ynb), columns=np.arange(xnb), fill_value=np.nan
        )

        # Convert to numpy
        mean_arr = mean_2d.to_numpy()
        std_arr = std_2d.to_numpy()
        count_arr = count_2d.to_numpy()

        # Compute standard error
        sem = std_arr / np.sqrt(count_arr)

        # Compute critical values (use t-distribution for small samples)
        alpha = 1 - confidence
        # Use t-distribution for all; it converges to normal for large n
        t_crit = np.full_like(mean_arr, np.nan)
        for i in range(xnb):
            for j in range(ynb):
                n = count_arr[i, j]
                if n >= 2:  # Need at least 2 observations for CI
                    df_val = n - 1
                    t_crit[i, j] = stats.t.ppf(1 - alpha / 2, df_val)

        # Compute confidence intervals
        margin = t_crit * sem
        ci_lower = mean_arr - margin
        ci_upper = mean_arr + margin

        # Stack results: (3, xnb, ynb)
        return np.stack([mean_arr, ci_lower, ci_upper], axis=0)

    result = xr.apply_ufunc(
        _binned_stats_core,
        Z,
        xb_bin,
        yb_bin,
        input_core_dims=[agg_dims, agg_dims, agg_dims],
        output_core_dims=[["statistic", "x_bin", "y_bin"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                "statistic": 3,
                "x_bin": xnb,
                "y_bin": ynb,
            },
        },
    )

    # Assign coordinates
    result = result.assign_coords(
        {
            "statistic": ["mean", "ci_lower", "ci_upper"],
            "x_bin": np.arange(xnb),
            "y_bin": np.arange(ynb),
        }
    )

    # Split into separate arrays
    mean = result.sel(statistic="mean").drop_vars("statistic")
    ci_lower = result.sel(statistic="ci_lower").drop_vars("statistic")
    ci_upper = result.sel(statistic="ci_upper").drop_vars("statistic")

    # Set attributes
    mean.name = "binned_mean"
    mean.attrs = {
        "long_name": f"binned mean {Z.name}",
        "units": Z.attrs.get("units", ""),
    }
    ci_lower.name = "binned_mean_ci_lower"
    ci_lower.attrs = {
        "long_name": f"binned mean {Z.name} {confidence*100:.0f}% CI lower",
        "units": Z.attrs.get("units", ""),
    }
    ci_upper.name = "binned_mean_ci_upper"
    ci_upper.attrs = {
        "long_name": f"binned mean {Z.name} {confidence*100:.0f}% CI upper",
        "units": Z.attrs.get("units", ""),
    }

    return mean, ci_lower, ci_upper


def get_joint_hist(
    xb_bin: xr.DataArray,
    yb_bin: xr.DataArray,
    xnb: int,
    ynb: int,
    agg_dims: Sequence[str],
) -> xr.DataArray:
    """Compute 2D joint histogram.

    Counts the number of observations in each (x_bin, y_bin) combination.

    Parameters
    ----------
    xb_bin : xr.DataArray
        Integer bin indices for the x-dimension.
    yb_bin : xr.DataArray
        Integer bin indices for the y-dimension.
    xnb : int
        Number of bins in the x-dimension.
    ynb : int
        Number of bins in the y-dimension.
    agg_dims : list of str
        Dimensions to aggregate over when computing histogram.

    Returns
    -------
    xr.DataArray
        2D histogram counts with dimensions ('x_bin', 'y_bin').
        Contains the count of observations in each bin combination.

    Notes
    -----
    Uses np.histogram2d for efficient computation.
    Bins with no observations have a count of 0.
    """

    # Define bin edges (n+1 edges for n bins)
    xb_edge = np.arange(xnb + 1)
    yb_edge = np.arange(ynb + 1)

    def _hist2d_core(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        """Core function to compute 2D histogram."""
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        if not np.any(valid_mask):
            # Return zeros if all NaN (xnb edges creates xnb bins)
            return np.zeros((xnb, ynb))

        hist, _, _ = np.histogram2d(
            x_data[valid_mask],
            y_data[valid_mask],
            bins=[xb_edge, yb_edge],
        )
        return hist.T

    joint_hist = xr.apply_ufunc(
        _hist2d_core,
        xb_bin,
        yb_bin,
        input_core_dims=[agg_dims, agg_dims],
        output_core_dims=[["x_bin", "y_bin"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                "x_bin": xnb,
                "y_bin": ynb,
            },
        },
    )

    joint_dist = joint_hist.assign_coords(
        {
            "x_bin": xb_edge[:-1],
            "y_bin": yb_edge[:-1],
        }
    )
    joint_dist.name = "joint_dist"
    joint_dist.attrs = {"long_name": "joint distribution", "units": "count"}

    return joint_dist


def get_quantile_binned_mean(
    Z: xr.DataArray,
    xb: xr.DataArray,
    yb: xr.DataArray,
    xnb: int,
    ynb: int,
    agg_dims: Sequence[str] = ("gridcell"),
) -> QBinnedMean:
    """Compute quantile-binned means and joint distribution.

    This function bins data into quantile-based bins and computes both the
    joint distribution (histogram) and conditional means of Z within each bin.

    Parameters
    ----------
    Z : xr.DataArray
        Dependent variable to compute conditional means for.
        Must have 'lat' and 'lon' dimensions.
    xb : xr.DataArray
        First binning variable (x-axis).
        Must have 'lat' and 'lon' dimensions.
    yb : xr.DataArray
        Second binning variable (y-axis).
        Must have 'lat' and 'lon' dimensions.
    xnb : int
        Number of quantile bins for xb.
    ynb : int
        Number of quantile bins for yb.
    agg_dims : list of str, optional
        Dimensions to aggregate over when computing statistics.
        Default is ['gridcell'].

    Returns
    -------
    joint_hist : xr.DataArray
        2D histogram showing the count of observations in each (xb, yb) bin.
        Has dimensions ('x_bin', 'y_bin').
    binned_mean : xr.DataArray
        2D array of mean Z values for each (xb, yb) bin.
        Has dimensions ('x_bin', 'y_bin').

    Notes
    -----
    The function performs the following steps:
    1. Stacks lat/lon into a single 'gridcell' dimension
    2. Computes quantile edges for xb and yb
    3. Assigns each observation to a bin
    4. Computes the joint histogram (counts per bin)
    5. Computes the mean of Z within each bin

    Bins are defined by quantiles, so each bin contains approximately
    the same number of observations (for the binning variables).

    Examples
    --------
    >>> # Compute temperature binned by precipitation and soil moisture
    >>> hist, mean_temp = quantile_binned_mean(
    ...     Z=temperature,
    ...     xb=precipitation,
    ...     yb=soil_moisture,
    ...     xnb=10,
    ...     ynb=10
    ... )
    """

    # Stack lat/lon into a single gridcell dimension for aggregation
    Z_s = Z.stack(gridcell=["lat", "lon"])
    xb_s = xb.stack(gridcell=["lat", "lon"])
    yb_s = yb.stack(gridcell=["lat", "lon"])

    # Compute the quantile edges
    xb_qedge = get_quantiles(xb_s, xnb, "gridcell")
    yb_qedge = get_quantiles(yb_s, ynb, "gridcell")

    # Assign each value to a bin index
    xb_bin = get_bins(xb_s, xb_qedge)
    yb_bin = get_bins(yb_s, yb_qedge)

    # Compute the joint histogram
    joint_hist = get_joint_hist(xb_bin, yb_bin, xnb, ynb, agg_dims=agg_dims)

    # Compute the binned mean
    binned_mean = get_binned_mean2d(Z_s, xb_bin, yb_bin, xnb, ynb, agg_dims=agg_dims)

    return QBinnedMean(
        xb_qedge=xb_qedge,
        yb_qedge=yb_qedge,
        xb_bin=xb_bin,
        yb_bin=yb_bin,
        joint_hist=joint_hist,
        binned_mean=binned_mean,
    )
