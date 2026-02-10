"""Utilities to compute bin means.

This module provides functions for computing bin statistics on xarray DataArrays.
The primary use case is to bin data into quantile-based bins along specified dimensions
and compute conditional means or joint distributions.

Key Functions
-------------
get_quantile_bin_mean : Compute bin means based on quantile bins
get_quantiles : Calculate quantile edges for binning
get_bins : Assign data points to bins based on quantile edges
get_bin_mean2d : Compute 2D bin means
get_joint_hist : Compute 2D joint histogram counts
"""

# Suppress Pylint / Pylance style warnings
# pylint: disable=consider-using-enumerate

from __future__ import annotations
from typing import List, Union, Sequence, Optional

import numpy as np
import xarray as xr
from numba import jit


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


def get_bins(da: xr.DataArray, da_edge: xr.DataArray, dim: str = "quantile") -> xr.DataArray:
    """Assign data points to bins based on edges.

    Parameters
    ----------
    da : xr.DataArray
        Input data array to bin.
    da_edge : xr.DataArray
        Edges defining bin boundaries.
    dim : str, optional
        Dimension to compute the bins along.
        Default is 'quantile'.

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
    assert dim in da_edge.dims
    bins =  xr.apply_ufunc(
        lambda x, edges: np.where(
            np.isnan(x),
            np.nan,
            # searchsorted returns insertion index; subtract 1 to get bin index
            # clip to max bin index to handle edge cases
            np.minimum(np.searchsorted(edges, x, side="right") - 1, len(edges) - 2),
        ),
        da,
        da_edge,
        input_core_dims=[[], [dim]],
        vectorize=True,
        dask="parallelized",
    )
    bins.attrs = {"long_name": "bin index"}
    return bins



@jit(nopython=True, cache=True)
def _bin_stats_numba(
    z_flat: np.ndarray, 
    x_flat: np.ndarray, 
    y_flat: np.ndarray,
    xnb: int,
    ynb: int
) -> np.ndarray:
    """
    Numba-optimized function to compute bin mean, standard deviation,
    sample size, and standard error.
    
    Returns
    -------
    np.ndarray
        Array of shape (xnb, ynb, 4) where the last dimension contains:
        [0] mean, [1] standard deviation, [2] sample size, [3] standard error
    """
    sums = np.zeros((xnb, ynb))
    sum_squares = np.zeros((xnb, ynb))
    counts = np.zeros((xnb, ynb))
    
    # Accumulate sums, sum of squares, and counts for each bin
    for i in range(len(z_flat)):
        if np.isfinite(z_flat[i]) and np.isfinite(x_flat[i]) and np.isfinite(y_flat[i]):
            xi = int(x_flat[i])
            yi = int(y_flat[i])
            if 0 <= xi < xnb and 0 <= yi < ynb:
                sums[xi, yi] += z_flat[i]
                sum_squares[xi, yi] += z_flat[i] * z_flat[i]
                counts[xi, yi] += 1
    
    # Compute statistics for each bin
    result = np.full((xnb, ynb, 4), np.nan)
    for xi in range(xnb):
        for yi in range(ynb):
            n = counts[xi, yi]
            if n > 0:
                mean = sums[xi, yi] / n
                result[xi, yi, 0] = mean  # mean
                result[xi, yi, 2] = n  # sample size
                
                if n > 1:
                    # Compute variance using: Var(X) = E[X^2] - E[X]^2
                    variance = (sum_squares[xi, yi] / n) - (mean * mean)
                    # Handle numerical precision issues
                    variance = max(variance, 0)
                    stddev = np.sqrt(variance)
                    result[xi, yi, 1] = stddev  # standard deviation
                    result[xi, yi, 3] = stddev / np.sqrt(n)  # standard error
    
    return result


def get_bin_stats2d(
    z: xr.DataArray,
    x_bin: xr.DataArray,
    y_bin: xr.DataArray,
    xnb: int,
    ynb: int,
    agg_dims: Sequence[str],
) -> xr.DataArray:
    """Compute 2D bin statistics.

    Computes the mean, standard deviation, sample size, and standard error
    of z for each combination of (ibx, iby) indices.

    Parameters
    ----------
    z : xr.DataArray
        Values to average within each bin.
    x_bin : xr.DataArray
        Integer bin indices for the x-dimension.
    y_bin : xr.DataArray
        Integer bin indices for the y-dimension.
    xnb : int
        Number of bins in the x-dimension.
    ynb : int
        Number of bins in the y-dimension.
    agg_dims : list of str
        Dimensions to aggregate over when computing bin means.

    Returns
    -------
    xr.DataArray
        Array of bin means with dimensions ('ibx', 'iby', 'stats').
        The last dimension containts:
            [0] mean, [1] standard deviation, [2] sample size, [3] standard error
        Bins without data are filled with NaN.

    Notes
    -----
    Uses a numba-JIT-compiled function for efficient computation.
    Output dimensions are ordered as (ibx, iby, 4).
    """

    bin_stats = xr.apply_ufunc(
        lambda z_data, x_data, y_data: _bin_stats_numba(
            z_data.flatten(), x_data.flatten(), y_data.flatten(), xnb, ynb
        ),
        z,
        x_bin,
        y_bin,
        input_core_dims=[agg_dims, agg_dims, agg_dims],
        output_core_dims=[["ibx", "iby", "stats"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                "ibx": xnb,
                "iby": ynb,
                "stats": 4,
            },
        },
    )

    bin_stats = bin_stats.assign_coords(
        {
            "ibx": np.arange(xnb),
            "iby": np.arange(ynb),
            "stats": np.arange(4),
        }
    )
    bin_stats = bin_stats.assign_coords(
        stats_name=(
            "stats",
            np.array(["mean", "stddev", "n", "stderr"])
            )
        )
    bin_stats.ibx.attrs = {"long_name": "x bin index"}
    bin_stats.iby.attrs = {"long_name": "y bin index"}
    bin_stats.stats.attrs = {
        "long_name": "bin statistics",
        "description": "[0] mean, [1] standard deviation, [2] sample size, [3] standard error",
    }
    bin_stats.name = "bin_stats"
    bin_stats.attrs = {
        "long_name": f"bin statistics: {z.name}",
        "units": z.attrs.get("units", ""),
    }

    return bin_stats.T


#######################
#### NEW FUNCTIONS ####
#######################


def compute_bin_stats(
    z: xr.DataArray,
    x: xr.DataArray,
    y: xr.DataArray,
    x_edge: xr.DataArray,
    y_edge: xr.DataArray,
    xnb: int,
    ynb: int,
    stack_dims: Sequence[str],
    agg_dims: Sequence[str],
    edge_type: str = "quantile",
) -> xr.Dataset:
    """Common function for computing binned statistics.
    
    Parameters
    ----------
    z : xr.DataArray
        Dependent variable to compute conditional statistics for.
    x : xr.DataArray
        First binning variable (x-axis).
    y : xr.DataArray
        Second binning variable (y-axis).
    x_edge : xr.DataArray
        Bin edges for x with dimension 'iex' or 'qx'.
    y_edge : xr.DataArray
        Bin edges for y with dimension 'iey' or 'qy'.
    xnb : int
        Number of bins for x.
    ynb : int
        Number of bins for y.
    stack_dims : sequence of str
        Dimensions to stack into a single gridcell dimension.
    agg_dims : sequence of str
        Dimensions to aggregate over when computing statistics.
    edge_type : str, optional
        Type of edges: 'quantile' or 'equalwidth'. Default is 'quantile'.
        
    Returns
    -------
    xr.Dataset
        Dataset containing bin_stats, x_bin, y_bin, x_edge, y_edge.
    """
    # Stack into a single gridcell dimension for aggregation
    z_s = z.stack(gridcell=stack_dims)
    x_s = x.stack(gridcell=stack_dims)
    y_s = y.stack(gridcell=stack_dims)

    # Get dimension name from edges
    x_edge_dim = str(list(x_edge.dims)[0])
    y_edge_dim = str(list(y_edge.dims)[0])
    print(f"x_edge_dim: {x_edge_dim}", flush=True)
    print(f"y_edge_dim: {y_edge_dim}", flush=True)

    # Assign each value to a bin index
    x_bin = get_bins(x_s, x_edge, dim=x_edge_dim)
    y_bin = get_bins(y_s, y_edge, dim=y_edge_dim)

    # Compute the bin stats
    bin_stats = get_bin_stats2d(z_s, x_bin, y_bin, xnb, ynb, agg_dims=agg_dims)

    # Create a combined dataset
    if edge_type == "quantile":
        result = xr.merge(
            [
                bin_stats,
                x_bin.unstack().rename("x_bin"),
                y_bin.unstack().rename("y_bin"),
                x_edge.rename("x_edge", **{x_edge_dim: "qx"}),
                y_edge.rename("y_edge", **{y_edge_dim: "qy"}),
            ]
        )
        # Edit coords to match format of equal-width bin stats
        result = result.assign_coords(iex=np.arange(xnb + 1), iey=np.arange(ynb + 1))
        result.qx.attrs["long_name"] = "quantile edges for x bins"
        result.qy.attrs["long_name"] = "quantile edges for y bins"
        result.qx.attrs["units"] = "quantile"
        result.qy.attrs["units"] = "quantile"
    else:  # equalwidth
        result = xr.Dataset(
            {
                "bin_stats": bin_stats,
                "x_bin": x_bin.unstack().rename("x_bin"),
                "y_bin": y_bin.unstack().rename("y_bin"),
                "x_edge": x_edge,
                "y_edge": y_edge,
            }
        )
        result.ex.attrs["long_name"] = "edges for x bins"
        result.ey.attrs["long_name"] = "edges for y bins"

    # Add common metadata
    result.iex.attrs["long_name"] = "index for x edges"
    result.iey.attrs["long_name"] = "index for y edges"
    result.iex.attrs["units"] = "index"
    result.iey.attrs["units"] = "index"

    result.x_bin.attrs["long_name"] = f"indices for x bins: {x.attrs.get('long_name', 'x')}"
    result.y_bin.attrs["long_name"] = f"indices for y bins: {y.attrs.get('long_name', 'y')}"
    result.x_bin.attrs["units"] = "index"
    result.y_bin.attrs["units"] = "index"

    result.x_edge.attrs["long_name"] = f"edges for x bins: {x.attrs.get('long_name', 'x')}"
    result.y_edge.attrs["long_name"] = f"edges for y bins: {y.attrs.get('long_name', 'y')}"
    result.x_edge.attrs["units"] = x.attrs.get('units')
    result.y_edge.attrs["units"] = y.attrs.get('units')

    # Drop unwanted variables
    for var_to_drop in ["ltype", "landunit"]:
        if var_to_drop in result.variables:
            result = result.drop_vars(var_to_drop)

    return result


def get_quantile_bin_stats(
    z: xr.DataArray,
    x: xr.DataArray,
    y: xr.DataArray,
    xnb: int,
    ynb: int,
    quantile_dims: Sequence[str] = ("lat", "lon"),
    agg_dims: Sequence[str] = ("gridcell",),
    x_edge: Optional[xr.DataArray] = None,
    y_edge: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """Compute quantile-binned statistics.

    This function bins data into quantile-based bins and computes comprehensive
    statistics (mean, standard deviation, sample size, standard error) of z
    within each bin combination.

    Parameters
    ----------
    z : xr.DataArray
        Dependent variable to compute conditional statistics for.
        Must have dimensions matching quantile_dims.
    x : xr.DataArray
        First binning variable (x-axis).
        Must have dimensions matching quantile_dims.
    y : xr.DataArray
        Second binning variable (y-axis).
        Must have dimensions matching quantile_dims.
    xnb : int
        Number of quantile bins for x.
    ynb : int
        Number of quantile bins for y.
    quantile_dims : sequence of str, optional
        Dimensions to collapse into a single gridcell dimension for computing
        the quantiles. Default is ('lat', 'lon').
    agg_dims : sequence of str, optional
        Dimensions to aggregate over when computing statistics.
        Default is ('gridcell',).
    x_edge : xr.DataArray, optional
        Pre-computed bin edges for x. If provided, xnb is ignored and edges
        are used directly. Must have a dimension matching the edge dimension
        (e.g., 'qx' or 'ex').
    y_edge : xr.DataArray, optional
        Pre-computed bin edges for y. If provided, ynb is ignored and edges
        are used directly. Must have a dimension matching the edge dimension
        (e.g., 'qy' or 'ey').

    Returns
    -------
    xr.Dataset
        A dataset containing the following variables:
        - bin_stats: 4D array (ibx, iby, stats) containing mean, stddev, n, stderr
        - x_bin: Bin assignments for x variable
        - y_bin: Bin assignments for y variable
        - x_edge: Quantile edges for x binning variable
        - y_edge: Quantile edges for y binning variable

    Notes
    -----
    The function performs the following steps:
    1. Stacks quantile_dims into a single 'gridcell' dimension
    2. Computes quantile edges for x and y (or uses provided edges)
    3. Assigns each observation to a bin
    4. Computes statistics of z within each bin

    Bins are defined by quantiles, so each bin contains approximately
    the same number of observations (for the binning variables).

    Examples
    --------
    >>> # Compute ET statistics binned by LAI and soil moisture quantiles
    >>> result = get_quantile_bin_stats(
    ...     z=et,
    ...     x=lai,
    ...     y=sm,
    ...     xnb=10,
    ...     ynb=10,
    ... )
    >>> mean_et = result.bin_stats.sel(stats_name='mean')
    >>> stderr_et = result.bin_stats.sel(stats_name='stderr')
    
    >>> # Use pre-computed edges
    >>> result = get_quantile_bin_stats(
    ...     z=et,
    ...     x=lai,
    ...     y=sm,
    ...     xnb=10,
    ...     ynb=10,
    ...     x_edge=precomputed_x_edges,
    ...     y_edge=precomputed_y_edges,
    ... )
    """

    # Compute or use provided quantile edges
    if x_edge is None:
        x_s = x.stack(gridcell=quantile_dims)
        x_edge = get_quantiles(x_s, xnb, "gridcell")
    else:
        # Infer xnb from edges
        edge_dim = list(x_edge)[0]
        xnb = len(x_edge[edge_dim]) - 1
        
    if y_edge is None:
        y_s = y.stack(gridcell=quantile_dims)
        y_edge = get_quantiles(y_s, ynb, "gridcell")
    else:
        # Infer ynb from edges
        edge_dim = list(y_edge)[0]
        ynb = len(y_edge[edge_dim]) - 1

    return compute_bin_stats(
        z, x, y, x_edge, y_edge, xnb, ynb, 
        quantile_dims, agg_dims, edge_type="quantile"
    )


def get_equalwidth_bin_stats(
    z: xr.DataArray,
    x: xr.DataArray,
    y: xr.DataArray,
    xnb: int,
    ynb: int,
    stack_dims: Sequence[str] = ("lat", "lon"),
    agg_dims: Sequence[str] = ("gridcell",),
    x_edge: Optional[xr.DataArray] = None,
    y_edge: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """Compute equal-width binned statistics.

    This function bins data into equal-width bins and computes comprehensive
    statistics (mean, standard deviation, sample size, standard error) of z
    within each bin combination.

    Parameters
    ----------
    z : xr.DataArray
        Dependent variable to compute conditional statistics for.
        Must have dimensions matching stack_dims.
    x : xr.DataArray
        First binning variable (x-axis).
        Must have dimensions matching stack_dims.
    y : xr.DataArray
        Second binning variable (y-axis).
        Must have dimensions matching stack_dims.
    xnb : int
        Number of equal-width bins for x.
    ynb : int
        Number of equal-width bins for y.
    stack_dims : sequence of str, optional
        Dimensions to stack into a single gridcell dimension.
        Default is ('lat', 'lon').
    agg_dims : sequence of str, optional
        Dimensions to aggregate over when computing statistics.
        Default is ('gridcell',).
    x_edge : xr.DataArray, optional
        Pre-computed bin edges for x. If provided, xnb is ignored and edges
        are used directly. Must have a dimension named 'ex'.
    y_edge : xr.DataArray, optional
        Pre-computed bin edges for y. If provided, ynb is ignored and edges
        are used directly. Must have a dimension named 'ey'.

    Returns
    -------
    xr.Dataset
        A dataset containing the following variables:
        - bin_stats: 4D array (ibx, iby, stats) containing mean, stddev, n, stderr
        - x_bin: Bin assignments for x variable
        - y_bin: Bin assignments for y variable
        - x_edge: Equal-width bin edges for x binning variable
        - y_edge: Equal-width bin edges for y binning variable

    Notes
    -----
    The function performs the following steps:
    1. Stacks stack_dims into a single 'gridcell' dimension
    2. Computes equal-width bin edges using np.histogram (or uses provided edges)
    3. Assigns each observation to a bin
    4. Computes statistics of z within each bin

    Unlike quantile bins, equal-width bins have uniform spacing in data units,
    but may contain varying numbers of observations.

    Examples
    --------
    >>> # Compute ET statistics binned by LAI and soil moisture (equal-width)
    >>> result = get_equalwidth_bin_stats(
    ...     z=et,
    ...     x=lai,
    ...     y=sm,
    ...     xnb=10,
    ...     ynb=10,
    ... )
    >>> mean_et = result.bin_stats.sel(stats_name='mean')
    >>> stderr_et = result.bin_stats.sel(stats_name='stderr')
    
    >>> # Use pre-computed edges
    >>> result = get_equalwidth_bin_stats(
    ...     z=et,
    ...     x=lai,
    ...     y=sm,
    ...     xnb=10,
    ...     ynb=10,
    ...     x_edge=precomputed_x_edges,
    ...     y_edge=precomputed_y_edges,
    ... )
    """
    # Stack for edge computation
    x_s = x.stack(gridcell=stack_dims)
    y_s = y.stack(gridcell=stack_dims)

    # Compute or use provided equal-width bin edges
    if x_edge is None:
        # # Use xarray operations to stay lazy
        # x_min = x_s.min(skipna=True)
        # x_max = x_s.max(skipna=True)
        # x_edge_vals = np.linspace(x_min, x_max, xnb + 1)
        # x_edge = xr.DataArray(
        #     x_edge_vals,
        #     dims=["ex"], 
        #     coords={"ex": np.arange(xnb + 1)},
        #     attrs=x.attrs,
        # )
        x_s_np = x_s.values[~np.isnan(x_s.values)]
        _, x_edge_vals = np.histogram(x_s_np, bins=xnb)
        x_edge = xr.DataArray(
            x_edge_vals,
            dims=["iex"], 
            coords={"iex": np.arange(xnb + 1)},
            attrs=x.attrs,
        )
    else:
        # Infer xnb from edges
        xnb = len(x_edge.iex) - 1
        
    if y_edge is None:
        # # Use xarray operations to stay lazy
        # y_min = y_s.min(skipna=True)
        # y_max = y_s.max(skipna=True)
        # y_edge_vals = np.linspace(y_min, y_max, ynb + 1)
        # y_edge = xr.DataArray(
        #     y_edge_vals,
        #     dims=["ey"], 
        #     coords={"ey": np.arange(ynb + 1)},
        #     attrs=y.attrs,
        # )
        y_s_np = y_s.values[~np.isnan(y_s.values)]
        _, y_edge_vals = np.histogram(y_s_np, bins=ynb)
        y_edge = xr.DataArray(
            y_edge_vals,
            dims=["iey"], 
            coords={"iey": np.arange(ynb + 1)},
            attrs=y.attrs,
        )
    else:
        # Infer ynb from edges
        ynb = len(y_edge.iey) - 1

    return compute_bin_stats(
        z, x, y, x_edge, y_edge, xnb, ynb, 
        stack_dims, agg_dims, edge_type="equalwidth"
    )



#######################
#### OLD FUNCTIONS ####
#######################



def get_quantile_bin_stats_old(
    z: xr.DataArray,
    x: xr.DataArray,
    y: xr.DataArray,
    xnb: int,
    ynb: int,
    quantile_dims: Sequence[str] = ("lat", "lon"),
    agg_dims: Sequence[str] = ("gridcell",),
) -> xr.Dataset:
    """Compute quantile-binned statistics.

    This function bins data into quantile-based bins and computes comprehensive
    statistics (mean, standard deviation, sample size, standard error) of z
    within each bin combination.

    Parameters
    ----------
    z : xr.DataArray
        Dependent variable to compute conditional statistics for.
        Must have dimensions matching quantile_dims.
    x : xr.DataArray
        First binning variable (x-axis).
        Must have dimensions matching quantile_dims.
    y : xr.DataArray
        Second binning variable (y-axis).
        Must have dimensions matching quantile_dims.
    xnb : int
        Number of quantile bins for x.
    ynb : int
        Number of quantile bins for y.
    quantile_dims : sequence of str, optional
        Dimensions to collapse into a single gridcell dimension for computing
        the quantiles. Default is ('lat', 'lon').
    agg_dims : sequence of str, optional
        Dimensions to aggregate over when computing statistics.
        Default is ('gridcell',).

    Returns
    -------
    xr.Dataset
        A dataset containing the following variables:
        - bin_stats: 4D array (ibx, iby, stats) containing mean, stddev, n, stderr
        - x_bin: Bin assignments for x variable
        - y_bin: Bin assignments for y variable
        - x_quantile: Quantile edges for x binning variable
        - y_quantile: Quantile edges for y binning variable

    Notes
    -----
    The function performs the following steps:
    1. Stacks quantile_dims into a single 'gridcell' dimension
    2. Computes quantile edges for x and y
    3. Assigns each observation to a bin
    4. Computes statistics of z within each bin

    Bins are defined by quantiles, so each bin contains approximately
    the same number of observations (for the binning variables).

    Examples
    --------
    >>> # Compute ET statistics binned by LAI and soil moisture quantiles
    >>> result = get_quantile_bin_stats(
    ...     z=et,
    ...     x=lai,
    ...     y=sm,
    ...     xnb=10,
    ...     ynb=10,
    ... )
    >>> mean_et = result.bin_stats.sel(stats_name='mean')
    >>> stderr_et = result.bin_stats.sel(stats_name='stderr')
    """
 
    # Stack lat/lon into a single gridcell dimension for aggregation
    z_s = z.stack(gridcell=quantile_dims)
    x_s = x.stack(gridcell=quantile_dims)
    y_s = y.stack(gridcell=quantile_dims)

    # Compute the quantile edges
    x_quantile = get_quantiles(x_s, xnb, "gridcell")
    y_quantile = get_quantiles(y_s, ynb, "gridcell")

    # Assign each value to a bin index
    x_bin = get_bins(x_s, x_quantile)
    y_bin = get_bins(y_s, y_quantile)

    # Compute the bin stats
    bin_stats = get_bin_stats2d(z_s, x_bin, y_bin, xnb, ynb, agg_dims=agg_dims)

    # Create a combined dataset
    result = xr.merge(
        [
            bin_stats,
            x_bin.unstack().rename("x_bin"),
            y_bin.unstack().rename("y_bin"),
            x_quantile.rename("x_edge", quantile="qx"),
            y_quantile.rename("y_edge", quantile="qy"),
        ]
    )

    # Edit coords to match format of equal-width bin stats
    result = result.assign_coords(ex=np.arange(xnb + 1), ey=np.arange(ynb + 1))
    result = result.assign_coords(qx=("ex", result.qx.values), qy=("ey", result.qy.values))
    result.ex.attrs["long_name"] = "edges for x bins"
    result.ey.attrs["long_name"] = "edges for y bins"

    # Add metadata
    result.x_bin.attrs["long_name"] = f"indices for x bins: {x.attrs.get('long_name', 'x')}"
    result.y_bin.attrs["long_name"] = f"indices for y bins: {y.attrs.get('long_name', 'y')}"
    result.x_bin.attrs["units"] = "index"
    result.y_bin.attrs["units"] = "index"

    result.x_edge.attrs["long_name"] = f"edges for x bins: {x.attrs.get('long_name', 'x')}"
    result.y_edge.attrs["long_name"] = f"edges for y bins: {y.attrs.get('long_name', 'y')}"
    result.x_edge.attrs["units"] = x.attrs.get('units')
    result.y_edge.attrs["units"] = y.attrs.get('units')

    result.qx.attrs["long_name"] = "quantile edges for x bins"
    result.qy.attrs["long_name"] = "quantile edges for y bins"
   
    for var_to_drop in ["ltype", "landunit"]:
        if var_to_drop in result.variables:
            result = result.drop_vars(var_to_drop)

    return result


def get_equalwidth_bin_stats_old(
    z: xr.DataArray,
    x: xr.DataArray,
    y: xr.DataArray,
    xnb: int,
    ynb: int,
    stack_dims: Sequence[str] = ("lat", "lon"),
    agg_dims: Sequence[str] = ("gridcell",),
) -> xr.Dataset:
    """Compute equal-width binned statistics.

    This function bins data into equal-width bins and computes comprehensive
    statistics (mean, standard deviation, sample size, standard error) of z
    within each bin combination.

    Parameters
    ----------
    z : xr.DataArray
        Dependent variable to compute conditional statistics for.
        Must have dimensions matching stack_dims.
    x : xr.DataArray
        First binning variable (x-axis).
        Must have dimensions matching stack_dims.
    y : xr.DataArray
        Second binning variable (y-axis).
        Must have dimensions matching stack_dims.
    xnb : int
        Number of equal-width bins for x.
    ynb : int
        Number of equal-width bins for y.
    stack_dims : sequence of str, optional
        Dimensions to stack into a single gridcell dimension.
        Default is ('lat', 'lon').
    agg_dims : sequence of str, optional
        Dimensions to aggregate over when computing statistics.
        Default is ('gridcell',).

    Returns
    -------
    xr.Dataset
        A dataset containing the following variables:
        - bin_stats: 4D array (ibx, iby, stats) containing mean, stddev, n, stderr
        - x_bin: Bin assignments for x variable
        - y_bin: Bin assignments for y variable
        - x_edge: Equal-width bin edges for x binning variable
        - y_edge: Equal-width bin edges for y binning variable

    Notes
    -----
    The function performs the following steps:
    1. Stacks stack_dims into a single 'gridcell' dimension
    2. Computes equal-width bin edges using np.histogram
    3. Assigns each observation to a bin
    4. Computes statistics of z within each bin

    Unlike quantile bins, equal-width bins have uniform spacing in data units,
    but may contain varying numbers of observations.

    Examples
    --------
    >>> # Compute ET statistics binned by LAI and soil moisture (equal-width)
    >>> result = get_equalwidth_bin_stats(
    ...     z=et,
    ...     x=lai,
    ...     y=sm,
    ...     xnb=10,
    ...     ynb=10,
    ... )
    >>> mean_et = result.bin_stats.sel(stats_name='mean')
    >>> stderr_et = result.bin_stats.sel(stats_name='stderr')
    """
    
    # Stack lat/lon into a single gridcell dimension for aggregation
    z_s = z.stack(gridcell=stack_dims)
    x_s = x.stack(gridcell=stack_dims)
    y_s = y.stack(gridcell=stack_dims)

    # Filter out nans
    x_s_np = x_s.values[~np.isnan(x_s.values)]
    y_s_np = y_s.values[~np.isnan(y_s.values)]

    # Compute the equal-width bin edges
    _, x_edge = np.histogram(x_s_np, bins=xnb)
    _, y_edge = np.histogram(y_s_np, bins=ynb)

    x_edge = xr.DataArray(
        x_edge,
        dims=["ex"], 
        coords={"ex": np.arange(xnb + 1)},
        attrs=x.attrs,
    )
    y_edge = xr.DataArray(
        y_edge,
        dims=["ey"], 
        coords={"ey": np.arange(ynb + 1)},
        attrs=y.attrs,
    )

    # Assign each value to a bin index
    x_bin = get_bins(x_s, x_edge, dim="ex")
    y_bin = get_bins(y_s, y_edge, dim="ey")

    # Compute the bin mean
    bin_stats = get_bin_stats2d(z_s, x_bin, y_bin, xnb, ynb, agg_dims=agg_dims)

    # Create a combined dataset
    result = xr.Dataset(
        {
            "bin_stats": bin_stats,
            "x_bin": x_bin.unstack().rename("x_bin"),
            "y_bin": y_bin.unstack().rename("y_bin"),
            "x_edge": x_edge,
            "y_edge": y_edge,
        }
    )

    # Add metadata
    result.x_bin.attrs["long_name"] = f"indices for x bins: {x.attrs.get('long_name', 'x')}"
    result.y_bin.attrs["long_name"] = f"indices for y bins: {y.attrs.get('long_name', 'y')}"
    result.x_bin.attrs["units"] = "index"
    result.y_bin.attrs["units"] = "index"

    result.x_edge.attrs["long_name"] = f"edges for x bins: {x.attrs.get('long_name', 'x')}"
    result.y_edge.attrs["long_name"] = f"edges for y bins: {y.attrs.get('long_name', 'y')}"
    result.x_edge.attrs["units"] = x.attrs.get('units')
    result.y_edge.attrs["units"] = y.attrs.get('units')

    result.ex.attrs["long_name"] = "edges for x bins"
    result.ey.attrs["long_name"] = "edges for y bins"
   
    for var_to_drop in ["ltype", "landunit"]:
        if var_to_drop in result.variables:
            result = result.drop_vars(var_to_drop)

    return result



#################################################
#### The following functions are DEPRECIATED ####
#################################################



@jit(nopython=True, cache=True)
def _bin_mean_numba(
    z_flat: np.ndarray, 
    x_flat: np.ndarray, 
    y_flat: np.ndarray,
    xnb: int,
    ynb: int
) -> np.ndarray:
    """
    Numba-optimized function to compute bin mean.
    
    Returns
    -------
    np.ndarray
        Array of shape (xnb, ynb) that contains the mean
    """
    sums = np.zeros((xnb, ynb))
    counts = np.zeros((xnb, ynb))
    
    # Accumulate sums and counts for each bin
    for i in range(len(z_flat)):
        if np.isfinite(z_flat[i]) and np.isfinite(x_flat[i]) and np.isfinite(y_flat[i]):
            xi = int(x_flat[i])
            yi = int(y_flat[i])
            if 0 <= xi < xnb and 0 <= yi < ynb:
                sums[xi, yi] += z_flat[i]
                counts[xi, yi] += 1
    
    # Compute means, setting bins with no data to NaN
    result = np.full((xnb, ynb), np.nan)
    for xi in range(xnb):
        for yi in range(ynb):
            if counts[xi, yi] > 0:
                result[xi, yi] = sums[xi, yi] / counts[xi, yi]
    
    return result


def get_bin_mean2d(
    Z: xr.DataArray,
    xb_bin: xr.DataArray,
    yb_bin: xr.DataArray,
    xnb: int,
    ynb: int,
    agg_dims: Sequence[str],
) -> xr.DataArray:
    """Compute 2D bin means.

    Computes the mean of Z for each combination of (ix_bin, iy_bin) indices.

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
        Dimensions to aggregate over when computing bin means.

    Returns
    -------
    xr.DataArray
        2D array of bin means with dimensions ('ix_bin', 'iy_bin').
        Bins without data are filled with NaN.

    Notes
    -----
    Uses a numba-JIT-compiled function for efficient computation.
    Output dimensions are ordered as (ix_bin, iy_bin).
    """

    bin_mean = xr.apply_ufunc(
        lambda z_data, x_data, y_data: _bin_mean_numba(
            z_data.flatten(), x_data.flatten(), y_data.flatten(), xnb, ynb
        ),
        Z,
        xb_bin,
        yb_bin,
        input_core_dims=[agg_dims, agg_dims, agg_dims],
        output_core_dims=[["ix_bin", "iy_bin"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                "ix_bin": xnb,
                "iy_bin": ynb,
            },
        },
    )

    bin_mean = bin_mean.assign_coords(
        {
            "ix_bin": np.arange(xnb),
            "iy_bin": np.arange(ynb),
        }
    )
    bin_mean.ix_bin.attrs = {"long_name": "x bin index"}
    bin_mean.iy_bin.attrs = {"long_name": "y bin index"}
    bin_mean.name = "bin_mean"
    bin_mean.attrs = {
        "long_name": f"bin mean {Z.name}",
        "units": Z.attrs.get("units", ""),
    }

    return bin_mean.T


@jit(nopython=True, cache=True)
def _hist2d_numba(x_data: np.ndarray, y_data: np.ndarray, xnb: int, ynb: int) -> np.ndarray:
    """Numba-optimized 2D histogram computation."""
    hist = np.zeros((xnb, ynb))
    
    for i in range(len(x_data)):
        if np.isfinite(x_data[i]) and np.isfinite(y_data[i]):
            xi = int(x_data[i])
            yi = int(y_data[i])
            if 0 <= xi < xnb and 0 <= yi < ynb:
                hist[xi, yi] += 1
    
    return hist


def get_joint_hist(
    xb_bin: xr.DataArray,
    yb_bin: xr.DataArray,
    xnb: int,
    ynb: int,
    agg_dims: Sequence[str],
) -> xr.DataArray:
    """Compute 2D joint histogram.

    Counts the number of observations in each (ix_bin, iy_bin) combination.

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
        2D histogram counts with dimensions ('ix_bin', 'iy_bin').
        Contains the count of observations in each bin combination.

    Notes
    -----
    Uses a numba-JIT-compiled function for efficient computation.
    Bins with no observations have a count of 0.
    """

    joint_hist = xr.apply_ufunc(
        lambda x_data, y_data: _hist2d_numba(
            x_data.flatten(), y_data.flatten(), xnb, ynb
        ),
        xb_bin,
        yb_bin,
        input_core_dims=[agg_dims, agg_dims],
        output_core_dims=[["ix_bin", "iy_bin"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                "ix_bin": xnb,
                "iy_bin": ynb,
            },
        },
    )

    joint_hist = joint_hist.assign_coords(
        {
            "ix_bin": np.arange(xnb),
            "iy_bin": np.arange(ynb),
        }
    )
    joint_hist.ix_bin.attrs = {"long_name": "x bin indices"}
    joint_hist.iy_bin.attrs = {"long_name": "y bin indices"}
    joint_hist.name = "joint_hist"
    joint_hist.attrs = {"long_name": "joint distribution", "units": "count"}

    return joint_hist.T


def get_quantile_bin_mean(
    Z: xr.DataArray,
    xb: xr.DataArray,
    yb: xr.DataArray,
    xnb: int,
    ynb: int,
    quantile_dims: Sequence[str] = ("lat", "lon"),
    agg_dims: Sequence[str] = ("gridcell",),
) -> xr.Dataset:
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
    quantile_dims : list of str, optional
        Dimensions to collapse into a single gridcell dimension for computing
        the quantiles. Default is ['lat', 'lon'].
    agg_dims : list of str, optional
        Dimensions to aggregate over when computing statistics.
        Default is ['gridcell'].

    Returns
    -------
    xr.Dataset
        A dataset containing the following variables:
        - xb_qedge: Quantile edges for x binning variable
        - yb_qedge: Quantile edges for y binning variable  
        - xb_bin: Bin assignments for x variable
        - yb_bin: Bin assignments for y variable
        - joint_hist: 2D histogram showing count of observations in each bin
        - bin_mean: 2D array of mean Z values for each bin

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
    >>> # Compute evapotranspiration binned by LAI and soil moisture
    >>> result = get_quantile_bin_mean(
    ...     Z=et,
    ...     xb=lai,
    ...     yb=sm,
    ...     xnb=10,
    ...     ynb=10,
    ... )
    >>> hist = result.joint_hist
    >>> mean_et = result.bin_mean
    """

    # Stack lat/lon into a single gridcell dimension for aggregation
    Z_s = Z.stack(gridcell=quantile_dims)
    xb_s = xb.stack(gridcell=quantile_dims)
    yb_s = yb.stack(gridcell=quantile_dims)

    # Compute the quantile edges
    xb_qedge = get_quantiles(xb_s, xnb, "gridcell")
    yb_qedge = get_quantiles(yb_s, ynb, "gridcell")

    # Assign each value to a bin index
    xb_bin = get_bins(xb_s, xb_qedge)
    yb_bin = get_bins(yb_s, yb_qedge)

    # Compute the joint histogram
    joint_hist = get_joint_hist(xb_bin, yb_bin, xnb, ynb, agg_dims=agg_dims)

    # Compute the bin mean
    bin_mean = get_bin_mean2d(Z_s, xb_bin, yb_bin, xnb, ynb, agg_dims=agg_dims)

    # Create a combined dataset
    result = xr.Dataset(
        {
            "bin_mean": bin_mean,
            "joint_hist": joint_hist,
            "xb_bin": xb_bin.unstack(),
            "yb_bin": yb_bin.unstack(),
            "xb_qedge": xb_qedge.rename({"quantile": "x_qedge"}),
            "yb_qedge": yb_qedge.rename({"quantile": "y_qedge"}),
        }
    )
    result.xb_bin.attrs["long_name"] = "quantile indices for x binning variable"
    result.yb_bin.attrs["long_name"] = "quantile indices for y binning variable"
    result.xb_bin.attrs["units"] = "index"
    result.yb_bin.attrs["units"] = "index"
    result.xb_qedge.attrs["long_name"] = "quantile edges for x binning variable in data units"
    result.yb_qedge.attrs["long_name"] = "quantile edges for y binning variable in data units"
    result.x_qedge.attrs["long_name"] = "quantile edges for x binning variable"
    result.y_qedge.attrs["long_name"] = "quantile edges for y binning variable"
   
    for var_to_drop in ["ltype", "landunit"]:
        if var_to_drop in result.variables:
            result = result.drop_vars(var_to_drop)

    return result


def get_equalwidth_bin_mean(
    Z: xr.DataArray,
    xb: xr.DataArray,
    yb: xr.DataArray,
    xnb: int,
    ynb: int,
    stack_dims: Sequence[str] = ("lat", "lon"),
    agg_dims: Sequence[str] = ("gridcell",),
) -> xr.Dataset:
    
    # Stack lat/lon into a single gridcell dimension for aggregation
    Z_s = Z.stack(gridcell=stack_dims)
    xb_s = xb.stack(gridcell=stack_dims)
    yb_s = yb.stack(gridcell=stack_dims)

    # Filter out nans
    xb_s_np = xb_s.values[~np.isnan(xb_s.values)]
    yb_s_np = yb_s.values[~np.isnan(yb_s.values)]

    # Compute the equal-width bin edges
    _, xb_dedge = np.histogram(xb_s_np, bins=xnb)
    _, yb_dedge = np.histogram(yb_s_np, bins=ynb)

    xb_dedge = xr.DataArray(
        xb_dedge,
        dims=["x_iedge"], 
        coords={"x_iedge": np.arange(xnb + 1)},
        attrs=xb.attrs,
    )
    yb_dedge = xr.DataArray(
        yb_dedge,
        dims=["y_iedge"], 
        coords={"y_iedge": np.arange(ynb + 1)},
        attrs=yb.attrs,
    )

    # Assign each value to a bin index
    xb_bin = get_bins(xb_s, xb_dedge, dim="x_iedge")
    yb_bin = get_bins(yb_s, yb_dedge, dim="y_iedge")

    # Compute the joint histogram
    joint_hist = get_joint_hist(xb_bin, yb_bin, xnb, ynb, agg_dims=agg_dims)

    # Compute the bin mean
    bin_mean = get_bin_mean2d(Z_s, xb_bin, yb_bin, xnb, ynb, agg_dims=agg_dims)

    # Create a combined dataset
    result = xr.Dataset(
        {
            "bin_mean": bin_mean,
            "joint_hist": joint_hist,
            "xb_bin": xb_bin.unstack(),
            "yb_bin": yb_bin.unstack(),
            "xb_dedge": xb_dedge,
            "yb_dedge": yb_dedge,
        }
    )
    result.xb_bin.attrs["long_name"] = "bin indices for x binning variable"
    result.yb_bin.attrs["long_name"] = "bin indices for y binning variable"
    result.xb_bin.attrs["units"] = "index"
    result.yb_bin.attrs["units"] = "index"
    result.xb_dedge.attrs["long_name"] = "equal-width bin edges for x binning variable in data units"
    result.yb_dedge.attrs["long_name"] = "equal-width bin edges for y binning variable in data units"
    result.x_iedge.attrs["long_name"] = "equal-width bin edge indices for x binning variable"
    result.y_iedge.attrs["long_name"] = "equal-width bin edge indices for y binning variable"
   
    for var_to_drop in ["ltype", "landunit"]:
        if var_to_drop in result.variables:
            result = result.drop_vars(var_to_drop)

    return result
