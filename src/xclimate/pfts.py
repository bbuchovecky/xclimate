"""Utility for converting PFT-indexed arrays to (lat, lon) gridded arrays."""

import numpy as np
import xarray as xr

try:
    import sparse
    HAS_SPARSE = True
except ImportError:
    HAS_SPARSE = False


def pft_to_gridcell(ds: xr.Dataset, varname: str, weighted: bool = True) -> xr.DataArray:
    """
    Convert a PFT-level variable to gridcell (lat/lon) spatial representation.
    
    This function takes a variable defined on plant functional types (PFTs) and
    maps it to a regular lat/lon grid using PFT-to-gridcell mapping indices,
    preserving the PFT dimension. Optionally applies PFT weights.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the PFT variable and necessary mapping arrays:
        - pfts1d_ixy: x-index (longitude) for each PFT
        - pfts1d_jxy: y-index (latitude) for each PFT
        - pfts1d_wtgcell: weight of each PFT within its gridcell (only required if weighted=True)
        - lat: latitude coordinates
        - lon: longitude coordinates
    varname : str
        Name of the variable in `ds` to convert from PFT to gridcell representation.
    weighted : bool, optional
        If True (default), multiply values by pfts1d_wtgcell weights.
        If False, use raw PFT values.
    
    Returns
    -------
    xr.DataArray
        The input variable mapped to a grid. Output dimensions are (time, pft, lat, lon)
        if time exists, otherwise (pft, lat, lon). Most gridcells will contain NaN
        since many (pft, lat, lon) combinations don't exist.
    
    Notes
    -----
    - The function automatically detects and converts 1-based indices (CESM/CLM
      convention) to 0-based indexing.
    - The PFT dimension name is automatically detected by searching for 'pft'
      in dimension names (case-insensitive).
    - When weighted=True, values are multiplied by pfts1d_wtgcell before mapping to grid.
    - The PFT dimension is preserved in the output.
    - For memory-efficient sparse array output, use pft_to_gridcell_sparse() instead.
    
    Examples
    --------
    >>> gridcell_var = pft_to_gridcell(clm_dataset, "GPP")
    >>> # Select a specific PFT
    >>> gridcell_var.isel(pft=10).plot()
    >>> # Aggregate across PFTs if desired
    >>> gridcell_var.sum("pft").plot()
    """

    v = ds[varname]

    # Identify the pft dimension name
    pft_dim = next(d for d in v.dims if "pft" in d.lower())

    ixy = ds["pfts1d_ixy"].astype(int)
    jxy = ds["pfts1d_jxy"].astype(int)

    # CESM/CLM convention is often 1-based indices; detect and convert to 0-based
    if ixy.min() == 1 or jxy.min() == 1:
        i0 = ixy - 1
        j0 = jxy - 1
    else:
        i0 = ixy
        j0 = jxy

    lat = ds["lat"]
    lon = ds["lon"]
    nlat = lat.size
    nlon = lon.size
    npft = ds.sizes[pft_dim]

    # Apply weights if requested
    if weighted:
        w = ds["pfts1d_wtgcell"].rename(
            {w_dim: pft_dim for w_dim in ds["pfts1d_wtgcell"].dims}
        )
        w = w.assign_coords({pft_dim: ds[pft_dim]})
        v_weighted = v * w
    else:
        v_weighted = v

    # Create output array filled with NaN
    has_time = "time" in v.dims
    if has_time:
        ntime = v.sizes["time"]
        out_shape = (ntime, npft, nlat, nlon)
        out_dims = ("time", pft_dim, "lat", "lon")
    else:
        out_shape = (npft, nlat, nlon)
        out_dims = (pft_dim, "lat", "lon")
    
    out_data = np.full(out_shape, np.nan, dtype=v.dtype)
    
    # Fill in values at their spatial locations
    if has_time:
        for i in range(npft):
            out_data[:, i, j0.values[i], i0.values[i]] = v_weighted.isel({pft_dim: i}).values
    else:
        for i in range(npft):
            out_data[i, j0.values[i], i0.values[i]] = v_weighted.isel({pft_dim: i}).values
    
    # Create output DataArray
    coords = {pft_dim: ds[pft_dim], "lat": lat, "lon": lon}
    if has_time:
        coords["time"] = v["time"]
    
    out = xr.DataArray(out_data, dims=out_dims, coords=coords)
    
    out.name = varname
    out.attrs.update(v.attrs)
    if weighted:
        out.attrs["note"] = (
            "Converted from PFT vector to (pft, lat, lon) using pfts1d_ixy/jxy and pfts1d_wtgcell."
        )
    else:
        out.attrs["note"] = (
            "Converted from PFT vector to (pft, lat, lon) using pfts1d_ixy/jxy (unweighted)."
        )
    return out


def pft_to_gridcell_sparse(
    ds: xr.Dataset,
    varname: str,
    weighted: bool = True,
) -> xr.DataArray:
    """
    Convert a PFT-level variable to gridcell using sparse arrays (more efficient).
    
    This is a more efficient implementation than pft_to_gridcell() using the
    sparse array library. It creates a sparse array (time, pft, lat, lon)
    which is memory-efficient and works well with Dask for parallel processing.
    
    Based on: https://ncar.github.io/esds/posts/2022/sparse-PFT-gridding/
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the PFT variable and necessary mapping arrays:
        - pfts1d_ixy: x-index (longitude) for each PFT
        - pfts1d_jxy: y-index (latitude) for each PFT
        - pfts1d_wtgcell: weight of each PFT within its gridcell (only required if weighted=True)
        - lat: latitude coordinates
        - lon: longitude coordinates
    varname : str
        Name of the variable in `ds` to convert from PFT to gridcell representation.
    weighted : bool, optional
        If True (default), multiply values by pfts1d_wtgcell weights.
        If False, keep raw PFT values in sparse array.
    
    Returns
    -------
    xr.DataArray
        Sparse array with dimensions (time, pft, lat, lon) if time exists,
        or (pft, lat, lon) otherwise. The underlying array type is sparse.COO
        which only stores non-zero values for memory efficiency.
    
    Raises
    ------
    ImportError
        If the sparse library is not installed.
    
    Notes
    -----
    - Requires the `sparse` library: `pip install sparse` or `conda install sparse`
    - The function automatically detects and converts 1-based indices to 0-based
    - Output is a sparse array that uses much less memory than dense representation
    - Works seamlessly with Dask for parallel processing
    - When weighted=True, values are multiplied by pfts1d_wtgcell weights
    
    Examples
    --------
    >>> sparse_var = pft_to_gridcell_sparse(clm_dataset, "GPP")
    >>> # Plot a specific PFT
    >>> sparse_var.isel(pft=10, time=0).plot()
    >>> # Compute sum over all PFTs (densifies automatically)
    >>> sparse_var.sum("pft")
    """
    if not HAS_SPARSE:
        raise ImportError(
            "The 'sparse' library is required for pft_to_gridcell_sparse. "
            "Install it with: pip install sparse"
        )
    
    v = ds[varname]
    
    # Identify the pft dimension name
    pft_dim = next(d for d in v.dims if "pft" in d.lower())
    
    # Load index arrays and convert to integers
    ixy = ds["pfts1d_ixy"].load().astype(int)
    jxy = ds["pfts1d_jxy"].load().astype(int)
    
    # CESM/CLM convention is 1-based; convert to 0-based
    if ixy.min() == 1:
        ixy = ixy - 1
    if jxy.min() == 1:
        jxy = jxy - 1
    
    lat = ds["lat"]
    lon = ds["lon"]
    nlat = lat.size
    nlon = lon.size
    npft = ds.sizes[pft_dim]
    
    # Helper function to convert numpy arrays to sparse
    def to_sparse(data, pft_indices, jxy, ixy, shape, weights=None):
        """Convert compressed PFT data to sparse COO array."""
        if data.ndim == 1:
            # Single timestep
            coords = np.stack([pft_indices, jxy, ixy], axis=0)
            if weights is not None:
                data = data * weights
        elif data.ndim == 2:
            # Multiple timesteps
            ntime = data.shape[0]
            itime = np.repeat(np.arange(ntime), data.shape[1])
            tostack = [
                np.concatenate([array] * ntime)
                for array in [pft_indices, jxy, ixy]
            ]
            coords = np.stack([itime] + tostack, axis=0)
            if weights is not None:
                weights_broadcast = np.tile(weights, ntime)
                data = data.ravel() * weights_broadcast
        else:
            raise NotImplementedError("Only 1D and 2D arrays supported")
        
        return sparse.COO(
            coords=coords,
            data=data.ravel() if data.ndim > 1 else data,
            shape=data.shape[:-1] + shape if data.ndim > 1 else shape,
            fill_value=np.nan,
        )
    
    # Create PFT index array (0, 1, 2, ..., npft-1)
    pft_indices = np.arange(npft)
    
    # Prepare weights if needed
    weights_array = None
    if weighted:
        weights_array = ds["pfts1d_wtgcell"].load().data
    
    # Apply the conversion using apply_ufunc
    result = xr.apply_ufunc(
        to_sparse,
        v,
        pft_indices,
        jxy,
        ixy,
        kwargs={
            "shape": (npft, nlat, nlon),
            "weights": weights_array,
        },
        input_core_dims=[[pft_dim], [pft_dim], [pft_dim], [pft_dim]],
        output_core_dims=[[pft_dim, "lat", "lon"]],
        dask="parallelized",
        dask_gufunc_kwargs={
            "meta": sparse.COO(np.array([], dtype=v.dtype)),
            "output_sizes": {pft_dim: npft, "lat": nlat, "lon": nlon},
        },
    )
    
    # Add coordinates
    result = result.assign_coords({pft_dim: ds[pft_dim], "lat": lat, "lon": lon})
    
    # Copy attributes
    result.name = varname
    result.attrs.update(v.attrs)
    if weighted:
        result.attrs["note"] = (
            "Converted from PFT to sparse (pft, lat, lon) array with pfts1d_wtgcell weights."
        )
    else:
        result.attrs["note"] = (
            "Converted from PFT to sparse (pft, lat, lon) array (unweighted)."
        )
    
    return result


# # Usage
# ds = xr.open_dataset("your_clm_pft_history.nc")
# da_grid = pft_to_gridcell(ds, "GPP")  # example variable
# da_grid.to_netcdf("GPP_gridded.nc")
