"""Functions to compute growing season at the gridcell level."""

import numpy as np
import xarray as xr


def growing_season_month(lai: xr.DataArray, nmon: int = 3):
    """
    Compute the center month of the climatological growing season.

    The growing season is defined as the n adjacent months with the greatest
    mean LAI, computed from monthly climatology. This function handles month
    wraparound (Dec -> Jan).

    Parameters
    ----------
    lai : xr.DataArray
        Leaf area index with a 'time' dimension. Should contain monthly data
        that will be grouped by month to create a climatology.
    nmon : int, optional
        Number of months to define the growing season window. Default is 3.
        Must be odd for centered window.

    Returns
    -------
    xr.DataArray
        The center month number (1-12) of the growing season for each gridcell.
        Dimension is typically (member, lat, lon).

    Examples
    --------
    >>> center_month = growing_season(fhist["TLAI_month_1"], nmon=3)
    >>> # Returns 6 (June) where Jun-Jul-Aug has highest mean LAI
    """
    pad = nmon // 2

    # Compute monthly climatology
    x = lai.groupby("time.month").mean()

    # Pad with edge months to handle wraparound
    x_padded = xr.concat(
        [
            x.isel(month=slice(-pad, None)),  # Last month(s)
            x,
            x.isel(month=slice(0, pad)),  # First month(s)
        ],
        dim="month",
    )

    # Find center month of maximum rolling mean
    growsn_center_month = (
        x_padded.rolling(month=nmon, center=True)
        .mean()
        .isel(month=slice(pad, -pad))  # remove padding
        .fillna(-np.inf)  # fill NaN with -inf so argmax doesn't fail
        .argmax(dim="month")
        + 1  # +1 to convert 0-indexed to 1-12
    )

    # Add metadata
    growsn_center_month = growsn_center_month.rename("GROWSNMON")
    growsn_center_month.attrs = {
        "long_name": "center month of the climatological growing season [1=Jan, 12=Dec]",
        "description": f"growing season is defined as the {nmon} adjacent months with the greatest climatological LAI",
    }

    return growsn_center_month


def growing_season_mean(da: xr.DataArray, lai: xr.DataArray, nmon: int = 3):
    """
    Compute annual mean of a variable during the climatological growing season.

    For each year, computes the mean of the input variable over the n-month
    growing season window defined by the LAI climatology. The growing season
    months are determined from `growing_season()` and applied to each year.

    Parameters
    ----------
    da : xr.DataArray
        Variable to average over the growing season. Must have a 'time' dimension
        with monthly data.
    lai : xr.DataArray
        Leaf area index used to define the growing season. Must have a 'time'
        dimension with monthly data.
    nmon : int, optional
        Number of months to define the growing season window. Default is 3.
        Must be odd for centered window.

    Returns
    -------
    xr.DataArray
        Annual mean of `da` during the growing season months, with dimensions
        (year, member, lat, lon) or similar.

    Examples
    --------
    >>> rn_growsn = growing_season_mean(
    ...     fhist["RN_month_1"],
    ...     fhist["TLAI_month_1"],
    ...     nmon=3
    ... )
    >>> # Returns net radiation averaged over the 3-month growing season for each year
    """
    pad = nmon // 2

    # Get the center month for each gridcell
    center = growing_season_month(lai, nmon)

    # Compute distance from center month (handling wraparound: Dec->Jan)
    month_nums = xr.DataArray(
        data=np.arange(1, 13), dims=["month"], coords={"month": np.arange(1, 13)}
    )
    dist = ((month_nums - center + 6) % 12) - 6  # range: -6 to +5

    # Select months within +/-1 of center (the 3-month window)
    mask = np.abs(dist) <= pad

    def _process_year(yearly_data):
        """
        Compute the annual mean of a specified subset of months.

        yearly_data has dims: (time: 12, member: N, lat: M, lon: L)
        """
        # Extract month numbers for this year (1-12)
        months = yearly_data.time.dt.month

        # Get the mask values for these specific months
        mask_for_year = mask.sel(month=months)

        # Apply the mask (set False values to NaN)
        masked_data = yearly_data.where(mask_for_year)

        # Compute mean over time dimension (collapse 12 months to 1 value)
        result = masked_data.mean(dim="time")

        return result

    # Process each year
    growsn_mean = da.groupby("time.year").map(_process_year)

    # Add metadata
    growsn_mean.attrs = {
        "long_name": f"annual growing season mean {da.name}",
        "description": f"growing season is defined as the {nmon} adjacent months with the greatest climatological LAI, computed with xclimate.science.growing_season_month()",
        "units": da.attrs.get("units", ""),
    }

    return growsn_mean
