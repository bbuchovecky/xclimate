"""Time coord utilities."""

from __future__ import annotations

import xarray as xr

def shift_time(ds: xr.Dataset) -> xr.Dataset:
    """Shifts time coordinate from [startyear-02, endyear-01] to [startyear-01, (endyear-1)-12]"""
    assert "time" in ds.dims
    if (ds.time[0].dt.month.item() == 2) and (ds.time[-1].dt.month.item() == 1):
        new_time = xr.date_range(
            start=str(ds.time[0].dt.year.item()) + "-01",
            end=str(ds.time[-1].dt.year.item() - 1) + "-12",
            freq="MS",
            calendar="noleap",
            use_cftime=True,
        )
        return ds.assign_coords(time=new_time)
    return ds
