"""Model output loading utilities."""

from __future__ import annotations
from typing import List, Sequence
import warnings
from pathlib import Path
import re

import numpy as np
import xarray as xr

from .time_coord import shift_time


CESM2_COMPONENT_MAP = {
    "atm": "cam",
    "lnd": "clm2",
    "cice": "cice",
    "mosart": "rof",
    "glc": "cism",
    "cpl": "cpl",
}


VARIABLES_TO_DROP = {
    "atm": ["gw", "hyam", "hybm", "P0", "ilev"],
    "lnd": [
        "nbedrock",
        "ZSOI",
        "DZSOI",
        "WATSAT",
        "SUCSAT",
        "BSW",
        "HKSAT",
        "ZLAKE",
        "DZLAKE",
        "time_written",
        "date_written",
    ],
    "cice": [],
    "mosart": [],
    "glc": [],
    "cpl": [],
}


_CESM_TSERIES_RANGE_RE = re.compile(r"\.(\d{8})-(\d{8})\.nc$")


def _filter_files_by_timerange(
    files: Sequence[Path],
    start: str | None,
    end: str | None,
) -> list[Path]:
    """
    Filter CESM timeseries files by overlap with [start, end].

    Assumes filenames end with: .YYYYMMDD-YYYYMMDD.nc
    """
    if start is None and end is None:
        return list(files)

    def _as_int_date(s: str) -> int:
        # Accept 'YYYY-MM', 'YYYY-MM-DD', or 'YYYYMMDD'
        s = s.replace("-", "")
        if len(s) == 6:
            s = s + "01"
        return int(s)

    start_i = _as_int_date(start) if start is not None else None
    end_i = _as_int_date(end) if end is not None else None

    kept: list[Path] = []
    for f in files:
        m = _CESM_TSERIES_RANGE_RE.search(f.name)
        if not m:
            # If pattern doesn't match, keep to be safe.
            kept.append(f)
            continue
        f0 = int(m.group(1))
        f1 = int(m.group(2))

        if start_i is not None and f1 < start_i:
            continue
        if end_i is not None and f0 > end_i:
            continue
        kept.append(f)

    return kept


def load_coupled_fhist_ppe(
    variable: str | List[str],
    gcomp: str,
    frequency: str,
    stream: str = "*",
    drop_vars: tuple = (),
    keep_var_only: bool = False,
    drop_outliers: Sequence[int] = (13, 28),
    members: Sequence[int] | None = None,
    time_slice: slice | tuple[str, str] | None = None,
    chunk: bool | dict = False,
) -> xr.Dataset:
    """
    Load variables from the coupled F-HIST PPE.

    Parameters
    ----------
    variable : str or list of str
        The name(s) of the variable(s) to load.
    gcomp : str
        The component of the model the variable belongs to (e.g., "atm", "lnd").
    frequency : str
        The frequency of the output (e.g., "month_1").
    stream : str, optional
        The stream of the output. Defaults to "*".
    drop_vars : tuple, optional
        A list of variables to drop from the dataset. Defaults to ().
    keep_var_only : bool, optional
        If True, only keep the specified variable and its coordinates.
        Defaults to False.
    drop_outliers : list of str
        List of outlier members to drop. Defaults to 13 and 28 which have
        unreasonable LAI and ET.
    members : sequence of int, optional
        If provided, only load these PPE members (0-28). Useful for quick tests.
        Defaults to loading all members (except those in drop_outliers).
    time_slice : slice or (start, end) tuple, optional
        If provided, filter the underlying timeseries files to those overlapping the
        requested range and then subset the dataset by time. This can dramatically
        reduce load time for large daily outputs. Accepts:
        - slice('YYYY-MM', 'YYYY-MM') or slice('YYYY-MM-DD','YYYY-MM-DD')
        - (start, end) tuple of strings
    chunk : bool or dict, optional
        If True, chunk along time. If dict, chunk using the provided mapping.
        Defaults to False.

    Returns
    -------
    xr.Dataset
        A dataset containing the loaded data, with an added "member" dimension.
    """
    if isinstance(variable, str):
        variables = [variable]
    else:
        variables = variable

    xr.set_options(use_new_combine_kwarg_defaults=True)

    rootpath = Path("/glade/campaign/univ/uwas0155/ppe/historical/coupled_simulations")
    basename = "f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE"
    scomp = CESM2_COMPONENT_MAP[gcomp]

    member_ids = []
    member_datasets = []
    iter_members = list(range(29)) if members is None else list(members)
    for m in iter_members:
        if m in drop_outliers:
            continue

        member_ids.append(m)

        ms = str(m).zfill(3)
        mcase = f"{basename}.{ms}"

        variable_datasets = []
        for v in variables:
            files = sorted(
                rootpath.glob(
                    f"{mcase}/{gcomp}/proc/tseries/{frequency}/{mcase}.{scomp}.{stream}.{v}.*.nc"
                )
            )
            if time_slice is not None:
                if isinstance(time_slice, tuple):
                    start, end = time_slice
                elif isinstance(time_slice, slice):
                    start, end = time_slice.start, time_slice.stop
                else:
                    raise TypeError("time_slice must be a slice or (start, end) tuple")
                files = _filter_files_by_timerange(files, start=start, end=end)

            if files:
                ds = xr.open_mfdataset(
                    files,
                    decode_timedelta=False,
                    drop_variables=VARIABLES_TO_DROP[gcomp],
                    coords="minimal",
                )
                variable_datasets.append(ds)

        if variable_datasets:
            member_ds = xr.merge(variable_datasets)
            if "time" in member_ds.dims:
                member_ds = shift_time(member_ds)
                if time_slice is not None:
                    if isinstance(time_slice, tuple):
                        start, end = time_slice
                    else:
                        start, end = time_slice.start, time_slice.stop
                    member_ds = member_ds.sel(time=slice(start, end))

            if chunk:
                if isinstance(chunk, dict):
                    member_ds = member_ds.chunk(chunk)
                else:
                    member_ds = member_ds.chunk({"time": -1})
            member_datasets.append(member_ds)

    if not member_datasets:
        return xr.Dataset()

    combined_ds = xr.concat(member_datasets, dim="member").assign_coords(member=np.array(member_ids))

    if drop_vars:
        existing_vars_to_drop = [
            var
            for var in drop_vars
            if var in combined_ds.coords or var in combined_ds.data_vars
        ]
        if existing_vars_to_drop:
            combined_ds = combined_ds.drop_vars(existing_vars_to_drop)

    if keep_var_only:
        found_variables = [v for v in variables if v in combined_ds.data_vars]
        combined_ds = combined_ds[found_variables]

    return shift_time(combined_ds)


def load_cesm2le(
    variable: str | list[str],
    gcomp: str,
    frequency: str,
    stream: str,
    experiment: str = "historical",
    drop_vars: tuple = (),
    keep_cosp: bool = False,
    keep_var_only: bool = False,
    chunk: bool | dict = False,
) -> xr.Dataset:
    """
    Load variables from the CESM2 Large Ensemble.

    Parameters
    ----------
    variable : str or list of str
        The name(s) of the variable(s) to load.
    gcomp : str
        The component of the model the variable belongs to (e.g., "atm", "lnd").
    frequency : str
        The frequency of the output (e.g., "month_1").
    stream : str
        The history tape stream of the output (e.g., "h0")
    experiment : str, optional
        The experiment to load data from. Can be "historical" or "ssp370".
        Defaults to "historical".
    drop_vars : list or tuple, optional
        A list of variables to drop from the dataset.
        Defaults to ('gw', 'hyam', 'hybm', 'P0', 'ilev', 'slat', 'slon', 'w_stag').
    keep_cosp : bool, optional
        If False, remove all COSP-related variables and coordinates.
        Defaults to False.
    keep_var_only : bool, optional
        If True, only keep the specified variable and its coordinates.
        Defaults to False.
    chunk : bool or dict, optional
        If True, rechunk the dataset to a default chunking scheme.
        If a dict, rechunk the dataset to the specified chunk sizes.
        Defaults to False.

    Returns
    -------
    xr.Dataset
        A dataset containing the loaded data, with an added "member" dimension.
    """
    if isinstance(variable, str):
        variables = [variable]
    else:
        variables = variable

    scomp = CESM2_COMPONENT_MAP[gcomp]
    exp_dict = {
        "historical": "BHISTsmbb",
        "ssp370": "BSSP370smbb",
    }

    xr.set_options(use_new_combine_kwarg_defaults=True)

    rootdir = Path("/glade/campaign/collections/gdex/data/d651056/CESM2-LE")

    member_datasets = []
    for m in range(1, 21):
        member_id = str(m).zfill(3)
        variable_datasets = []
        for var_name in variables:
            tsdir = rootdir / f"{gcomp}/proc/tseries/{frequency}/{var_name}"
            file_pattern = f"b.e21.{exp_dict[experiment]}.f09_g17.LE2-*.{member_id}.{scomp}.{stream}.{var_name}.*.nc"
            files = sorted(list(tsdir.glob(file_pattern)))
            if files:
                ds = xr.open_mfdataset(files, decode_timedelta=False, drop_variables=VARIABLES_TO_DROP[gcomp], coords="minimal")
                variable_datasets.append(ds)

        if variable_datasets:
            # Merge variables for the current member
            member_ds = xr.merge(variable_datasets)
            member_datasets.append(member_ds)

    if not member_datasets:
        return xr.Dataset()

    # Concatenate all member datasets
    combined_ds = xr.concat(member_datasets, dim="member", coords="minimal")
    combined_ds = combined_ds.assign_coords(member=range(1, len(member_datasets) + 1))

    if drop_vars:
        existing_vars_to_drop = [
            var
            for var in drop_vars
            if var in combined_ds.coords or var in combined_ds.data_vars
        ]
        if existing_vars_to_drop:
            combined_ds = combined_ds.drop_vars(existing_vars_to_drop)

    if not keep_cosp:
        cosp_vars = [var for var in combined_ds.variables if "cosp" in var]
        if cosp_vars:
            combined_ds = combined_ds.drop_vars(cosp_vars)

    if keep_var_only:
        # Ensure we only keep variables that were actually found and loaded
        found_variables = [v for v in variables if v in combined_ds.data_vars]
        combined_ds = combined_ds[found_variables]

    if chunk:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if isinstance(chunk, dict):
                combined_ds = combined_ds.chunk(chunk)
            else:
                # Default rechunking: chunk time dimension fully
                combined_ds = combined_ds.chunk({"time": -1})

    return shift_time(combined_ds)
