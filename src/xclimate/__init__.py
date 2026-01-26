"""
xclimate: Climate data analysis utilities for xarray and dask.
"""

__version__ = "0.1.0"

from . import plot

from .daskhelper import (
    is_dask_available,
    create_dask_cluster,
)
from .regression import (
    ols_single,
    ols_field,
    odr_single,
    odr_field,
)
from .multiple_testing import (
    calculate_pval_fdr,
)

from .load import (
    load_cesm2le,
    load_coupled_fhist_ppe,
)

__all__ = [
    "plot",
    "is_dask_available",
    "create_dask_cluster",
    "ols_single",
    "ols_field",
    "odr_single",
    "odr_field",
    "calculate_pval_fdr",
    "load_cesm2le",
    "load_coupled_fhist_ppe",
]
