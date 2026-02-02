"""
xclimate: Climate data analysis utilities.
"""

__version__ = "0.1.0"

from . import plot

from . import ppe

from . import regression

from . import multiple_testing

from .daskhelper import (
    is_dask_available,
    create_dask_cluster,
    close_dask_cluster,
    get_ncpus,
    get_memory_per_worker,
)

from .binned_mean import (
    get_quantile_binned_mean,
    get_binned_mean2d,
    get_bins,
    get_joint_hist,
    get_quantiles,
)

from .load import (
    load_cesm2le,
    load_fhist,
    load_coupled_fhist_ppe,
    load_fhist_ppe_grid,
)

__all__ = [
    "plot",
    "ppe",
    "regression",
    "multiple_testing",
    "is_dask_available",
    "create_dask_cluster",
    "close_dask_cluster",
    "get_ncpus",
    "get_memory_per_worker",
    "load_cesm2le",
    "load_fhist",
    "load_coupled_fhist_ppe",
    "load_fhist_ppe_grid",
    "get_quantile_binned_mean",
    "get_binned_mean2d",
    "get_bins",
    "get_joint_hist",
    "get_quantiles",
]
