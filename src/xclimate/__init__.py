"""
xclimate: Climate data analysis utilities.
"""

__version__ = "0.1.0"

from . import plot

from . import ppe

from .daskhelper import (
    is_dask_available,
    create_dask_cluster,
    close_dask_cluster,
)

from .binned_mean import (
    get_quantile_binned_mean,
    get_binned_mean2d,
    get_bins,
    get_joint_hist,
    get_quantiles,
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
    "close_dask_cluster",
    "ols_single",
    "ols_field",
    "odr_single",
    "odr_field",
    "calculate_pval_fdr",
    "load_cesm2le",
    "load_coupled_fhist_ppe",
    "get_quantile_binned_mean",
    "get_binned_mean2d",
    "get_bins",
    "get_joint_hist",
    "get_quantiles",
]
