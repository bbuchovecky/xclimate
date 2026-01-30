"""Facetgrid line plotting utility for DataArrays."""

from __future__ import annotations
from collections.abc import Hashable, Callable
from typing import Tuple, List, Optional

import xarray as xr

import xclimate.ppe
from .helpers import center_axis_at_zero


def _normalize_da_inputs(
    da: xr.DataArray | List[xr.DataArray],
    da_kwargs: dict | List[dict] | None,
    labels: List[str] | None,
) -> tuple[List[xr.DataArray], List[dict], List[str] | None]:
    """
    Normalize DataArray inputs to lists and validate labels.

    Parameters
    ----------
    da : xr.DataArray or List[xr.DataArray]
        Input data array(s)
    da_kwargs : dict, List[dict], or None
        Keyword arguments for plotting
    labels : List[str] or None
        Labels for legend

    Returns
    -------
    tuple
        (da_list, da_kwargs_list, labels)
    """
    # Convert to list if necessary
    da_list = da if isinstance(da, list) else [da]

    # Normalize kwargs
    if da_kwargs is None:
        da_kwargs_list = [{} for _ in range(len(da_list))]
    elif isinstance(da_kwargs, list):
        if len(da_kwargs) != len(da_list):
            raise ValueError("da_kwargs list must have same length as da list")
        da_kwargs_list = da_kwargs
    else:
        da_kwargs_list = [da_kwargs.copy() for _ in range(len(da_list))]

    # Validate labels
    if labels is not None:
        if not isinstance(da, list):
            raise ValueError("labels can only be provided when da is a list")
        if len(labels) != len(da_list):
            raise ValueError("labels must have same length as da list")

    return da_list, da_kwargs_list, labels


def _validate_and_assign_colors(
    da_kwargs_list: List[dict],
    da2_kwargs: dict | None,
) -> tuple[List[dict], dict, str, str]:
    """
    Assign default colors to DataArrays and ensure no conflicts.

    Parameters
    ----------
    da_kwargs_list : List[dict]
        List of kwargs dictionaries for primary DataArrays
    da2_kwargs : dict or None
        Kwargs for secondary DataArray

    Returns
    -------
    tuple
        (da_kwargs_list, da2_kwargs, da_color, da2_color)
    """
    default_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    # Assign colors to primary DataArrays
    for i, kwargs_dict in enumerate(da_kwargs_list):
        if "color" not in kwargs_dict:
            kwargs_dict["color"] = default_colors[i]

    # Determine primary axis color (from first DataArray)
    da_color = da_kwargs_list[0].get("color", "C0")

    # Assign color to secondary DataArray if needed
    _da2_kwargs = da2_kwargs.copy() if da2_kwargs is not None else {}
    if "color" not in _da2_kwargs:
        used_colors = [kwargs.get("color", "C0") for kwargs in da_kwargs_list]
        for candidate in default_colors[1:]:  # Start from C1
            if candidate not in used_colors:
                da2_color = candidate
                break
        else:
            da2_color = "C1"
        _da2_kwargs["color"] = da2_color
    else:
        da2_color = _da2_kwargs["color"]

    return da_kwargs_list, _da2_kwargs, da_color, da2_color


def _configure_panel_axis(
    ax,
    da_list: List[xr.DataArray],
    da_kwargs_list: List[dict],
    da2: xr.DataArray | None,
    da2_kwargs: dict,
    da_color: str,
    da2_color: str,
    dim: str,
    x: Hashable,
    val,
    center_y: bool,
    labels: List[str] | None,
    is_first_panel: bool,
    parse_name: Callable,
) -> Optional[object]:
    """
    Configure a single panel with data, styling, and optional twin axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to configure
    da_list : List[xr.DataArray]
        List of DataArrays to plot
    da_kwargs_list : List[dict]
        Plotting kwargs for each DataArray
    da2 : xr.DataArray or None
        Optional second DataArray for twin axis
    da2_kwargs : dict
        Plotting kwargs for da2
    da_color : str
        Color for primary y-axis
    da2_color : str
        Color for secondary y-axis
    dim : str
        Dimension to select along
    x : Hashable
        X-axis coordinate name
    val
        Value along dim for this panel
    center_y : bool
        Whether to center y-axis at zero
    labels : List[str] or None
        Labels for legend
    is_first_panel : bool
        Whether this is the first panel (for legend)
    parse_name : Callable
        Function to parse the dimension value into a panel title

    Returns
    -------
    ax2 or None
        Twin axis if created, else None
    """
    # Plot additional DataArrays (beyond the first)
    if len(da_list) > 1:
        for i in range(1, len(da_list)):
            da_sel = da_list[i].sel({dim: val})
            _kwargs = da_kwargs_list[i].copy()
            if labels is not None and is_first_panel:
                _kwargs["label"] = labels[i]
            da_sel.plot.line(x=x, ax=ax, **_kwargs)

    # Center y-axis if requested
    if center_y:
        center_axis_at_zero(ax)
        ax.axhline(lw=0.8, c="k", zorder=0)

    # Add twin axis if da2 provided
    ax2 = None
    if da2 is not None:
        ax2 = ax.twinx()
        da2_sel = da2.sel({dim: val})
        da2_sel.plot.line(x=x, ax=ax2, **da2_kwargs)
        ax2.set_title("")
        ax2.set_xlabel("")

        if center_y:
            center_axis_at_zero(ax2)
        
        # Color primary y-axis
        ax.tick_params(axis="y", colors=da_color, which="both")
        ax.yaxis.label.set_color(da_color)

        # Color secondary y-axis
        ax2.tick_params(axis="y", colors=da2_color, which="both")
        ax2.yaxis.label.set_color(da2_color)
    
    # Set panel title
    try:
        val_item = val.item()
    except Exception:
        val_item = val
    ax.set_title(parse_name(val_item), fontsize=8)

    return ax2


def _configure_outer_labels(
    fg,
    axes: List,
    twin_axes: dict,
    show_outer_labels: bool,
    xlabel: str | None,
    ylabel: str | None,
    ylabel2: str | None,
):
    """
    Configure axis labels to show only on outer panels.

    Parameters
    ----------
    fg : FacetGrid
        The xarray FacetGrid object
    axes : List
        List of valid axes
    twin_axes : dict
        Mapping from primary axis to twin axis
    show_outer_labels : bool
        Whether to show labels on outer panels
    xlabel : str or None
        X-axis label
    ylabel : str or None
        Primary y-axis label
    ylabel2 : str or None
        Secondary y-axis label
    """
    if not show_outer_labels:
        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")
        return

    nrows, ncols = fg.axs.shape

    for r in range(nrows):
        for c in range(ncols):
            ax = fg.axs[r, c]
            if ax is None:
                continue

            # Determine edge positions
            is_left = c == 0
            is_right = (c == ncols - 1) or (fg.axs[r, c + 1] is None)
            is_bottom = (r == nrows - 1) or (r < nrows - 1 and fg.axs[r + 1, c] is None)

            # Clear all labels first
            ax.set_xlabel("")
            ax.set_ylabel("")
            if ax in twin_axes:
                twin_axes[ax].set_ylabel("")
                if not is_right:
                    twin_axes[ax].set_yticklabels([])

            # Set labels for outer edges
            if is_bottom and xlabel is not None:
                ax.set_xlabel(xlabel)
            if is_left and ylabel is not None:
                ax.set_ylabel(ylabel)
            if is_right and ax in twin_axes and ylabel2 is not None:
                twin_axes[ax].set_ylabel(ylabel2)


def plot_facetgrid_line(
    da: xr.DataArray | List[xr.DataArray],
    dim: str,
    x: Hashable,
    da_kwargs: dict | List[dict] | None = None,
    da2: xr.DataArray | None = None,
    da2_kwargs: dict | None = None,
    center_y: bool = False,
    ncol: int = 6,
    figsize: Tuple = (14, 8),
    show_outer_labels: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ylabel2: str | None = None,
    labels: List[str] | None = None,
    hspace: float = 0.25,
    parse_name: Callable = xclimate.ppe.get_member_name,
):
    """
    Create faceted line plot panels for visualizing time series or other 1D data across multiple categories.

    This function generates a grid of line plots (facets) based on a specified dimension,
    with each panel showing the temporal or sequential evolution of a DataArray for one category.
    Optionally supports dual y-axes to plot a second DataArray on the right y-axis, and can
    center the y-axis around zero. Custom titles are generated for each panel based on member names.

    Parameters
    ----------
    da : xr.DataArray or List[xr.DataArray]
        Input data array (or list of data arrays) with at least 1D dimension (typically time)
        and the faceting dimension `dim`. Must contain valid data for line plotting. This will
        be plotted on the left y-axis. If a list is provided, each DataArray will be plotted
        on the same y-axis within each panel with different colors.
    dim : str
        Name of the dimension to facet over. Each unique value along this dimension will
        create a separate panel in the grid.
    x : Hashable
        Name of the x-coordinate dimension (typically time or another sequential variable).
    da_kwargs : dict, List[dict], or None, optional
        Dictionary (or list of dictionaries) of keyword arguments to pass to the plotting
        function for `da`. Can include 'color', 'linewidth', 'linestyle', etc. If `da` is
        a list and `da_kwargs` is a single dict, it will be applied to all DataArrays. If
        `da_kwargs` is a list, it should have the same length as `da`. Default is None.
    da2 : xr.DataArray or None, optional
        Optional second data array to plot on the right y-axis. Must have the same
        dimensions as `da`. Default is None.
    da2_kwargs : dict or None, optional
        Dictionary of keyword arguments to pass to the plotting function for `da2`.
        Can include 'color', 'linewidth', 'linestyle', etc. Default is None.
    center_y : bool, optional
        If True, center the y-axis around zero by making the limits symmetric. Applies
        to both left and right y-axes when `da2` is provided. Default is False.
    ncol : int, optional
        Number of columns in the facet grid. Rows are added automatically as needed.
        Default is 6.
    figsize : Tuple, optional
        Figure size in inches as (width, height). Default is (14, 8).
    show_outer_labels : bool, optional
        If True, show x/y axis labels only on the outer edge panels (left column and
        bottom row for x/y, right column for y2). If False, all axis labels are removed.
        Default is False.
    xlabel : str or None, optional
        Label for the x-axis on bottom row panels when show_outer_labels=True.
        Default is None.
    ylabel : str or None, optional
        Label for the left y-axis on left column panels when show_outer_labels=True.
        Default is None.
    ylabel2 : str or None, optional
        Label for the right y-axis on right column panels when show_outer_labels=True
        and `da2` is provided. Default is None.
    labels : List[str] or None, optional
        List of labels for the DataArrays in `da` when `da` is a list. If provided,
        a legend will be created in the bottom right of the figure. Should have the
        same length as `da` list. Default is None.
    hspace : float, optional
        Vertical spacing between subplot rows. Default is 0.25.
    parse_name : Callable, optional
        Function to parse dimension values into subplot titles. Takes a single value
        and returns a string. Default is xclimate.ppe.get_member_name.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, numpy.ndarray]
        A tuple containing:
        - fig : The matplotlib Figure object
        - axs : 2D numpy array of matplotlib Axes objects (may contain None for empty slots)

    Notes
    -----
    - The function uses xclim.ppe.get_member_name() to generate custom titles for each panel.
    - Panel titles are displayed at fontsize 8 for compact presentation.
    - All panels share the same y-axis scaling for easy comparison across facets.
    - When center_y=True, a horizontal line at zero is added for reference.
    - When using dual y-axes, both axes are centered independently if center_y=True.
    - Y-axes are color-coded to match the line colors for clarity.

    Examples
    --------
    >>> # Simple line plot
    >>> fig, axs = plot_facetgrid_line(
    ...     da=temperature_timeseries,
    ...     dim='member',
    ...     x='time',
    ...     xlabel='Year',
    ...     ylabel='Temperature Anomaly [K]',
    ...     color='blue',
    ...     linewidth=1.5
    ... )

    >>> # Multiple DataArrays with legend
    >>> fig, axs = plot_facetgrid_line(
    ...     da=[temp_ts, precip_ts],
    ...     dim='member',
    ...     x='time',
    ...     labels=['Temperature', 'Precipitation'],
    ...     show_outer_labels=True
    ... )

    >>> # Custom title parsing with functools.partial
    >>> from functools import partial
    >>> import xclimate.ppe
    >>> # Customize get_member_name to use specific formatting
    >>> custom_parser = partial(
    ...     xclimate.ppe.get_member_name,
    ...     no_id=True,       # Do not include the ID number
    ...     delimiter='_',    # Set the delimiter
    ...     ppe='fhist',      # Choose the PPE, which decides with YAML member dictionary to load
    ... )
    >>> fig, axs = plot_facetgrid_line(
    ...     da=ensemble_data,
    ...     dim='member',
    ...     x='time',
    ...     parse_name=custom_parser,  # Custom title formatter
    ...     show_outer_labels=True
    ... )
    """
    # Phase 1: Normalize and validate inputs
    da_list, da_kwargs_list, labels = _normalize_da_inputs(da, da_kwargs, labels)
    da_kwargs_list, _da2_kwargs, da_color, da2_color = _validate_and_assign_colors(
        da_kwargs_list, da2_kwargs
    )

    # Phase 2: Add label to first DataArray if labels provided
    if labels is not None:
        da_kwargs_list[0] = {**da_kwargs_list[0], "label": labels[0]}

    # Phase 3: Create initial facet grid with first DataArray
    fg = da_list[0].plot.line(
        x=x,
        col=dim,
        col_wrap=ncol,
        **da_kwargs_list[0],
    )

    fig = fg.fig
    fig.set_size_inches(*figsize)

    # Phase 4: Collect valid axes and coordinate values
    axes = [ax for ax in fg.axs.ravel() if ax is not None]
    names = [name for name in fg.name_dicts.ravel() if name is not None]
    twin_axes = {}

    # Phase 5: Configure each panel
    for i, (ax, nd) in enumerate(zip(axes, names)):
        val = nd[dim]
        is_first_panel = i == 0

        ax2 = _configure_panel_axis(
            ax=ax,
            da_list=da_list,
            da_kwargs_list=da_kwargs_list,
            da2=da2,
            da2_kwargs=_da2_kwargs,
            da_color=da_color,
            da2_color=da2_color,
            dim=dim,
            x=x,
            val=val,
            center_y=center_y,
            labels=labels,
            is_first_panel=is_first_panel,
            parse_name=parse_name,
        )

        if ax2 is not None:
            twin_axes[ax] = ax2

    # Phase 6: Configure axis labels
    _configure_outer_labels(
        fg=fg,
        axes=axes,
        twin_axes=twin_axes,
        show_outer_labels=show_outer_labels,
        xlabel=xlabel,
        ylabel=ylabel,
        ylabel2=ylabel2,
    )

    # Phase 7: Add legend if labels provided
    if labels is not None:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="lower right",
            bbox_to_anchor=(0.95, 0.1),
            fontsize=12,
            ncols=3,
        )

    # Phase 8: Final layout adjustment
    fig.subplots_adjust(hspace=hspace, bottom=0.1)

    return fg.fig, fg.axs
