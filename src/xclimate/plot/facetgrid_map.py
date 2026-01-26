"""Facetgrid map plotting utility for DataArrays."""

from __future__ import annotations
from collections.abc import Hashable, Callable
from typing import Tuple, List

import xarray as xr
import cartopy.crs as ccrs

import xclimate.ppe


def _configure_map_panel(
    ax,
    val,
    projection,
    parse_name: Callable,
) -> None:
    """
    Configure a single map panel with coastlines and custom title.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to configure
    val
        Value along dim for this panel
    projection
        Cartopy projection or None
    parse_name : Callable
        Function to parse the dimension value into a panel title
    """
    # Add coastlines if projection is specified
    if projection:
        ax.coastlines(color="k", lw=0.8)
    
    # Set panel title
    try:
        val_item = val.item()
    except Exception:
        val_item = val
    ax.set_title(parse_name(val_item), fontsize=8)


def _configure_map_outer_labels(
    fg,
    axes: List,
    show_outer_labels: bool,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    """
    Configure axis labels to show only on outer panels.
    
    Parameters
    ----------
    fg : FacetGrid
        The xarray FacetGrid object
    axes : List
        List of valid axes
    show_outer_labels : bool
        Whether to show labels on outer panels
    xlabel : str or None
        X-axis label
    ylabel : str or None
        Y-axis label
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
            is_bottom = (r == nrows - 1) or (r < nrows - 1 and fg.axs[r + 1, c] is None)
            
            # Clear all labels first
            ax.set_xlabel("")
            ax.set_ylabel("")
            
            # Set labels for outer edges
            if is_bottom and xlabel is not None:
                ax.set_xlabel(xlabel)
            if is_left and ylabel is not None:
                ax.set_ylabel(ylabel)


def plot_facetgrid_map(
    da: xr.DataArray,
    dim: str,
    label: str,
    x: Hashable = "lon",
    y: Hashable = "lat",
    ncol: int = 6,
    figsize: Tuple = (16, 8),
    projection = ccrs.Robinson(),
    show_outer_labels: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    hspace: float = 0.25,
    parse_name: Callable = xclimate.ppe.get_member_name,
    **kwargs,
):
    """
    Create faceted map panels for visualizing spatial data across multiple categories.
    
    This function generates a grid of map plots (facets) based on a specified dimension,
    with each panel showing the spatial distribution of a DataArray for one category.
    Coastlines are automatically added when using a projection, and custom titles are
    generated for each panel based on member names.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with at least 2D spatial dimensions (lon/lat) and the faceting
        dimension `dim`. Must contain valid data for pcolormesh plotting.
    dim : str
        Name of the dimension to facet over. Each unique value along this dimension will
        create a separate panel in the grid.
    label : str
        Label for the colorbar.
    x : Hashable, optional
        Name of the x-coordinate dimension (typically longitude). Default is "lon".
    y : Hashable, optional
        Name of the y-coordinate dimension (typically latitude). Default is "lat".
    ncol : int, optional
        Number of columns in the facet grid. Rows are added automatically as needed.
        Default is 6.
    figsize : Tuple, optional
        Figure size in inches as (width, height). Default is (16, 8).
    projection : cartopy.crs projection or None, optional
        Cartographic projection for the map subplots. If None, no projection is used.
        Default is ccrs.Robinson().
    show_outer_labels : bool, optional
        If True, show x/y axis labels only on the outer edge panels (left column and
        bottom row). If False, all axis labels are removed. Default is False.
    xlabel : str or None, optional
        Label for the x-axis on bottom row panels when show_outer_labels=True.
        Default is None.
    ylabel : str or None, optional
        Label for the y-axis on left column panels when show_outer_labels=True.
        Default is None.
    hspace : float, optional
        Vertical spacing between subplot rows. Default is 0.25.
    parse_name : Callable, optional
        Function to parse dimension values into subplot titles. Takes a single value
        and returns a string. Default is xclimate.ppe.get_member_name.
    **kwargs
        Additional keyword arguments passed to xarray.plot.pcolormesh, such as
        'cmap', 'vmin', 'vmax', 'levels', etc.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, numpy.ndarray]
        A tuple containing:
        - fig : The matplotlib Figure object
        - axs : 2D numpy array of matplotlib Axes objects (may contain None for empty slots)
        
    Notes
    -----
    - The function uses xclim.get_member_name() to generate custom titles for each panel.
    - Coastlines are automatically added with black lines (linewidth=0.8) when a projection
      is specified.
    - The colorbar is positioned horizontally at the bottom of the figure.
    - Panel titles are displayed at fontsize 8 for compact presentation.
    
    Examples
    --------
    >>> # Simple map plot
    >>> fig, axs = plot_facetgrid_map(
    ...     da=temperature_data,
    ...     dim='member',
    ...     label='Temperature [°C]',
    ...     cmap='RdBu_r',
    ...     vmin=-2,
    ...     vmax=2
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
    >>> fig, axs = plot_facetgrid_map(
    ...     da=spatial_data,
    ...     dim='member',
    ...     label='Precipitation Anomaly [mm/day]',
    ...     parse_name=custom_parser,  # Custom title formatter
    ...     cmap='BrBG',
    ...     show_outer_labels=True
    ... )
    """
    # Phase 1: Configure projection and subplot settings
    if projection:
        transform = ccrs.PlateCarree()
        subplot_kws = {"projection": projection}
    else:
        transform = None
        subplot_kws = None

    # Phase 2: Create initial facet grid with colorbar
    fg = da.plot.pcolormesh(
        x=x,
        y=y,
        col=dim,
        col_wrap=ncol,
        transform=transform,
        subplot_kws=subplot_kws,
        add_colorbar=True,
        cbar_kwargs={
            "label": label,
            "location": "bottom",
            "orientation": "horizontal",
            "fraction": 0.03,
            "pad": 0.15,
            "extend": "both",
        },
        **kwargs,
    )

    fig = fg.fig
    fig.set_size_inches(*figsize)

    # Phase 3: Collect valid axes and coordinate values
    axes = [ax for ax in fg.axs.ravel() if ax is not None]
    names = [name for name in fg.name_dicts.ravel() if name is not None]

    # Phase 4: Configure each panel
    for ax, nd in zip(axes, names):
        val = nd[dim]
        _configure_map_panel(ax, val, projection, parse_name)

    # Phase 5: Configure axis labels
    _configure_map_outer_labels(
        fg=fg,
        axes=axes,
        show_outer_labels=show_outer_labels,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    # Phase 6: Final layout adjustment
    fig.subplots_adjust(hspace=hspace, bottom=0.1)

    return (fg.fig, fg.axs)