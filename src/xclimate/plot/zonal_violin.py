"""Zonal mean plotting with violin distributions."""

from __future__ import annotations
from typing import Tuple, List, Optional, Dict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .helpers import whiten


# Constants
DEFAULT_MEMBER_LINE_ALPHA = 0.3
DEFAULT_MEMBER_LINE_WIDTH = 1
DEFAULT_VIOLIN_WIDTH = 15
DEFAULT_VIOLIN_ALPHA = 0.75
DEFAULT_VIOLIN_EDGE_WIDTH = 1


def _compute_latitude_band_means(
    da_zm: xr.DataArray,
    lat_coord: str,
    lat_bands: List[Tuple[float, float]],
) -> List[xr.DataArray]:
    """
    Compute means over specified latitude bands.
    
    Parameters
    ----------
    da_zm : xr.DataArray
        Input data array with latitude dimension
    lat_coord : str
        Name of latitude coordinate
    lat_bands : List[Tuple[float, float]]
        List of (lat_min, lat_max) tuples defining bands
        
    Returns
    -------
    List[xr.DataArray]
        List of band means, one per band
    """
    band_means = []
    for lat_min, lat_max in lat_bands:
        band_data = da_zm.sel({lat_coord: slice(lat_min, lat_max)})
        band_mean = band_data.mean(dim=lat_coord)
        band_means.append(band_mean)
    return band_means


def _get_latitude_band_centers(
    lat_bands: List[Tuple[float, float]]
) -> List[float]:
    """
    Calculate center latitude for each band.
    
    Parameters
    ----------
    lat_bands : List[Tuple[float, float]]
        List of (lat_min, lat_max) tuples
        
    Returns
    -------
    List[float]
        Center latitude for each band
    """
    return [(lat_min + lat_max) / 2 for lat_min, lat_max in lat_bands]


def _plot_member_lines(
    ax,
    da_zm: xr.DataArray,
    member_coord: str,
    lat_coord: str,
    color: str | tuple,
    label: Optional[str] = None,
) -> None:
    """
    Plot zonal mean lines for all ensemble members.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    da_zm : xr.DataArray
        Zonal mean data with member dimension
    member_coord : str
        Name of member coordinate
    lat_coord : str
        Name of latitude coordinate
    color : str
        Color for the lines
    label : str or None, optional
        Label for legend (applied to first member only)
    """
    for i, member in enumerate(da_zm[member_coord]):
        lab = label if i == 0 else None
        da_zm.sel({member_coord: member}).plot(
            ax=ax,
            x=lat_coord,
            color=color,
            alpha=DEFAULT_MEMBER_LINE_ALPHA,
            lw=DEFAULT_MEMBER_LINE_WIDTH,
            label=lab,
            _labels=False,
        )


def _add_violin_plots(
    ax,
    band_means: List[xr.DataArray],
    band_centers: List[float],
    color: str,
    violin_width: float = DEFAULT_VIOLIN_WIDTH,
) -> None:
    """
    Add violin plots at specified latitude positions.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    band_means : List[xr.DataArray]
        List of band mean data arrays
    band_centers : List[float]
        Center latitude for each band
    member_coord : str
        Name of member coordinate
    color : str
        Color for violin plots
    violin_width : float, optional
        Width of violin plots in latitude degrees
    """
    for band_mean, lat_center in zip(band_means, band_centers):
        # Extract values across members
        values = band_mean.values
            
        vp = ax.violinplot(
            [values],
            [lat_center],
            vert=True,
            widths=violin_width,
            showmeans=False,
            showextrema=True,
            showmedians=True,
        )
        
        # Style violin plot
        vp["bodies"][0].set(facecolor=color, alpha=DEFAULT_VIOLIN_ALPHA)
        vp["cbars"].set(linewidth=0)
        vp["cmedians"].set(linewidth=DEFAULT_VIOLIN_EDGE_WIDTH, color=color)
        vp["cmins"].set(linewidth=DEFAULT_VIOLIN_EDGE_WIDTH, color=color)
        vp["cmaxes"].set(linewidth=DEFAULT_VIOLIN_EDGE_WIDTH, color=color)


def plot_zonal_violin(
    da_zm: xr.DataArray,
    member_coord: str = "member",
    lat_coord: str = "lat",
    lat_bands: Optional[List[Tuple[float, float]]] = None,
    color: Optional[str] = None,
    vp_color: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: str = "Latitude",
    title: str = "",
    figsize: Tuple[float, float] = (10, 6),
    ylim: Optional[Tuple[float, float]] = None,
    violin_width: float = DEFAULT_VIOLIN_WIDTH,
):
    """
    Plot zonal mean profiles with violin distributions at latitude bands.
    
    This function creates a visualization showing individual ensemble member zonal
    mean profiles as transparent lines, overlaid with violin plots that show the
    distribution of values across ensemble members within specified latitude bands.

    Parameters
    ----------
    da_zm : xr.DataArray
        Zonal mean data array with latitude and member dimensions. Should contain
        data varying along latitude for multiple ensemble members.
    member_coord : str, optional
        Name of the ensemble member coordinate dimension. Default is "member".
    lat_coord : str, optional
        Name of the latitude coordinate dimension. Default is "lat".
    lat_bands : List[Tuple[float, float]] or None, optional
        List of latitude band tuples (lat_min, lat_max) where violin plots should
        be placed. If None, defaults to three bands: (-90, -30), (-30, 30), (30, 90).
    color : str or None, optional
        Color for lines and violin plots. If None, uses default matplotlib color 'C0'.
    vp_color : str or None, optional
        Color for violin plots. If None, uses 'color' argument.
    ylabel : str or None, optional
        Label for the y-axis. If None, uses the DataArray's name or units.
    xlabel : str, optional
        Label for the x-axis. Default is "Latitude".
    title : str, optional
        Title for the plot. Default is an empty string.
    figsize : Tuple[float, float], optional
        Figure size in inches as (width, height). Default is (10, 6).
    ylim : Tuple[float, float] or None, optional
        Y-axis limits as (min, max). If None, uses automatic scaling.
    violin_width : float, optional
        Width of violin plots in latitude degrees. Default is 5.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing the figure and axes objects.

    Notes
    -----
    - Individual ensemble members are plotted as semi-transparent lines for visual clarity.
    - Violin plots show the full distribution (median, quartiles, and density) of ensemble
      members within each latitude band.
    - The violin plots are positioned at the center latitude of each band.
    - Useful for comparing zonal mean structures across ensemble members and identifying
      regions of high/low ensemble spread.

    Examples
    --------
    >>> # Simple zonal violin plot with default latitude bands
    >>> fig, ax = plot_zonal_violin(
    ...     da=zonal_temperature,
    ...     member_coord='member',
    ...     ylabel='Temperature [K]',
    ...     title='Zonal Mean Temperature Distribution'
    ... )
    
    >>> # Custom latitude bands
    >>> fig, ax = plot_zonal_violin(
    ...     da=zonal_precipitation,
    ...     lat_bands=[(-60, -40), (-20, 20), (40, 60)],
    ...     color='darkblue',
    ...     ylabel='Precipitation [mm/day]',
    ...     violin_width=8
    ... )
    """
    # Phase 1: Set defaults
    if lat_bands is None:
        lat_bands = [(-90, -30), (-30, 30), (30, 90)]
    
    if color is None:
        color = 'C0'
    
    if vp_color is None:
        vp_color = color
    
    member_color = whiten(color, 0.1)
    
    if ylabel is None:
        if da_zm.name:
            ylabel = str(da_zm.name)
        elif hasattr(da_zm, 'units'):
            ylabel = f"[{da_zm.units}]"
        else:
            ylabel = ""
    
    # Phase 2: Compute latitude band means
    band_means = _compute_latitude_band_means(da_zm, lat_coord, lat_bands)
    band_centers = _get_latitude_band_centers(lat_bands)
    
    # Phase 3: Create figure and plot member lines
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    label = f"Members (n={len(da_zm[member_coord])})"
    _plot_member_lines(ax, da_zm, member_coord, lat_coord, member_color, label=label)
    
    # Phase 4: Add violin plots
    _add_violin_plots(
        ax, band_means, band_centers, vp_color, violin_width
    )
    
    # Phase 5: Configure axes and styling
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    
    return (fig, ax)
