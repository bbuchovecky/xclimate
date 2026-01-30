"""Plotting utilities for ensemble data."""

from __future__ import annotations
from typing import Tuple, List, Optional, Union, Sequence

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .helpers import whiten

# Constants
DEFAULT_VIOLIN_X = 2020
DEFAULT_VIOLIN_WIDTH = 15
HIGHLIGHT_MARKER_OFFSET = 3
MEMBER_LINE_ALPHA = 0.5
MEMBER_LINE_WIDTH = 1
HIGHLIGHT_LINE_WIDTH = 2
VIOLIN_ALPHA = 0.3
VIOLIN_EDGE_WIDTH = 1
VIOLIN_EDGE_HALF_WIDTH = 1


def _normalize_ensemble_inputs(
    n_datasets: int,
    member_coord: str | List[str],
    highlight_member: Optional[Union[int, str, List]],
    violin_settings: Optional[Union[dict, List[dict]]],
) -> Tuple[List[str], List[Optional[Union[int, str]]], Sequence[Optional[dict]]]:
    """
    Normalize ensemble input parameters to consistent list format.

    Parameters
    ----------
    n_datasets : int
        Number of datasets being plotted
    member_coord : str or List[str]
        Member coordinate name(s)
    highlight_member : int, str, List, or None
        Member(s) to highlight
    violin_settings : dict or List[dict] or None
        Violin plot settings for each dataset

    Returns
    -------
    Tuple[List[str], List[Optional[Union[int, str]]], List[Optional[dict]]]
        Normalized (member_coords, highlight_members, violin_settings_list)
    """
    # Normalize member_coord
    if isinstance(member_coord, str):
        member_coords = [member_coord] * n_datasets
    else:
        assert isinstance(member_coord, list) and len(member_coord) == n_datasets
        member_coords = member_coord

    # Normalize highlight_member
    highlight_members: List[Optional[Union[int, str]]]
    if highlight_member is not None:
        if isinstance(highlight_member, list):
            assert len(highlight_member) == n_datasets
            highlight_members = highlight_member
        else:
            highlight_members = [None] * n_datasets
    else:
        highlight_members = [None] * n_datasets

    # Normalize violin_settings
    violin_settings_list: Sequence[Optional[dict]]
    if violin_settings is None:
        violin_settings_list = [None] * n_datasets
    elif isinstance(violin_settings, dict) and len(violin_settings) < n_datasets:
        violin_settings_list = [violin_settings] * n_datasets
    else:
        violin_settings_list = list(violin_settings)

    return member_coords, highlight_members, violin_settings_list


def _parse_violin_settings(
    vs: Optional[dict],
    default_color: str,
    default_x: float = DEFAULT_VIOLIN_X,
) -> Tuple[float, str, str, str]:
    """
    Extract violin plot settings with defaults.

    Parameters
    ----------
    vs : dict or None
        Violin settings dictionary
    default_color : str
        Default color to use if not specified
    default_x : float, optional
        Default x-position for violin plot

    Returns
    -------
    Tuple[float, str, str, str]
        (x_position, marker, facecolor, edgecolor)
    """
    if vs is None:
        return default_x, "o", default_color, default_color

    return (
        vs.get("x", default_x),
        vs.get("marker", "o"),
        vs.get("facecolor", default_color),
        vs.get("edgecolor", default_color),
    )


def _create_violin_plot(
    ax,
    da_tm: xr.DataArray,
    vp_xpos: float,
    vp_facecolor: str,
    vp_edgecolor: str,
):
    """
    Create and style a violin plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    da_tm : xr.DataArray
        Time mean data for violin plot
    vp_xpos : float
        X-position for violin plot
    vp_facecolor : str
        Face color for violin
    vp_edgecolor : str
        Edge color for violin

    Returns
    -------
    dict
        Violin plot dictionary from violinplot()
    """
    vp = ax.violinplot(
        da_tm,
        [vp_xpos],
        vert=True,
        widths=DEFAULT_VIOLIN_WIDTH,
        side="high",
        showmeans=False,
        showextrema=True,
        showmedians=True,
    )

    _style_violin_plot(vp, vp_xpos, vp_facecolor, vp_edgecolor)

    return vp


def _style_violin_plot(
    vp: dict,
    vp_xpos: float,
    vp_facecolor: str,
    vp_edgecolor: str,
) -> None:
    """
    Apply custom styling to violin plot components.

    Parameters
    ----------
    vp : dict
        Violin plot dictionary from violinplot()
    vp_xpos : float
        X-position of violin plot
    vp_facecolor : str
        Face color for violin body
    vp_edgecolor : str
        Edge color for violin lines
    """
    # Style body
    vp["bodies"][0].set(facecolor=vp_facecolor, alpha=VIOLIN_ALPHA)

    # Style bars and lines
    vp["cbars"].set(linewidth=0)
    vp["cmedians"].set(linewidth=VIOLIN_EDGE_WIDTH, color=vp_edgecolor)
    vp["cmins"].set(linewidth=VIOLIN_EDGE_WIDTH, color=vp_edgecolor)
    vp["cmaxes"].set(linewidth=VIOLIN_EDGE_WIDTH, color=vp_edgecolor)

    # Adjust segment positions for median, min, max
    segmed = vp["cmedians"].get_segments().copy()
    segmin = vp["cmins"].get_segments().copy()
    segmax = vp["cmaxes"].get_segments().copy()

    for smed, smin, smax in zip(segmed, segmin, segmax):
        for s in [smed, smin, smax]:
            s[0][0] = vp_xpos - VIOLIN_EDGE_HALF_WIDTH
            s[1][0] = vp_xpos + VIOLIN_EDGE_HALF_WIDTH

    vp["cmedians"].set_segments(segmed)
    vp["cmins"].set_segments(segmin)
    vp["cmaxes"].set_segments(segmax)


def _configure_axes(
    ax,
    ylabel: str,
    xlabel: str,
    title: str,
    xlim: Tuple,
    ylim: Tuple,
):
    """
    Configure axis labels, limits, and title.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Primary axis
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    title : str
        Plot title
    xlim : Tuple
        X-axis limits
    ylim : Tuple
        Y-axis limits

    Returns
    -------
    matplotlib.axes.Axes
        Twin axis
    """
    # Create twin axis for visual balance
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_ylabel(ylabel, labelpad=24)

    # Configure primary axis
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="y", labelright=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim[0], xlim[1])

    # Set y-limits if specified
    if ylim[0] is not None and ylim[1] is not None:
        ax.set_ylim(ylim[0], ylim[1])
        ax2.set_ylim(ylim[0], ylim[1])

    return ax2


def plot_ensemble_line(
    das: xr.DataArray | List[xr.DataArray],
    das_labels: str | List[str],
    ylabel: str,
    plot_dim: str,
    das_violin: Optional[xr.DataArray | List[xr.DataArray]] = None,
    xlabel: str = "",
    title: str = "",
    member_coord: str | List[str] = "member",
    colors: Optional[List[str]] = None,
    ylim: Tuple = (None, None),
    xlim: Tuple = (None, None),
    highlight_member: Optional[Union[int, str, List]] = None,
    violin_settings: Optional[Union[dict, List[dict]]] = None,
    violin_xrange: Optional[Tuple[float, float]] = None,
    add_legend: bool = True,
) -> Tuple:
    """
    Plot ensemble line plots with optional violin plots.

    This is a pure plotting utility that creates line plots for ensemble data.
    All data processing (averaging, grouping, selection) should be done before
    calling this function.

    Parameters
    ----------
    das : xr.DataArray or List[xr.DataArray]
        Single or list of pre-computed xarray DataArrays to plot. Each DataArray
        should have the plot dimension and a member dimension.
    das_labels : str or List[str]
        Labels for each dataset to be shown in the legend.
    ylabel : str
        Label for the y-axis.
    plot_dim : str
        Dimension to plot along the x-axis (e.g., 'year', 'lat').
    das_violin : Optional[xr.DataArray | List[xr.DataArray]], optional
        Single or list of pre-computed data for violin plots. Should have only
        the member dimension. If None, no violin plots are created. Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is empty string.
    title : str, optional
        Title for the plot. Default is an empty string.
    member_coord : str or List[str], optional
        Name(s) of the ensemble member coordinate dimension. Can be a single string applied
        to all datasets or a list of strings for each dataset. Default is "member".
    colors : Optional[List[str]], optional
        List of colors to use for each dataset. Must have at least as many colors as datasets.
        If None, uses matplotlib's TABLEAU_COLORS. Default is None.
    ylim : Tuple, optional
        Y-axis limits as (min, max). Default is (None, None).
    xlim : Tuple, optional
        X-axis limits as (min, max). Default is (None, None).
    highlight_member : Optional[Union[int, str, List]], optional
        Ensemble member(s) to highlight with a bold line. Can be a single value applied to
        all datasets or a list of values for each dataset. Default is None.
    violin_settings : Optional[Union[dict, List[dict]]], optional
        List of dictionaries containing violin plot settings for each dataset. If one dictionary
        is passed, the settings are applied to each violin plot. Each dictionary can contain 'x', 
        'marker', 'facecolor', and 'edgecolor' keys. Default is None.
    violin_xrange : Optional[Tuple[float, float]], optional
        X-axis range to shade for violin plot period (e.g., (1995, 2014) for time period).
        If None, no shading is added. Default is None.
    add_legend : bool = True
        Option to add legend in the upper right corner of the plot. Defaults to True.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing the figure and axes objects.

    Notes
    -----
    - This is a pure plotting function - all data processing should be done beforehand.
    - Violin plots show the distribution of ensemble members from das_violin data.

    Examples
    --------
    Plot ensemble timeseries with violin plots:
    >>> # Compute data first
    >>> da_ts = da.weighted(weights).mean(dim=['lat', 'lon']).groupby('time.year').mean()
    >>> da_violin = da.sel(time=slice('1995', '2014')).weighted(weights).mean(
    ...     dim=['lat', 'lon']).groupby('time.year').mean().mean(dim='year')
    >>> fig, ax = plot_ensemble_line(
    ...     das=da_ts,
    ...     das_labels='Model A',
    ...     ylabel='Temperature [°C]',
    ...     plot_dim='year',
    ...     das_violin=da_violin,
    ...     xlabel='Year',
    ...     violin_xrange=(1995, 2014)
    ... )

    Plot ensemble zonal mean without violin plots:
    >>> # Compute zonal mean first
    >>> da_zonal = da.weighted(weights).mean(dim=['lon', 'time'])
    >>> fig, ax = plot_ensemble_line(
    ...     das=da_zonal,
    ...     das_labels='Model A',
    ...     ylabel='Temperature [°C]',
    ...     plot_dim='lat',
    ...     xlabel='Latitude'
    ... )
    """
    # Phase 1: Validate inputs and set defaults
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS.keys())
    
    # Convert single to list
    if isinstance(das, xr.DataArray):
        das = [das]
    if isinstance(das_labels, str):
        das_labels = [das_labels]
    if das_violin is not None and isinstance(das_violin, xr.DataArray):
        das_violin_list = [das_violin]
    elif das_violin is None:
        das_violin_list = [None] * len(das)
    else:
        das_violin_list = das_violin

    assert len(das) == len(das_labels)
    assert len(colors) >= len(das)
    assert len(das_violin_list) == len(das)

    n = len(das)
    member_colors = [whiten(c, 0.1) for c in colors]

    # Phase 2: Normalize input parameters
    member_coords, highlight_members, violin_settings_list = _normalize_ensemble_inputs(
        n, member_coord, highlight_member, violin_settings
    )

    # Phase 3: Create figure and optionally add shaded region
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Add shaded region if violin_xrange is provided
    if violin_xrange is not None:
        ax.axvspan(
            violin_xrange[0],
            violin_xrange[1],
            alpha=0.25,
            facecolor="silver",
            edgecolor=None,
        )

    # Phase 4: Plot each dataset
    vp_xpos = None  # Initialize to track last violin x position
    for da, da_violin, label, member, highlight, color, mem_color, vs in zip(
        das,
        das_violin_list,
        das_labels,
        member_coords,
        highlight_members,
        colors,
        member_colors,
        violin_settings_list,
    ):
        # Parse violin settings
        default_vp_x = da[plot_dim].values[-1] if da_violin is not None else None
        vp_xpos, vp_marker, vp_facecolor, vp_edgecolor = _parse_violin_settings(
            vs, color, default_x=default_vp_x if default_vp_x is not None else DEFAULT_VIOLIN_X
        )

        # Plot all ensemble members
        for i, m in enumerate(da[member]):
            lab = f"{label} (n={len(da[member])})" if i == 0 else None
            da.sel({member: m}).plot(
                ax=ax,
                color=mem_color,
                ls="-",
                alpha=MEMBER_LINE_ALPHA,
                lw=MEMBER_LINE_WIDTH,
                label=lab,
                _labels=False,
            )

        # Plot highlighted member if specified
        if highlight is not None and isinstance(highlight, (int, str)):
            da.sel({member: highlight}).plot(
                ax=ax,
                color=color,
                ls="-",
                alpha=1,
                lw=HIGHLIGHT_LINE_WIDTH,
                label=f"{label} {highlight}",
                _labels=False,
            )

            # Add marker for highlighted member in violin plot
            if da_violin is not None:
                ax.scatter(
                    vp_xpos + HIGHLIGHT_MARKER_OFFSET,
                    da_violin.sel({member: highlight}),
                    s=20,
                    marker=vp_marker,
                    facecolor=vp_facecolor,
                    edgecolor=vp_edgecolor,
                )

        # Add violin plot if data available
        if da_violin is not None:
            _create_violin_plot(ax, da_violin, vp_xpos, vp_facecolor, vp_edgecolor)

    # Phase 5: Configure axes and layout
    # Add legend
    if add_legend:
        ax.legend(loc="upper right", ncols=2)

    # Auto-set xlim if not specified
    final_xlim = xlim
    da_xmin = das[0][plot_dim].values.min()
    da_xmax = das[0][plot_dim].values.max()
    da_xspan = da_xmax - da_xmin
    if xlim == (None, None):
        if das_violin is not None and vp_xpos is not None:
            final_xlim = (da_xmin - da_xspan / 40, vp_xpos + DEFAULT_VIOLIN_WIDTH * 0.75)
        else:
            final_xlim = (da_xmin - da_xspan / 40, da_xmax + da_xspan / 40)
    elif xlim[0] is None:
        final_xlim = (da_xmin - da_xspan / 40, xlim[1])
    elif xlim[1] is None:
        final_xlim = (xlim[0], da_xmax + da_xspan / 40)
    
    # Auto-set ylim if not specified and adding legend
    final_ylim = ylim
    if add_legend:
        ymin, ymax = ax.get_ylim()
        if ylim == (None, None) or ylim[1] is None:
            final_ylim = (ymin, ymax + (ymax - ymin) / 15)

    _configure_axes(ax, ylabel, xlabel, title, final_xlim, final_ylim)
    plt.tight_layout()

    return (fig, ax)


def plot_ensemble_zonal(
    das: xr.DataArray | List[xr.DataArray],
    das_labels: str | List[str],
    ylabel: str,
    das_weights: Optional[xr.DataArray | List[xr.DataArray]] = None,
    xlabel: str = "Latitude",
    title: str = "",
    member_coord: str | List[str] = "member",
    colors: Optional[List[str]] = None,
    ylim: Tuple = (None, None),
    xlim: Tuple = (-90, 90),
    highlight_member: Optional[Union[int, str, List]] = None,
) -> Tuple:
    """
    Plot ensemble zonal mean profiles without violin plots.

    This is a convenience wrapper around plot_ensemble_line for zonal mean plots.
    It creates a visualization of zonal mean (latitude) profiles for multiple
    ensemble datasets with individual ensemble members shown as transparent lines and
    optional highlighted members shown with bold lines.

    Parameters
    ----------
    das : xr.DataArray or List[xr.DataArray]
        Single or list of xarray DataArrays containing the data to plot. Each DataArray should have
        dimensions including lat, lon, and a member dimension.
    das_labels : str or List[str]
        Labels for each dataset to be shown in the legend.
    ylabel : str
        Label for the y-axis.
    das_weights : Optional[xr.DataArray | List[xr.DataArray]], optional
        Weight array(s) corresponding to each DataArray in `das`, used for weighted
        averaging over longitude. If None, uniform weights of ones are used. Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is "Latitude".
    title : str, optional
        Title for the plot. Default is an empty string.
    member_coord : str or List[str], optional
        Name(s) of the ensemble member coordinate dimension. Can be a single string applied
        to all datasets or a list of strings for each dataset. Default is "member".
    colors : Optional[List[str]], optional
        List of colors to use for each dataset. Must have at least as many colors as datasets.
        If None, uses matplotlib's TABLEAU_COLORS. Default is None.
    ylim : Tuple, optional
        Y-axis limits as (min, max). Default is (None, None).
    xlim : Tuple, optional
        X-axis limits as (min, max). Default is (-90, 90).
    highlight_member : Optional[Union[int, str, List]], optional
        Ensemble member(s) to highlight with a bold line. Can be a single value applied to
        all datasets or a list of values for each dataset. Default is None.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing the figure and axes objects.

    Notes
    -----
    - The function performs weighted averaging over the longitude dimension.
    - If the data has a time dimension, it will be averaged over time as well.

    Examples
    --------
    >>> fig, ax = plot_ensemble_zonal(
    ...     das=[da1, da2],
    ...     das_labels=['Model A', 'Model B'],
    ...     ylabel='Temperature [°C]',
    ...     title='Ensemble Zonal Mean Temperature'
    ... )
    """
    # Convert single to list
    if isinstance(das, xr.DataArray):
        das_list = [das]
    else:
        das_list = das
    
    if isinstance(das_weights, xr.DataArray):
        das_weights_list = [das_weights]
    elif das_weights is None:
        das_weights_list = [xr.ones_like(da.isel({da.dims[0]: 0}, drop=True)) for da in das_list]
    else:
        das_weights_list = das_weights
    
    # Compute zonal means for each dataset
    das_zonal = []
    for da, weight in zip(das_list, das_weights_list):
        # Determine averaging dimensions
        avg_dims = ['lon']
        if 'time' in da.dims:
            avg_dims.append('time')
        
        # Compute weighted zonal mean
        da_zonal = da.weighted(weight).mean(dim=avg_dims)
        das_zonal.append(da_zonal)
    
    return plot_ensemble_line(
        das=das_zonal,
        das_labels=das_labels,
        ylabel=ylabel,
        plot_dim='lat',
        das_violin=None,
        xlabel=xlabel,
        title=title,
        member_coord=member_coord,
        colors=colors,
        ylim=ylim,
        xlim=xlim,
        highlight_member=highlight_member,
        violin_settings=None,
        violin_xrange=None,
    )


def plot_ensemble_timeseries(
    das: xr.DataArray | List[xr.DataArray],
    das_weights: xr.DataArray | List[xr.DataArray],
    das_labels: str | List[str],
    ylabel: str,
    xlabel: str = "Year",
    title: str = "",
    member_coord: str | List[str] = "member",
    colors: Optional[List[str]] = None,
    ylim: Tuple = (None, None),
    xlim: Tuple = (1948, 2035),
    highlight_member: Optional[Union[int, str, List]] = None,
    time_mean_period: slice = slice("1995-01", "2014-12"),
    violin_settings: Optional[List[dict]] = None,
) -> Tuple:
    """
    Plot ensemble timeseries with violin plots showing distribution of time-mean values.

    This is a convenience wrapper around plot_ensemble_line for timeseries plots.
    It creates a visualization of multiple ensemble timeseries with individual
    ensemble members shown as transparent lines, optional highlighted members, and violin
    plots showing the distribution of time-mean values for a specified period.

    Parameters
    ----------
    das : xr.DataArray or List[xr.DataArray]
        Single or list of xarray DataArrays containing the timeseries data to plot. Each DataArray
        should have dimensions including time, lat, lon, and a member dimension.
    das_weights : xr.DataArray or List[xr.DataArray]
        Weight array(s) corresponding to each DataArray in `das`, used for weighted
        spatial averaging.
    das_labels : str or List[str]
        Labels for each dataset to be shown in the legend.
    ylabel : str
        Label for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is "Year".
    title : str, optional
        Title for the plot. Default is an empty string.
    member_coord : str or List[str], optional
        Name(s) of the ensemble member coordinate dimension. Can be a single string applied
        to all datasets or a list of strings for each dataset. Default is "member".
    colors : Optional[List[str]], optional
        List of colors to use for each dataset. Must have at least as many colors as datasets.
        If None, uses matplotlib's TABLEAU_COLORS. Default is None.
    ylim : Tuple, optional
        Y-axis limits as (min, max). Default is (None, None).
    xlim : Tuple, optional
        X-axis limits as (min, max). Default is (1948, 2035).
    highlight_member : Optional[Union[int, str, List]], optional
        Ensemble member(s) to highlight with a bold line. Can be a single value applied to
        all datasets or a list of values for each dataset. Default is None.
    time_mean_period : slice, optional
        Time period over which to compute the mean for the violin plots. Default is
        slice("1995-01", "2014-12").
    violin_settings : Optional[List[dict]], optional
        List of dictionaries containing violin plot settings for each dataset. Each dictionary
        can contain 'x', 'marker', 'facecolor', and 'edgecolor' keys. Default is None.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing the figure and axes objects.

    Notes
    -----
    - The function performs weighted spatial averaging over lat/lon dimensions and annual
      averaging over the time dimension.
    - A shaded region indicates the time period used for computing the violin plot distributions.
    - Violin plots are positioned at x=2020 by default and show the distribution of ensemble
      members' time-mean values.
    - The function creates a twin y-axis for visual balance.

    Examples
    --------
    >>> fig, ax = plot_ensemble_timeseries(
    ...     das=[da1, da2],
    ...     das_weights=[weights1, weights2],
    ...     das_labels=['Model A', 'Model B'],
    ...     ylabel='Temperature [°C]',
    ...     title='Ensemble Temperature Timeseries'
    ... )
    """
    # Convert single to list
    if isinstance(das, xr.DataArray):
        das_list = [das]
    else:
        das_list = das
    
    if isinstance(das_weights, xr.DataArray):
        das_weights_list = [das_weights]
    else:
        das_weights_list = das_weights
    
    # Compute timeseries and violin data for each dataset
    das_ts = []
    das_violin = []
    for da, weight in zip(das_list, das_weights_list):
        # Compute weighted spatial average and annual mean
        da_ts = da.weighted(weight).mean(dim=['lat', 'lon']).groupby('time.year').mean()
        
        # Compute time mean over specified period for violin plot
        da_violin = (
            da.sel(time=time_mean_period)
            .weighted(weight)
            .mean(dim=['lat', 'lon'])
            .groupby('time.year')
            .mean()
            .mean(dim='year')
        )
        
        das_ts.append(da_ts)
        das_violin.append(da_violin)
    
    # Extract year range for shading
    violin_xrange = (
        int(time_mean_period.start[:4]),
        int(time_mean_period.stop[:4])
    )
    
    return plot_ensemble_line(
        das=das_ts,
        das_labels=das_labels,
        ylabel=ylabel,
        plot_dim='year',
        das_violin=das_violin,
        xlabel=xlabel,
        title=title,
        member_coord=member_coord,
        colors=colors,
        ylim=ylim,
        xlim=xlim,
        highlight_member=highlight_member,
        violin_settings=violin_settings,
        violin_xrange=violin_xrange,
    )
