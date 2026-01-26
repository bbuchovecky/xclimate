"""Plotting utilities for ensemble data."""

from __future__ import annotations
from typing import Tuple, List, Optional, Union

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
    violin_settings: Optional[List[dict]],
) -> Tuple[List[str], List[Optional[Union[int, str]]], List[Optional[dict]]]:
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
    violin_settings : List[dict] or None
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
    if highlight_member is not None:
        if isinstance(highlight_member, list):
            assert len(highlight_member) == n_datasets
            highlight_members = highlight_member
        elif isinstance(highlight_member, (int, str)):
            highlight_members = [highlight_member] * n_datasets
    else:
        highlight_members = [None] * n_datasets

    # Normalize violin_settings
    if violin_settings is None or len(violin_settings) != n_datasets:
        violin_settings_list = [None] * n_datasets
    else:
        violin_settings_list = violin_settings

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


def _compute_timeseries_data(
    da: xr.DataArray,
    weight: xr.DataArray,
    time_mean_period: slice,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute weighted spatial average timeseries and time mean.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with lat, lon, and time dimensions
    weight : xr.DataArray
        Weights for spatial averaging
    time_mean_period : slice
        Time period for computing the mean

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        (timeseries, time_mean) - annual timeseries and period mean
    """
    # Compute weighted spatial average and annual mean
    da_ts = da.weighted(weight).mean(dim=["lat", "lon"]).groupby("time.year").mean()

    # Compute time mean over specified period
    da_tm = (
        da.sel(time=time_mean_period)
        .weighted(weight)
        .mean(dim=["lat", "lon"])
        .groupby("time.year")
        .mean()
        .mean(dim="year")
    )

    return da_ts, da_tm


def _plot_ensemble_members(
    ax,
    da_ts: xr.DataArray,
    member_coord: str,
    label: str,
    color: str,
) -> None:
    """
    Plot all ensemble members as transparent lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    da_ts : xr.DataArray
        Timeseries data with member dimension
    member_coord : str
        Name of member coordinate
    label : str
        Label for legend
    color : str
        Color for the lines
    """
    for i, m in enumerate(da_ts[member_coord]):
        lab = f"{label} (n={len(da_ts[member_coord])})" if i == 0 else None
        da_ts.sel({member_coord: m}).plot(
            ax=ax,
            color=color,
            ls="-",
            alpha=MEMBER_LINE_ALPHA,
            lw=MEMBER_LINE_WIDTH,
            label=lab,
            _labels=False,
        )


def _plot_highlighted_member(
    ax,
    da_ts: xr.DataArray,
    da_tm: xr.DataArray,
    member_coord: str,
    highlight: Union[int, str],
    label: str,
    color: str,
    vp_xpos: float,
    vp_marker: str,
    vp_facecolor: str,
    vp_edgecolor: str,
) -> None:
    """
    Plot a highlighted ensemble member with bold line and marker.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    da_ts : xr.DataArray
        Timeseries data
    da_tm : xr.DataArray
        Time mean data
    member_coord : str
        Name of member coordinate
    highlight : int or str
        Member to highlight
    label : str
        Base label for legend
    color : str
        Color for the line
    vp_xpos : float
        Violin plot x-position
    vp_marker : str
        Marker style
    vp_facecolor : str
        Marker face color
    vp_edgecolor : str
        Marker edge color
    """
    da_ts.sel({member_coord: highlight}).plot(
        ax=ax,
        color=color,
        ls="-",
        alpha=1,
        lw=HIGHLIGHT_LINE_WIDTH,
        label=f"{label} {highlight}",
        _labels=False,
    )

    ax.scatter(
        vp_xpos + HIGHLIGHT_MARKER_OFFSET,
        da_tm.sel({member_coord: highlight}),
        s=20,
        marker=vp_marker,
        facecolor=vp_facecolor,
        edgecolor=vp_edgecolor,
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


def plot_ensemble_timeseries(
    das: List[xr.DataArray],
    das_weights: List[xr.DataArray],
    das_labels: List[str],
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

    This function creates a visualization of multiple ensemble timeseries with individual
    ensemble members shown as transparent lines, optional highlighted members, and violin
    plots showing the distribution of time-mean values for a specified period.

    Parameters
    ----------
    das : List[xr.DataArray]
        List of xarray DataArrays containing the timeseries data to plot. Each DataArray
        should have dimensions including time, lat, lon, and a member dimension.
    das_weights : List[xr.DataArray]
        List of weight arrays corresponding to each DataArray in `das`, used for weighted
        spatial averaging.
    das_labels : List[str]
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
    # Phase 1: Validate inputs and set defaults
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS.keys())

    assert len(das) == len(das_weights) and len(das) == len(das_labels)
    assert len(colors) >= len(das)

    n = len(das)
    member_colors = [whiten(c, 0.1) for c in colors]

    # Phase 2: Normalize input parameters
    member_coords, highlight_members, violin_settings_list = _normalize_ensemble_inputs(
        n, member_coord, highlight_member, violin_settings
    )

    # Phase 3: Create figure and add time period shading
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.axvspan(
        int(time_mean_period.start[:4]),
        int(time_mean_period.stop[:4]),
        alpha=0.25,
        facecolor="silver",
        edgecolor=None,
    )

    # Phase 4: Plot each dataset
    for da, weight, label, member, highlight, color, mem_color, vs in zip(
        das,
        das_weights,
        das_labels,
        member_coords,
        highlight_members,
        colors,
        member_colors,
        violin_settings_list,
    ):
        # Parse violin settings
        vp_xpos, vp_marker, vp_facecolor, vp_edgecolor = _parse_violin_settings(
            vs, color
        )

        # Compute timeseries and time mean
        da_ts, da_tm = _compute_timeseries_data(da, weight, time_mean_period)

        # Plot all ensemble members
        _plot_ensemble_members(ax, da_ts, member, label, mem_color)

        # Plot highlighted member if specified
        if highlight is not None and isinstance(highlight, (int, str)):
            _plot_highlighted_member(
                ax,
                da_ts,
                da_tm,
                member,
                highlight,
                label,
                color,
                vp_xpos,
                vp_marker,
                vp_facecolor,
                vp_edgecolor,
            )

        # Add violin plot
        _create_violin_plot(ax, da_tm, vp_xpos, vp_facecolor, vp_edgecolor)

    # Phase 5: Configure axes and layout
    _configure_axes(ax, ylabel, xlabel, title, xlim, ylim)
    plt.tight_layout()

    return (fig, ax)
