"""Helper plotting utilities."""

from __future__ import annotations
from typing import Any

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def whiten(color, alpha):
    """
    Linearly mix a color with white.

    Parameters
    ----------
    color : str or tuple
        Any matplotlib color spec.
    alpha : float
        Whitening fraction in [0, 1].

    Returns
    -------
    tuple
        RGB tuple in [0, 1].
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - alpha) * rgb + alpha)


def center_axis_at_zero(ax, axis="y"):
    """
    Center the specified axis around zero by making the limits symmetric.

    Parameters:
    ax : matplotlib axis object
        The axis to modify
    axis : str
        Which axis to center ('x', 'y', or 'both')
    """
    if axis == "y":
        ymin, ymax = ax.get_ylim()
        max_abs = max(abs(ymin), abs(ymax))
        ax.set_ylim(-max_abs, max_abs)

    elif axis == "x":
        xmin, xmax = ax.get_xlim()
        max_abs = max(abs(xmin), abs(xmax))
        ax.set_xlim(-max_abs, max_abs)

    elif axis == "both":
        ymin, ymax = ax.get_ylim()
        max_abs = max(abs(ymin), abs(ymax))
        ax.set_ylim(-max_abs, max_abs)

        xmin, xmax = ax.get_xlim()
        max_abs = max(abs(xmin), abs(xmax))
        ax.set_xlim(-max_abs, max_abs)

    else:
        raise ValueError("axis must be 'x', 'y', or 'both'")


def add_orphan_xlabel(
    fig: plt.Figure,
    ax: np.ndarray | list,
    nplots: int,
    ncols: int,
    nrows: int,
    xlabel: str,
    ypos_adjust: float = -0.025,
    **kwargs: Any,
) -> None:
    """
    Add xlabel to orphan subplots in the second-to-last row.
    
    Orphan subplots are those in the second-to-last row that have no subplot
    directly below them. This function adds x-axis labels to these plots using
    fig.text to avoid affecting the layout.
    
    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure containing the subplots.
    ax : np.ndarray or list
        Array or list of subplot axes.
    nplots : int
        Total number of subplots.
    ncols : int
        Number of columns in the subplot grid.
    nrows : int
        Number of rows in the subplot grid.
    xlabel : str
        The x-axis label text to add.
    ypos_adjust : float, default -0.025
        Vertical offset for the label in figure coordinates.
        Negative values place the label below the subplot.
    **kwargs : Any
        Additional keyword arguments passed to fig.text.
        Common options include fontsize, fontweight, etc.
    
    Examples
    --------
    >>> fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    >>> add_orphan_xlabel(fig, ax.flat, 5, 3, 2, "Time (years)")
    """
    # Force constrained layout to finalize positions
    fig.canvas.draw()

    # Add xlabels to second-to-last row where no subplot exists below
    for i in range(nplots):
        if i // ncols == nrows - 2 and i + ncols >= nplots:
            pos = ax[i].get_position()  # get position after layout is finalized
            fig.text(
                (pos.x0 + pos.x1) / 2,  # centered horizontally
                pos.y0 + ypos_adjust,   # slightly below the subplot
                xlabel,
                ha="center",
                va="top",
                **kwargs
            )


def add_orphan_xticklabels(
    fig: plt.Figure,
    ax: np.ndarray | list,
    nplots: int,
    ncols: int,
    nrows: int,
    xticklabels: list[str],
    use_fig_text: bool = False,
    ypos_adjust: float = -0.01,
    **kwargs: Any,
) -> None:
    """
    Add xticklabels to orphan subplots in the second-to-last row.
    
    Orphan subplots are those in the second-to-last row that have no subplot
    directly below them. This function provides two methods for adding tick labels:
    using fig.text (preserves layout) or ax.set_xticklabels (standard approach).
    
    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure containing the subplots.
    ax : np.ndarray or list
        Array or list of subplot axes.
    nplots : int
        Total number of subplots.
    ncols : int
        Number of columns in the subplot grid.
    nrows : int
        Number of rows in the subplot grid.
    xticklabels : list of str
        List of tick label strings to apply.
    use_fig_text : bool, default False
        If True, use fig.text to place labels (doesn't affect subplot spacing).
        If False, use ax.set_xticklabels (simpler but may trigger layout recalculation).
    ypos_adjust : float, default -0.01
        Vertical offset for labels when use_fig_text=True (negative = below subplot).
    **kwargs : Any
        Additional keyword arguments passed to fig.text or ax.set_xticklabels.
        Common options include fontsize, rotation, ha (horizontal alignment), etc.
    
    Examples
    --------
    >>> fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    >>> labels = ["Jan", "Feb", "Mar", "Apr"]
    >>> add_orphan_xticklabels(fig, ax.flat, 5, 3, 2, labels, use_fig_text=True)
    """
    if use_fig_text:
        # Force constrained layout to finalize positions
        fig.canvas.draw()
        
        # Add xticklabels using fig.text to avoid affecting layout
        for i in range(nplots):
            if i // ncols == nrows - 2 and i + ncols >= nplots:
                pos = ax[i].get_position()
                xticks = ax[i].get_xticks()
                xlim = ax[i].get_xlim()
                
                # Convert data coordinates to figure coordinates
                for tick_val, label_text in zip(xticks, xticklabels):
                    # Normalize tick position to [0, 1] within axis limits
                    tick_norm = (tick_val - xlim[0]) / (xlim[1] - xlim[0])
                    # Convert to figure coordinates
                    x_fig = pos.x0 + tick_norm * (pos.x1 - pos.x0)
                    
                    fig.text(
                        x_fig,
                        pos.y0 + ypos_adjust,
                        label_text,
                        va="top",
                        **kwargs
                    )
    else:
        # Use standard matplotlib method
        for i in range(nplots):
            if i // ncols == nrows - 2 and i + ncols >= nplots:
                ax[i].set_xticklabels(xticklabels, **kwargs)
