"""Helper plotting utilities."""

from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors


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
