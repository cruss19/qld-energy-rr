"""Shared publication styling for RR plots."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib import cm


RR_COLOR_COUNT = 10
PAIRPLOT_FEATURE_LABEL_ANGLE = 45


def plasma_colors(count: int = RR_COLOR_COUNT) -> tuple:
    """Return evenly spaced colours from Matplotlib's plasma colour map."""
    if count < 1:
        raise ValueError("The RR colour cycle requires at least one colour")
    return tuple(cm.plasma(np.linspace(0.0, 1.0, count)))


def apply_rr_plot_style(count: int = RR_COLOR_COUNT) -> tuple:
    """Set and return the repository-wide Matplotlib plasma colour cycle."""
    colors = plasma_colors(count)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)
    return colors


def plasma_tints(
    count: int = RR_COLOR_COUNT,
    white_fraction: float = 0.72,
) -> tuple[str, ...]:
    """Return light plasma-derived fills suitable for text-labelled boxes."""
    if not 0.0 <= white_fraction <= 1.0:
        raise ValueError("white_fraction must lie in [0, 1]")
    rgb = np.asarray(plasma_colors(count))[:, :3]
    tinted = white_fraction + (1.0 - white_fraction) * rgb
    return tuple(mpl.colors.to_hex(color) for color in tinted)


def rotate_pairplot_feature_labels(
    grid,
    angle: float = PAIRPLOT_FEATURE_LABEL_ANGLE,
) -> None:
    """Rotate PairGrid x-axis feature names and tick labels diagonally."""
    for axis in grid.axes.flat:
        if axis is None:
            continue
        label = axis.xaxis.get_label()
        label.set_rotation(angle)
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
        axis.xaxis.labelpad = 10
        axis.tick_params(axis="x", labelrotation=angle)
