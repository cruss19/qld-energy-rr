from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib import cm

matplotlib.use("Agg")

from src.rr_plot_style import (
    PAIRPLOT_FEATURE_LABEL_ANGLE,
    RR_COLOR_COUNT,
    apply_rr_plot_style,
    rotate_pairplot_feature_labels,
)


def test_rr_plot_style_uses_ten_evenly_spaced_plasma_colors() -> None:
    colors = apply_rr_plot_style()
    expected = cm.plasma(np.linspace(0.0, 1.0, RR_COLOR_COUNT))
    actual_cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]

    assert len(colors) == RR_COLOR_COUNT
    np.testing.assert_allclose(np.asarray(colors), expected)
    np.testing.assert_allclose(np.asarray(actual_cycle), expected)


def test_pairplot_feature_names_and_ticks_are_diagonal() -> None:
    class Label:
        def set_rotation(self, value):
            self.rotation = value

        def set_horizontalalignment(self, value):
            self.horizontalalignment = value

        def set_rotation_mode(self, value):
            self.rotation_mode = value

    class XAxis:
        def __init__(self):
            self.label = Label()
            self.labelpad = 0

        def get_label(self):
            return self.label

    class Axis:
        def __init__(self):
            self.xaxis = XAxis()
            self.tick_rotation = None

        def tick_params(self, *, axis, labelrotation):
            assert axis == "x"
            self.tick_rotation = labelrotation

    axes = np.asarray([[Axis(), Axis()]])

    class Grid:
        pass

    grid = Grid()
    grid.axes = axes
    rotate_pairplot_feature_labels(grid)

    for axis in axes.flat:
        assert axis.xaxis.label.rotation == PAIRPLOT_FEATURE_LABEL_ANGLE
        assert axis.xaxis.label.horizontalalignment == "right"
        assert axis.xaxis.label.rotation_mode == "anchor"
        assert axis.xaxis.labelpad == 10
        assert axis.tick_rotation == PAIRPLOT_FEATURE_LABEL_ANGLE
