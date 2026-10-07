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

        def set_verticalalignment(self, value):
            self.verticalalignment = value

        def set_rotation_mode(self, value):
            self.rotation_mode = value

    class AxisSide:
        def __init__(self):
            self.label = Label()
            self.labelpad = 0

        def get_label(self):
            return self.label

    class Axis:
        def __init__(self):
            self.xaxis = AxisSide()
            self.yaxis = AxisSide()
            self.tick_rotations = {}
            self.x_tick_labels = [Label(), Label()]
            self.y_tick_labels = [Label(), Label()]

        def tick_params(self, *, axis, labelrotation):
            self.tick_rotations[axis] = labelrotation

        def get_xticklabels(self):
            return self.x_tick_labels

        def get_yticklabels(self):
            return self.y_tick_labels

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
        assert axis.yaxis.label.rotation == PAIRPLOT_FEATURE_LABEL_ANGLE
        assert axis.yaxis.label.horizontalalignment == "right"
        assert axis.yaxis.label.verticalalignment == "bottom"
        assert axis.yaxis.label.rotation_mode == "anchor"
        assert axis.yaxis.labelpad == 12
        assert axis.tick_rotations == {
            "x": PAIRPLOT_FEATURE_LABEL_ANGLE,
            "y": PAIRPLOT_FEATURE_LABEL_ANGLE,
        }
        for label in [*axis.x_tick_labels, *axis.y_tick_labels]:
            assert label.horizontalalignment == "right"
            assert label.rotation_mode == "anchor"
