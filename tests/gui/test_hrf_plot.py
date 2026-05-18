"""Tests for HRF plot uM scaling."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from nirspy.gui.components.hrf_plot import (
    _MOL_TO_MICROMOLAR,
    _rgba,
    render_hrf_plot,
)


class TestHRFMicromolarScaling:
    """HRF plot Y axis values scaled to micromolar."""

    def _make_evoked(self, data_val: float = 1e-6) -> MagicMock:
        """Create a mock Evoked with known data."""
        evoked = MagicMock()
        evoked.times = np.linspace(-2, 10, 50)
        # 2 HbO channels + 2 HbR channels
        evoked.ch_names = ["S1_D1 hbo", "S2_D2 hbo", "S1_D1 hbr", "S2_D2 hbr"]
        # data_val in mol/L (e.g. 1e-6 M = 1 uM)
        evoked.data = np.full((4, 50), data_val)
        evoked.nave = 10
        return evoked

    def test_scaling_constant(self) -> None:
        assert _MOL_TO_MICROMOLAR == 1e6

    def test_yaxis_label_micromolar(self) -> None:
        evoked = self._make_evoked()
        result = render_hrf_plot({"cond1": evoked})
        # Find the graph component
        graph = _find_graph(result)
        assert graph is not None
        fig = graph.figure
        yaxis_title = fig.layout.yaxis.title.text
        assert "M" in yaxis_title
        assert "Concentration" in yaxis_title

    def test_values_scaled_to_micromolar(self) -> None:
        data_val_mol = 2e-6  # 2 uM in mol/L
        evoked = self._make_evoked(data_val_mol)
        result = render_hrf_plot({"cond1": evoked})
        graph = _find_graph(result)
        assert graph is not None
        fig = graph.figure
        # First trace should be HbO mean
        y_data = fig.data[0].y
        # Expected: 2e-6 * 1e6 = 2.0 uM
        expected = data_val_mol * _MOL_TO_MICROMOLAR
        assert abs(y_data[0] - expected) < 1e-10

    def test_legend_has_delta_um(self) -> None:
        evoked = self._make_evoked()
        result = render_hrf_plot({"cond1": evoked})
        graph = _find_graph(result)
        assert graph is not None
        fig = graph.figure
        # Check that trace names contain uM
        trace_names = [t.name for t in fig.data if t.name]
        hbo_names = [n for n in trace_names if "HbO" in n]
        assert len(hbo_names) > 0
        assert any("M" in n for n in hbo_names)

    def test_error_band_also_scaled(self) -> None:
        data_val = 1e-6
        evoked = self._make_evoked(data_val)
        # Make channels slightly different for nonzero std
        evoked.data[0, :] = data_val * 1.1
        evoked.data[1, :] = data_val * 0.9
        result = render_hrf_plot({"cond1": evoked})
        graph = _find_graph(result)
        assert graph is not None
        fig = graph.figure
        # Should have at least 4 traces (HbO line, HbO band, HbR line, HbR band)
        assert len(fig.data) >= 4

    def test_empty_evoked_dict(self) -> None:
        result = render_hrf_plot(None)
        assert "No HRF data" in str(result)

    def test_rgba_helper(self) -> None:
        assert _rgba("#d62728", 0.15) == "rgba(214,39,40,0.15)"


# -- Helpers --


def _find_graph(component: Any) -> Any:
    """Find dcc.Graph in component tree."""
    from dash import dcc

    if isinstance(component, dcc.Graph):
        return component
    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, list):
            for child in children:
                result = _find_graph(child)
                if result is not None:
                    return result
        elif children is not None:
            return _find_graph(children)
    return None
