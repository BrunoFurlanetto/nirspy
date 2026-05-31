"""Smoke tests for GLM visualization component (T-036)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest
from dash import html

from nirspy.domain.glm_result import GLMResult
from nirspy.gui.components.glm_topo import render_glm_summary, render_glm_topo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_glm_result() -> GLMResult:
    """Create a minimal GLMResult for testing."""
    rng = np.random.default_rng(42)
    n_regressors = 3
    n_channels = 8
    n_timepoints = 100

    return GLMResult(
        theta=rng.standard_normal((n_regressors, n_channels)),
        t_stats=rng.standard_normal((n_regressors, n_channels)) * 2,
        p_values=rng.uniform(0, 1, (n_regressors, n_channels)),
        mse=rng.uniform(0.01, 0.1, (n_channels,)),
        channel_names=[f"S{i}D{i} hbo" for i in range(1, n_channels + 1)],
        regressor_names=["Tapping", "Rest", "drift_0"],
        design_matrix=rng.standard_normal((n_timepoints, n_regressors)),
        noise_model="ar1",
        metadata={"r_squared": rng.uniform(0.3, 0.9, (n_channels,))},
    )


# ---------------------------------------------------------------------------
# render_glm_topo tests
# ---------------------------------------------------------------------------


class TestRenderGlmTopo:
    """Tests for render_glm_topo."""

    def test_returns_figure(self, sample_glm_result: GLMResult) -> None:
        fig = render_glm_topo(sample_glm_result, "Tapping")
        assert isinstance(fig, go.Figure)

    def test_figure_has_bar_trace(self, sample_glm_result: GLMResult) -> None:
        fig = render_glm_topo(sample_glm_result, "Tapping")
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_bar_has_correct_channels(
        self, sample_glm_result: GLMResult
    ) -> None:
        fig = render_glm_topo(sample_glm_result, "Tapping")
        bar = fig.data[0]
        assert list(bar.x) == sample_glm_result.channel_names

    def test_stat_theta(self, sample_glm_result: GLMResult) -> None:
        fig = render_glm_topo(sample_glm_result, "Tapping", stat="theta")
        assert isinstance(fig, go.Figure)
        assert "Coefficient" in fig.layout.yaxis.title.text

    def test_invalid_stat_raises(self, sample_glm_result: GLMResult) -> None:
        with pytest.raises(ValueError, match="stat must be"):
            render_glm_topo(sample_glm_result, "Tapping", stat="invalid")

    def test_invalid_regressor_raises(
        self, sample_glm_result: GLMResult
    ) -> None:
        with pytest.raises(KeyError, match="not found"):
            render_glm_topo(sample_glm_result, "NonExistent")

    def test_significance_coloring(
        self, sample_glm_result: GLMResult
    ) -> None:
        # Force all p-values below threshold for first regressor
        sample_glm_result.p_values[0, :] = 0.01
        fig = render_glm_topo(
            sample_glm_result, "Tapping", significance_threshold=0.05
        )
        bar = fig.data[0]
        # All bars should be blue (significant)
        assert all(c == "#2196F3" for c in bar.marker.color)

    def test_custom_threshold(self, sample_glm_result: GLMResult) -> None:
        fig = render_glm_topo(
            sample_glm_result, "Tapping", significance_threshold=0.001
        )
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# render_glm_summary tests
# ---------------------------------------------------------------------------


class TestRenderGlmSummary:
    """Tests for render_glm_summary."""

    def test_returns_div(self, sample_glm_result: GLMResult) -> None:
        layout = render_glm_summary(sample_glm_result)
        assert isinstance(layout, html.Div)

    def test_contains_regressor_count(
        self, sample_glm_result: GLMResult
    ) -> None:
        layout = render_glm_summary(sample_glm_result)
        text = _extract_text(layout)
        assert "3" in text  # n_regressors

    def test_contains_channel_count(
        self, sample_glm_result: GLMResult
    ) -> None:
        layout = render_glm_summary(sample_glm_result)
        text = _extract_text(layout)
        assert "8" in text  # n_channels

    def test_contains_noise_model(
        self, sample_glm_result: GLMResult
    ) -> None:
        layout = render_glm_summary(sample_glm_result)
        text = _extract_text(layout)
        assert "ar1" in text

    def test_contains_r_squared_when_present(
        self, sample_glm_result: GLMResult
    ) -> None:
        layout = render_glm_summary(sample_glm_result)
        text = _extract_text(layout)
        assert "R²" in text

    def test_no_r_squared_when_absent(
        self, sample_glm_result: GLMResult
    ) -> None:
        sample_glm_result.metadata = {}
        layout = render_glm_summary(sample_glm_result)
        text = _extract_text(layout)
        assert "R²" not in text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(component: html.Div) -> str:
    """Recursively extract all text from a Dash component tree."""
    parts: list[str] = []

    def _walk(node: object) -> None:
        if isinstance(node, str):
            parts.append(node)
        elif hasattr(node, "children"):
            children = node.children
            if isinstance(children, list):
                for child in children:
                    _walk(child)
            elif children is not None:
                _walk(children)

    _walk(component)
    return " ".join(parts)
