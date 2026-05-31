"""Tests for HTMLReportBlock (T-038)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nirspy.blocks.report import HTMLReportBlock, HTMLReportParams
from nirspy.domain.glm_result import GLMResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeContext:
    """Minimal context mock."""

    def __init__(self, extra: dict[str, Any] | None = None) -> None:
        self.extra = extra or {}


@pytest.fixture
def sample_glm_result() -> GLMResult:
    """Create a minimal GLMResult for testing."""
    rng = np.random.default_rng(42)
    n_regressors = 2
    n_channels = 4
    n_timepoints = 50

    return GLMResult(
        theta=rng.standard_normal((n_regressors, n_channels)),
        t_stats=rng.standard_normal((n_regressors, n_channels)) * 2,
        p_values=rng.uniform(0, 1, (n_regressors, n_channels)),
        mse=rng.uniform(0.01, 0.1, (n_channels,)),
        channel_names=[f"S{i}D{i} hbo" for i in range(1, n_channels + 1)],
        regressor_names=["Tapping", "Rest"],
        design_matrix=rng.standard_normal((n_timepoints, n_regressors)),
        noise_model="ar1",
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame."""
    return pd.DataFrame({
        "channel": ["S1D1", "S2D2", "S3D3"],
        "value": [1.0, 2.0, 3.0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHTMLReportBlock:
    """Tests for HTMLReportBlock."""

    def test_generates_html_from_glm(self, sample_glm_result: GLMResult, tmp_path: Path) -> None:
        """Block generates HTML file from GLMResult input."""
        output = tmp_path / "test_report.html"
        params = HTMLReportParams(output_path=str(output), title="Test Report")
        block = HTMLReportBlock(params=params)

        context = _FakeContext(extra={"pipeline_steps": ["load_snirf", "glm"]})
        result = block.run(context, {"data": sample_glm_result})

        assert output.exists()
        assert result.data is None  # sink block
        assert result.block_id == "html_report"
        assert result.metadata["output_file"] == str(output)
        assert result.metadata["file_size_bytes"] > 0

    def test_html_contains_title(self, sample_glm_result: GLMResult, tmp_path: Path) -> None:
        """Generated HTML contains the configured title."""
        output = tmp_path / "report.html"
        params = HTMLReportParams(output_path=str(output), title="My Analysis")
        block = HTMLReportBlock(params=params)

        block.run(None, {"data": sample_glm_result})

        content = output.read_text(encoding="utf-8")
        assert "My Analysis" in content

    def test_html_contains_pipeline_steps(
        self, sample_glm_result: GLMResult, tmp_path: Path
    ) -> None:
        """Generated HTML includes pipeline steps from context."""
        output = tmp_path / "report.html"
        params = HTMLReportParams(output_path=str(output))
        block = HTMLReportBlock(params=params)

        context = _FakeContext(extra={"pipeline_steps": ["load_snirf", "optical_density", "glm"]})
        block.run(context, {"data": sample_glm_result})

        content = output.read_text(encoding="utf-8")
        assert "load_snirf" in content
        assert "optical_density" in content

    def test_html_contains_data_table(self, sample_glm_result: GLMResult, tmp_path: Path) -> None:
        """Generated HTML contains a data table for GLM results."""
        output = tmp_path / "report.html"
        params = HTMLReportParams(output_path=str(output))
        block = HTMLReportBlock(params=params)

        block.run(None, {"data": sample_glm_result})

        content = output.read_text(encoding="utf-8")
        assert "<table" in content
        assert "Tapping" in content

    def test_dataframe_input(self, sample_dataframe: pd.DataFrame, tmp_path: Path) -> None:
        """Block handles DataFrame input."""
        output = tmp_path / "report.html"
        params = HTMLReportParams(output_path=str(output))
        block = HTMLReportBlock(params=params)

        block.run(None, {"data": sample_dataframe})

        content = output.read_text(encoding="utf-8")
        assert output.exists()
        assert "S1D1" in content

    def test_no_plots_when_disabled(self, sample_glm_result: GLMResult, tmp_path: Path) -> None:
        """No Plotly embed when include_plots=False."""
        output = tmp_path / "report.html"
        params = HTMLReportParams(output_path=str(output), include_plots=False)
        block = HTMLReportBlock(params=params)

        block.run(None, {"data": sample_glm_result})

        content = output.read_text(encoding="utf-8")
        assert "plotly-graph-div" not in content

    def test_qc_section_from_context(self, sample_glm_result: GLMResult, tmp_path: Path) -> None:
        """QC section is included when context has qc data."""
        output = tmp_path / "report.html"
        params = HTMLReportParams(output_path=str(output))
        block = HTMLReportBlock(params=params)

        context = _FakeContext(extra={"qc": {"good_channels": "15/16", "sci_mean": "0.85"}})
        block.run(context, {"data": sample_glm_result})

        content = output.read_text(encoding="utf-8")
        assert "good_channels" in content
        assert "15/16" in content

    def test_empty_input_raises(self, tmp_path: Path) -> None:
        """Block raises ValidationError with empty inputs."""
        from nirspy.domain.exceptions import ValidationError

        output = tmp_path / "report.html"
        params = HTMLReportParams(output_path=str(output))
        block = HTMLReportBlock(params=params)

        with pytest.raises(ValidationError):
            block.run(None, {})

    def test_creates_parent_dirs(self, sample_glm_result: GLMResult, tmp_path: Path) -> None:
        """Block creates parent directories if they do not exist."""
        output = tmp_path / "subdir" / "deep" / "report.html"
        params = HTMLReportParams(output_path=str(output))
        block = HTMLReportBlock(params=params)

        block.run(None, {"data": sample_glm_result})

        assert output.exists()

    def test_spec_attributes(self) -> None:
        """Block spec has correct attributes."""
        block = HTMLReportBlock()
        assert block.spec.block_id == "html_report"
        assert block.spec.input_type.value == "any"
        assert block.spec.output_type.value == "none"
