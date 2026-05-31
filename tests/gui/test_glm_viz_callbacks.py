"""Tests for GLM tab viz callbacks (T-036).

Covers:
- _find_glm_result: unit tests for the helper that scans results
- update_glm_tab: integration with _VIZ_CACHE (None and real GLMResult)
- update_glm_topo: integration with _VIZ_CACHE (None, no selector, real GLMResult)
- Layout smoke: tab-glm present in create_app().layout
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from dash import dcc, html

from nirspy.domain.block import BlockResult
from nirspy.domain.glm_result import GLMResult
from nirspy.gui.callbacks.execution_callbacks import _VIZ_CACHE
from nirspy.gui.callbacks.viz_callbacks import (
    _find_glm_result,
    update_glm_tab,
    update_glm_topo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_glm_result() -> GLMResult:
    """Minimal GLMResult with all required fields."""
    rng = np.random.default_rng(0)
    n_reg, n_ch, n_tp = 2, 2, 50
    return GLMResult(
        regressor_names=["Tapping_Left", "Tapping_Right"],
        channel_names=["S1_D1 hbo", "S1_D2 hbo"],
        t_stats=rng.standard_normal((n_reg, n_ch)),
        p_values=rng.uniform(0, 1, (n_reg, n_ch)),
        theta=rng.standard_normal((n_reg, n_ch)),
        mse=rng.uniform(0.001, 0.01, (n_ch,)),
        design_matrix=rng.standard_normal((n_tp, n_reg)),
        noise_model="ar1",
    )


def _make_block_result(data: Any) -> BlockResult:
    return BlockResult(data=data, block_id="glm_block")


def _inject_cache(results: list[Any]) -> dict[str, str]:
    """Push results into _VIZ_CACHE and return a run_results dict."""
    key = str(uuid.uuid4())
    _VIZ_CACHE[key] = {"results": results}
    return {"cache_key": key}


# ---------------------------------------------------------------------------
# TestFindGlmResult
# ---------------------------------------------------------------------------


class TestFindGlmResult:
    def test_finds_glm_result(self) -> None:
        glm = _make_glm_result()
        results = [
            _make_block_result(None),
            _make_block_result(glm),
        ]
        assert _find_glm_result(results) is glm

    def test_returns_none_when_absent(self) -> None:
        results = [
            _make_block_result(None),
            _make_block_result("not a glm"),
        ]
        assert _find_glm_result(results) is None

    def test_returns_last_glm_when_multiple(self) -> None:
        glm1 = _make_glm_result()
        glm2 = _make_glm_result()
        results = [
            _make_block_result(glm1),
            _make_block_result(glm2),
        ]
        assert _find_glm_result(results) is glm2

    def test_empty_results_returns_none(self) -> None:
        assert _find_glm_result([]) is None

    def test_single_non_glm_result(self) -> None:
        results = [_make_block_result(42)]
        assert _find_glm_result(results) is None

    def test_glm_at_first_position(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm), _make_block_result(None)]
        # reversed scan returns glm (only one GLMResult in the chain)
        assert _find_glm_result(results) is glm

    def test_mock_data_not_confused_with_glm(self) -> None:
        mock = MagicMock()
        results = [_make_block_result(mock)]
        # MagicMock is not an instance of GLMResult
        assert _find_glm_result(results) is None


# ---------------------------------------------------------------------------
# TestUpdateGlmTab
# ---------------------------------------------------------------------------


class TestUpdateGlmTab:
    def test_returns_empty_on_none_run_results(self) -> None:
        options, value, disabled, summary = update_glm_tab(None)
        assert options == []
        assert value is None
        assert disabled is True
        assert summary == ""

    def test_returns_empty_when_no_cache_key(self) -> None:
        options, value, disabled, summary = update_glm_tab({})
        assert options == []
        assert value is None
        assert disabled is True

    def test_returns_empty_when_cache_miss(self) -> None:
        run_results = {"cache_key": "nonexistent-key-xyz"}
        options, value, disabled, summary = update_glm_tab(run_results)
        assert options == []
        assert disabled is True

    def test_returns_empty_when_no_glm_in_results(self) -> None:
        results = [_make_block_result(None), _make_block_result("other")]
        run_results = _inject_cache(results)
        options, value, disabled, summary = update_glm_tab(run_results)
        assert options == []
        assert value is None
        assert disabled is True
        assert summary == ""

    def test_populates_options_from_glm_result(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        options, value, disabled, summary = update_glm_tab(run_results)
        assert len(options) == 2
        assert options[0] == {"label": "Tapping_Left", "value": "Tapping_Left"}
        assert options[1] == {"label": "Tapping_Right", "value": "Tapping_Right"}

    def test_first_regressor_selected_by_default(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        _, value, _, _ = update_glm_tab(run_results)
        assert value == "Tapping_Left"

    def test_dropdown_enabled_when_glm_present(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        _, _, disabled, _ = update_glm_tab(run_results)
        assert disabled is False

    def test_summary_is_dash_component_when_glm_present(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        _, _, _, summary = update_glm_tab(run_results)
        # render_glm_summary returns html.Div
        assert isinstance(summary, html.Div)

    def test_summary_contains_noise_model(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        _, _, _, summary = update_glm_tab(run_results)
        assert "ar1" in str(summary)

    def test_summary_contains_channel_count(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        _, _, _, summary = update_glm_tab(run_results)
        assert "2" in str(summary)  # 2 channels


# ---------------------------------------------------------------------------
# TestUpdateGlmTopo
# ---------------------------------------------------------------------------


class TestUpdateGlmTopo:
    def test_returns_placeholder_on_none_results(self) -> None:
        result = update_glm_topo(None, None)
        assert isinstance(result, html.P)
        assert "Run pipeline" in str(result)

    def test_returns_placeholder_when_no_selector(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        result = update_glm_topo(run_results, None)
        assert isinstance(result, html.P)

    def test_returns_placeholder_on_cache_miss(self) -> None:
        run_results = {"cache_key": "missing-key-abc"}
        result = update_glm_topo(run_results, "Tapping_Left")
        assert isinstance(result, html.P)

    def test_returns_graph_for_valid_regressor(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        result = update_glm_topo(run_results, "Tapping_Left")
        assert isinstance(result, dcc.Graph)

    def test_returns_graph_for_second_regressor(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        result = update_glm_topo(run_results, "Tapping_Right")
        assert isinstance(result, dcc.Graph)

    def test_returns_no_glm_placeholder_on_unknown_regressor(self) -> None:
        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        result = update_glm_topo(run_results, "NonExistent")
        assert isinstance(result, html.P)
        assert "No GLM results" in str(result)

    def test_returns_placeholder_when_no_glm_in_results(self) -> None:
        results = [_make_block_result(None)]
        run_results = _inject_cache(results)
        result = update_glm_topo(run_results, "Tapping_Left")
        assert isinstance(result, html.P)
        assert "No GLM results" in str(result)

    def test_graph_figure_has_bar_trace(self) -> None:

        glm = _make_glm_result()
        results = [_make_block_result(glm)]
        run_results = _inject_cache(results)
        result = update_glm_topo(run_results, "Tapping_Left")
        assert isinstance(result, dcc.Graph)
        fig = result.figure
        assert len(fig["data"]) == 1
        assert fig["data"][0]["type"] == "bar"


# ---------------------------------------------------------------------------
# TestGlmTabLayout
# ---------------------------------------------------------------------------


class TestGlmTabLayout:
    def test_tab_glm_present_in_layout(self) -> None:
        from nirspy.gui.app import create_app

        layout_str = str(create_app().layout)
        assert "tab-glm" in layout_str

    def test_glm_summary_container_present(self) -> None:
        from nirspy.gui.app import create_app

        layout_str = str(create_app().layout)
        assert "glm-summary-container" in layout_str

    def test_glm_regressor_selector_present(self) -> None:
        from nirspy.gui.app import create_app

        layout_str = str(create_app().layout)
        assert "glm-regressor-selector" in layout_str

    def test_glm_topo_container_present(self) -> None:
        from nirspy.gui.app import create_app

        layout_str = str(create_app().layout)
        assert "glm-topo-container" in layout_str

    def test_glm_tab_label(self) -> None:
        from nirspy.gui.app import create_app

        layout_str = str(create_app().layout)
        assert "GLM" in layout_str
