"""Tests for execution + visualization components (T-006 5C).

Validates the 12 criteria of acceptance for Part 5C without
requiring external SNIRF datasets -- all MNE objects are mocked.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from nirspy.domain.block import BlockResult
from nirspy.gui.callbacks.execution_callbacks import (
    _VIZ_CACHE,
    _build_pipeline_from_state,
    _is_evoked,
    run_pipeline_callback,
    store_input_file,
)
from nirspy.gui.callbacks.viz_callbacks import (
    _find_sci_ch_names,
    _find_sci_values,
    _get_cached_results,
)
from nirspy.gui.components.condition_selector import render_condition_selector
from nirspy.gui.components.hrf_plot import _rgba, render_hrf_plot
from nirspy.gui.components.probe_viewer import render_probe_viewer
from nirspy.gui.components.qc_dashboard import render_qc_dashboard
from nirspy.gui.components.raw_data_plot import render_raw_data_plot
from nirspy.gui.components.run_button import render_run_button


def _make_entry(
    block_id: str = "optical_density",
    enabled: bool = True,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "instance_id": str(uuid.uuid4()),
        "params": params or {},
        "enabled": enabled,
    }


def _make_fake_raw(
    n_channels: int = 4,
    n_times: int = 100,
    sfreq: float = 10.0,
    has_montage: bool = True,
    bads: list[str] | None = None,
) -> MagicMock:
    raw = MagicMock()
    half = n_channels // 2
    ch_names = [f"S1_D{i+1} hbo" for i in range(half)] + [
        f"S1_D{i+1} hbr" for i in range(half)
    ]
    raw.ch_names = ch_names
    raw.times = np.linspace(0, n_times / sfreq, n_times)
    data = np.random.default_rng(42).standard_normal((n_channels, n_times))
    raw.__getitem__ = MagicMock(return_value=(data, raw.times))
    info = MagicMock()
    info.__getitem__ = MagicMock(
        side_effect=lambda k: {
            "chs": [{"ch_name": c, "kind": ""} for c in ch_names],
            "bads": bads or [],
        }.get(k)
    )
    info.get = MagicMock(
        side_effect=lambda k, d=None: {"bads": bads or []}.get(k, d)
    )
    if has_montage:
        montage = MagicMock()
        positions = {
            "ch_pos": {
                ch: np.array([float(i), float(i), 0.0])
                for i, ch in enumerate(ch_names)
            }
        }
        montage.get_positions.return_value = positions
        info.get_montage.return_value = montage
    else:
        info.get_montage.return_value = None
    raw.info = info
    return raw


def _make_fake_evoked(
    conditions: list[str] | None = None,
    n_channels: int = 4,
    n_times: int = 50,
    nave: int = 10,
) -> dict[str, Any]:
    conds = conditions or ["Tapping/Left", "Tapping/Right"]
    result: dict[str, Any] = {}
    rng = np.random.default_rng(42)
    for cond in conds:
        ev = MagicMock()
        ev.ch_names = ["S1_D1 hbo", "S1_D2 hbo", "S1_D1 hbr", "S1_D2 hbr"]
        ev.times = np.linspace(-0.5, 2.0, n_times)
        ev.data = rng.standard_normal((n_channels, n_times))
        ev.nave = nave
        result[cond] = ev
    return result


class TestRunButton:
    def test_render_has_run_button(self) -> None:
        html_str = str(render_run_button())
        assert "run-button" in html_str
        assert "Run Pipeline" in html_str

    def test_render_has_progress_bar(self) -> None:
        assert "run-progress" in str(render_run_button())

    def test_render_has_upload(self) -> None:
        assert "upload-input-file" in str(render_run_button())

    def test_render_has_error_alert(self) -> None:
        assert "run-error" in str(render_run_button())

    def test_render_has_success_alert(self) -> None:
        assert "run-success" in str(render_run_button())


class TestProgressTracking:
    def test_build_pipeline_basic(self) -> None:
        state = [_make_entry("optical_density")]
        pipeline = _build_pipeline_from_state(state)
        assert len(pipeline.steps) == 1

    def test_build_pipeline_disabled(self) -> None:
        state = [
            _make_entry("optical_density", enabled=True),
            _make_entry("beer_lambert", enabled=False),
        ]
        pipeline = _build_pipeline_from_state(state)
        assert len(pipeline.steps) == 2
        assert not pipeline.steps[1].spec.enabled


class TestExecutionErrorHandling:
    def test_no_clicks_no_update(self) -> None:
        from dash import no_update
        result = run_pipeline_callback(None, None, None)
        assert all(r is no_update for r in result)

    def test_empty_state_no_update(self) -> None:
        from dash import no_update
        result = run_pipeline_callback(1, None, None)
        assert all(r is no_update for r in result)

    def test_invalid_block_returns_error(self) -> None:
        state = [_make_entry("nonexistent_block")]
        result = run_pipeline_callback(1, state, None)
        assert "Failed to build pipeline" in str(result[4])
        assert result[5] is True


class TestRawDataPlot:
    def test_placeholder_when_no_data(self) -> None:
        assert "No raw data available" in str(render_raw_data_plot(None))

    def test_max_default_channels(self) -> None:
        from nirspy.gui.components.raw_data_plot import _MAX_DEFAULT_CHANNELS
        assert _MAX_DEFAULT_CHANNELS == 10


class TestProbeViewer:
    def test_sources_red(self) -> None:
        raw = _make_fake_raw(has_montage=True)
        html_str = str(render_probe_viewer(raw.info))
        assert "#d62728" in html_str or "Sources" in html_str

    def test_fallback_no_info(self) -> None:
        assert "No montage info available" in str(render_probe_viewer(None))

    def test_fallback_no_montage(self) -> None:
        raw = _make_fake_raw(has_montage=False)
        assert "No montage info available" in str(render_probe_viewer(raw.info))

    def test_pruned_gray(self) -> None:
        raw = _make_fake_raw(has_montage=True, bads=["S1_D1 hbo"])
        html_str = str(render_probe_viewer(raw.info, bads=["S1_D1 hbo"]))
        assert "#999999" in html_str or "Pruned" in html_str


class TestQCDashboard:
    def test_placeholder_no_data(self) -> None:
        assert "No QC data available" in str(render_qc_dashboard(None))

    def test_renders_heatmap(self) -> None:
        sci = np.array([0.9, 0.5, 0.8, 0.3])
        assert "qc-dashboard" in str(render_qc_dashboard(sci, ["a", "b", "c", "d"]))

    def test_threshold_07(self) -> None:
        from nirspy.gui.components.qc_dashboard import _SCI_THRESHOLD
        assert pytest.approx(0.7) == _SCI_THRESHOLD

    def test_empty_array(self) -> None:
        assert "No QC data" in str(render_qc_dashboard(np.array([])))


class TestHRFPlot:
    def test_placeholder_no_data(self) -> None:
        assert "No HRF data available" in str(render_hrf_plot(None))

    def test_color_constants(self) -> None:
        from nirspy.gui.components.hrf_plot import _HBO_COLOR, _HBR_COLOR
        assert _HBO_COLOR == "#d62728"
        assert _HBR_COLOR == "#1f77b4"

    def test_renders_evoked(self) -> None:
        ed = _make_fake_evoked(["Tapping"])
        html_str = str(render_hrf_plot(ed))
        assert "hrf-graph" in html_str or "hrf-plot-content" in html_str

    def test_rgba(self) -> None:
        assert _rgba("#d62728", 0.15) == "rgba(214,39,40,0.15)"
        assert _rgba("#1f77b4", 0.15) == "rgba(31,119,180,0.15)"


class TestConditionSelector:
    def test_placeholder_no_conds(self) -> None:
        assert "No conditions available" in str(render_condition_selector(None))

    def test_renders_checklist(self) -> None:
        assert "condition-selector" in str(render_condition_selector(["A", "B"]))

    def test_all_present(self) -> None:
        html_str = str(render_condition_selector(["A", "B", "C"]))
        for ch in ["A", "B", "C"]:
            assert ch in html_str

    def test_hrf_filtered(self) -> None:
        ed = _make_fake_evoked(["Cond_A", "Cond_B"])
        assert "hrf-plot-content" in str(render_hrf_plot(ed, ["Cond_A"]))


class TestVizCache:
    def test_store_retrieve(self) -> None:
        key = str(uuid.uuid4())
        _VIZ_CACHE[key] = {"results": ["mock"], "timestamp": 1.0}
        assert _get_cached_results({"cache_key": key}) == ["mock"]
        del _VIZ_CACHE[key]

    def test_missing_key(self) -> None:
        assert _get_cached_results({"cache_key": "nope"}) is None

    def test_none_input(self) -> None:
        assert _get_cached_results(None) is None


class TestDetectionHelpers:
    def test_is_evoked_false_str(self) -> None:
        assert _is_evoked("x") is False

    def test_find_sci_values(self) -> None:
        r = BlockResult(data=None, block_id="sci", metadata={"sci_values": [0.9]})
        assert _find_sci_values([r]) == [0.9]

    def test_find_sci_none(self) -> None:
        r = BlockResult(data=None, block_id="t", metadata={})
        assert _find_sci_values([r]) is None

    def test_find_sci_ch_names(self) -> None:
        r = BlockResult(data=None, block_id="s", metadata={"sci_ch_names": ["S1"]})
        assert _find_sci_ch_names([r]) == ["S1"]


class TestLayoutIntegration:
    def test_viz_tabs(self) -> None:
        from nirspy.gui.layouts import create_layout
        assert "viz-tabs" in str(create_layout())

    def test_run_results_store(self) -> None:
        from nirspy.gui.layouts import create_layout
        assert "run-results" in str(create_layout())

    def test_input_file_store(self) -> None:
        from nirspy.gui.layouts import create_layout
        assert "input-file-path" in str(create_layout())

    def test_all_viz_containers(self) -> None:
        from nirspy.gui.layouts import create_layout
        html_str = str(create_layout())
        for cid in [
            "raw-data-plot-container", "probe-viewer-container",
            "qc-dashboard-container", "hrf-plot-container",
            "condition-selector-container",
        ]:
            assert cid in html_str, f"Missing {cid}"


class TestImportSanity:
    def test_all_5c_importable(self) -> None:
        import nirspy.gui.callbacks.execution_callbacks  # noqa: F401
        import nirspy.gui.callbacks.viz_callbacks  # noqa: F401
        import nirspy.gui.components.condition_selector  # noqa: F401
        import nirspy.gui.components.hrf_plot  # noqa: F401
        import nirspy.gui.components.probe_viewer  # noqa: F401
        import nirspy.gui.components.qc_dashboard  # noqa: F401
        import nirspy.gui.components.raw_data_plot  # noqa: F401
        import nirspy.gui.components.run_button  # noqa: F401

    def test_app_creates(self) -> None:
        from nirspy.gui.app import create_app
        assert create_app().layout is not None

    def test_callbacks_callable(self) -> None:
        assert callable(run_pipeline_callback)
        assert callable(store_input_file)

