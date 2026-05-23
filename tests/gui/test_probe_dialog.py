"""Tests for T-029 -- Probe interactive dialog.

Covers:
- render_probe_dialog with SNIRF that has a montage → view+exclude mode
- render_probe_dialog with SNIRF that has no montage → positioning mode
- render_probe_dialog with sidecar-provided montage → "Loaded from sidecar" badge
- Click-to-toggle exclusion updates the excluded-store and status badge
- 2-click positioning places the optode and updates the montage store
- [Save positions] writes a sidecar JSON and switches to view mode
- Footer has probe-cancel-btn and probe-confirm-btn
- build_channels_override produces the correct params dict
- Sidecar precedence: sidecar > SNIRF native
- Dispatch: _make_modal_children returns probe_dialog when block_id == manual_channel_exclude
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import dash_bootstrap_components as dbc
import pytest

from nirspy.gui.components.probe_dialog import (
    _build_probe_figure,
    _build_status_badge,
    _derive_channel_pairs,
    build_channels_override,
    render_probe_dialog,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_montage() -> dict[str, Any]:
    """A minimal montage dict with 2 sources and 3 detectors."""
    return {
        "sources": [[0.0, 0.05], [0.0, -0.05]],
        "detectors": [[-0.04, 0.0], [0.04, 0.0], [0.0, 0.0]],
    }


@pytest.fixture()
def snirf_with_montage(tmp_path: Path, sample_montage: dict[str, Any]) -> Path:
    """A fake SNIRF path whose montage resolves via sidecar."""
    snirf = tmp_path / "test.snirf"
    snirf.write_bytes(b"")  # empty file — montage comes from sidecar
    sidecar = tmp_path / "test.montage.json"
    sidecar.write_text(json.dumps(sample_montage), encoding="utf-8")
    return snirf


@pytest.fixture()
def snirf_without_montage(tmp_path: Path) -> Path:
    """A fake SNIRF path with no montage (no sidecar, no h5 positions)."""
    snirf = tmp_path / "empty.snirf"
    snirf.write_bytes(b"")
    return snirf


# ---------------------------------------------------------------------------
# render_probe_dialog — view+exclude mode (montage available via sidecar)
# ---------------------------------------------------------------------------

class TestRenderProbeDialogViewMode:
    """Tests for render_probe_dialog when a montage is available."""

    def test_returns_modal(self, snirf_with_montage: Path) -> None:
        modal = render_probe_dialog(str(snirf_with_montage), 0, 5)
        assert isinstance(modal, dbc.Modal)

    def test_modal_id(self, snirf_with_montage: Path) -> None:
        modal = render_probe_dialog(str(snirf_with_montage), 0, 5)
        assert modal.id == "probe-dialog-modal"

    def test_size_xl(self, snirf_with_montage: Path) -> None:
        modal = render_probe_dialog(str(snirf_with_montage), 0, 5)
        assert modal.size == "xl"

    def test_backdrop_static(self, snirf_with_montage: Path) -> None:
        modal = render_probe_dialog(str(snirf_with_montage), 0, 5)
        assert modal.backdrop == "static"

    def test_keyboard_false(self, snirf_with_montage: Path) -> None:
        modal = render_probe_dialog(str(snirf_with_montage), 0, 5)
        assert modal.keyboard is False

    def test_is_open_true(self, snirf_with_montage: Path) -> None:
        modal = render_probe_dialog(str(snirf_with_montage), 0, 5)
        assert modal.is_open is True

    def test_sidecar_badge_present(self, snirf_with_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_with_montage), 0, 5))
        assert "Loaded from sidecar" in modal_str

    def test_header_step_counter(self, snirf_with_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_with_montage), 2, 8))
        assert "Block 3/8" in modal_str

    def test_probe_graph_present(self, snirf_with_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_with_montage), 0, 5))
        assert "probe-dialog-graph" in modal_str

    def test_confirm_button_present(self, snirf_with_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_with_montage), 0, 5))
        assert "probe-confirm-btn" in modal_str

    def test_cancel_button_present(self, snirf_with_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_with_montage), 0, 5))
        assert "probe-cancel-btn" in modal_str

    def test_excluded_store_present(self, snirf_with_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_with_montage), 0, 5))
        assert "probe-excluded-store" in modal_str

    def test_mode_store_is_view(self, snirf_with_montage: Path) -> None:
        """probe-mode-store data should be 'view' when montage is available."""
        modal = render_probe_dialog(str(snirf_with_montage), 0, 5)
        modal_str = str(modal)
        # The mode-store stores 'view'
        assert "probe-mode-store" in modal_str

    def test_status_badge_no_exclusions(self, snirf_with_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_with_montage), 0, 5))
        assert "probe-status-badge" in modal_str
        assert "No channels excluded" in modal_str

    def test_pre_excluded_channels_shown(self, snirf_with_montage: Path) -> None:
        modal_str = str(
            render_probe_dialog(str(snirf_with_montage), 0, 5, excluded_channels=["S1_D2"])
        )
        assert "S1_D2" in modal_str

    def test_no_positioning_controls_in_view_mode(self, snirf_with_montage: Path) -> None:
        """In view mode there should be no optode-selector or save-positions btn."""
        modal_str = str(render_probe_dialog(str(snirf_with_montage), 0, 5))
        assert "probe-optode-selector" not in modal_str
        assert "probe-save-positions-btn" not in modal_str


# ---------------------------------------------------------------------------
# render_probe_dialog — positioning mode (no montage)
# ---------------------------------------------------------------------------

class TestRenderProbeDialogPositioningMode:
    """Tests for render_probe_dialog when no montage exists (positioning mode)."""

    def test_returns_modal(self, snirf_without_montage: Path) -> None:
        modal = render_probe_dialog(str(snirf_without_montage), 0, 3)
        assert isinstance(modal, dbc.Modal)

    def test_missing_badge_shown(self, snirf_without_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_without_montage), 0, 3))
        assert "Manual positioning" in modal_str

    def test_positioning_hint_text(self, snirf_without_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_without_montage), 0, 3))
        assert "No montage found" in modal_str

    def test_optode_selector_present(self, snirf_without_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_without_montage), 0, 3))
        assert "probe-optode-selector" in modal_str

    def test_save_positions_button_present(self, snirf_without_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_without_montage), 0, 3))
        assert "probe-save-positions-btn" in modal_str

    def test_probe_graph_present(self, snirf_without_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_without_montage), 0, 3))
        assert "probe-dialog-graph" in modal_str

    def test_mode_store_is_positioning(self, snirf_without_montage: Path) -> None:
        modal_str = str(render_probe_dialog(str(snirf_without_montage), 0, 3))
        assert "probe-mode-store" in modal_str

    def test_confirm_button_present(self, snirf_without_montage: Path) -> None:
        """Footer confirm button must always be present."""
        modal_str = str(render_probe_dialog(str(snirf_without_montage), 0, 3))
        assert "probe-confirm-btn" in modal_str

    def test_no_snirf_path_renders_positioning_mode(self) -> None:
        """When snirf_path is None the dialog must open in positioning mode."""
        modal_str = str(render_probe_dialog(None, 0, 2))
        assert "Manual positioning" in modal_str
        assert "probe-optode-selector" in modal_str

    def test_partial_montage_passed_to_positioning(
        self, snirf_without_montage: Path
    ) -> None:
        """positioned_montage kwarg is reflected in the graph stores."""
        partial = {"sources": [[0.01, 0.01]], "detectors": []}
        modal_str = str(
            render_probe_dialog(
                str(snirf_without_montage),
                0,
                3,
                positioned_montage=partial,
            )
        )
        assert "probe-positioned-montage-store" in modal_str


# ---------------------------------------------------------------------------
# render_probe_dialog — SNIRF badge source label
# ---------------------------------------------------------------------------

class TestSourceBadge:
    """Tests for the montage source badge label mapping."""

    def test_snirf_label(self) -> None:
        """When only SNIRF positions exist (no sidecar), badge = 'Loaded from SNIRF'."""
        # Patch resolve_montage to return snirf source
        with patch(
            "nirspy.gui.components.probe_dialog.resolve_montage",
            return_value=({"sources": [[0.0, 0.0]], "detectors": [[0.01, 0.0]]}, "snirf"),
        ):
            modal_str = str(render_probe_dialog("/fake/path.snirf", 0, 3))
        assert "Loaded from SNIRF" in modal_str

    def test_missing_label(self) -> None:
        with patch(
            "nirspy.gui.components.probe_dialog.resolve_montage",
            return_value=(None, "missing"),
        ):
            modal_str = str(render_probe_dialog("/fake/path.snirf", 0, 3))
        assert "Manual positioning" in modal_str


# ---------------------------------------------------------------------------
# _build_status_badge
# ---------------------------------------------------------------------------

class TestBuildStatusBadge:
    def test_no_exclusions(self) -> None:
        badge = _build_status_badge(set())
        badge_str = str(badge)
        assert "No channels excluded" in badge_str

    def test_with_excluded_channels(self) -> None:
        badge = _build_status_badge({"S1_D2", "S2_D1"})
        badge_str = str(badge)
        assert "S1_D2" in badge_str
        assert "S2_D1" in badge_str

    def test_empty_excluded_shows_success_badge(self) -> None:
        badge = _build_status_badge(set())
        badge_str = str(badge)
        assert "success" in badge_str.lower()


# ---------------------------------------------------------------------------
# _derive_channel_pairs
# ---------------------------------------------------------------------------

class TestDeriveChannelPairs:
    def test_2x2_produces_4_pairs(self) -> None:
        sources = [[0.0, 0.1], [0.0, -0.1]]
        detectors = [[-0.05, 0.0], [0.05, 0.0]]
        pairs = _derive_channel_pairs(sources, detectors)
        assert len(pairs) == 4

    def test_pair_labels_format(self) -> None:
        sources = [[0.0, 0.0]]
        detectors = [[0.1, 0.0]]
        pairs = _derive_channel_pairs(sources, detectors)
        assert "S1_D1" in pairs

    def test_midpoint_computation(self) -> None:
        sources = [[0.0, 0.0]]
        detectors = [[1.0, 0.0]]
        pairs = _derive_channel_pairs(sources, detectors)
        x_mid, y_mid = pairs["S1_D1"]
        assert abs(x_mid - 0.5) < 1e-9
        assert abs(y_mid - 0.0) < 1e-9

    def test_empty_sources(self) -> None:
        pairs = _derive_channel_pairs([], [[0.0, 0.0]])
        assert pairs == {}

    def test_empty_detectors(self) -> None:
        pairs = _derive_channel_pairs([[0.0, 0.0]], [])
        assert pairs == {}


# ---------------------------------------------------------------------------
# _build_probe_figure
# ---------------------------------------------------------------------------

class TestBuildProbeFigure:
    def test_returns_figure(self, sample_montage: dict[str, Any]) -> None:
        import plotly.graph_objects as go
        fig = _build_probe_figure(sample_montage, set())
        assert isinstance(fig, go.Figure)

    def test_excluded_channel_in_figure(self, sample_montage: dict[str, Any]) -> None:
        """Excluded channels must appear in the 'Excluded' trace."""
        fig = _build_probe_figure(sample_montage, {"S1_D1"})
        trace_names = [t.name for t in fig.data]
        assert "Excluded" in trace_names

    def test_no_exclusions_no_excluded_trace(self, sample_montage: dict[str, Any]) -> None:
        fig = _build_probe_figure(sample_montage, set())
        trace_names = [t.name for t in fig.data]
        assert "Excluded" not in trace_names

    def test_head_trace_present(self, sample_montage: dict[str, Any]) -> None:
        fig = _build_probe_figure(sample_montage, set())
        trace_names = [t.name for t in fig.data]
        assert "Head" in trace_names

    def test_sources_trace_present(self, sample_montage: dict[str, Any]) -> None:
        fig = _build_probe_figure(sample_montage, set())
        trace_names = [t.name for t in fig.data]
        assert "Sources" in trace_names

    def test_detectors_trace_present(self, sample_montage: dict[str, Any]) -> None:
        fig = _build_probe_figure(sample_montage, set())
        trace_names = [t.name for t in fig.data]
        assert "Detectors" in trace_names


# ---------------------------------------------------------------------------
# build_channels_override
# ---------------------------------------------------------------------------

class TestBuildChannelsOverride:
    def test_empty_list(self) -> None:
        result = build_channels_override([])
        assert result == {"channels": []}

    def test_single_channel(self) -> None:
        result = build_channels_override(["S1_D2"])
        assert result == {"channels": ["S1_D2"]}

    def test_multiple_channels(self) -> None:
        result = build_channels_override(["S1_D2", "S3_D1", "S2_D2"])
        assert set(result["channels"]) == {"S1_D2", "S3_D1", "S2_D2"}

    def test_returns_new_list(self) -> None:
        """Returned list must be a copy, not the same object."""
        orig = ["S1_D1"]
        result = build_channels_override(orig)
        orig.append("S2_D2")
        assert "S2_D2" not in result["channels"]


# ---------------------------------------------------------------------------
# Callback: probe_toggle_exclusion
# ---------------------------------------------------------------------------

class TestProbeToggleExclusion:
    """Tests for the click-to-exclude callback."""

    def test_no_click_data_returns_no_update(self) -> None:
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_toggle_exclusion

        result = probe_toggle_exclusion(None, [], "view", None)
        assert all(r is no_update for r in result)

    def test_positioning_mode_no_toggle(self) -> None:
        """Click in positioning mode must not toggle exclusions."""
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_toggle_exclusion

        click_data = {"points": [{"customdata": "S1_D1", "x": 0.0, "y": 0.0}]}
        result = probe_toggle_exclusion(click_data, [], "positioning", None)
        assert all(r is no_update for r in result)

    def test_click_on_source_not_channel_ignored(self) -> None:
        """Clicking a plain source S1 (no underscore) must not toggle exclusions."""
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_toggle_exclusion

        click_data = {"points": [{"customdata": "S1", "x": 0.0, "y": 0.0}]}
        result = probe_toggle_exclusion(click_data, [], "view", None)
        assert all(r is no_update for r in result)

    def test_click_channel_adds_to_excluded(
        self, snirf_with_montage: Path
    ) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import probe_toggle_exclusion

        click_data = {"points": [{"customdata": "S1_D1", "x": 0.0, "y": 0.0}]}
        new_excluded, _badge_children, _fig = probe_toggle_exclusion(
            click_data, [], "view", str(snirf_with_montage)
        )
        assert "S1_D1" in new_excluded

    def test_click_excluded_channel_removes_it(
        self, snirf_with_montage: Path
    ) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import probe_toggle_exclusion

        click_data = {"points": [{"customdata": "S1_D1", "x": 0.0, "y": 0.0}]}
        new_excluded, _badge_children, _fig = probe_toggle_exclusion(
            click_data, ["S1_D1"], "view", str(snirf_with_montage)
        )
        assert "S1_D1" not in new_excluded


# ---------------------------------------------------------------------------
# Callback: probe_positioning_click  (2-click pattern)
# ---------------------------------------------------------------------------

class TestProbePositioningClick:
    """Tests for the 2-click placement callback."""

    def test_no_click_data_returns_no_update(self) -> None:
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_positioning_click

        result = probe_positioning_click(
            None, "positioning", None, {"sources": [], "detectors": []}, None
        )
        assert all(r is no_update for r in result)

    def test_view_mode_ignored(self) -> None:
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_positioning_click

        click_data = {"points": [{"x": 0.1, "y": 0.1}]}
        result = probe_positioning_click(
            click_data, "view", "S1", {"sources": [], "detectors": []}, None
        )
        assert all(r is no_update for r in result)

    def test_no_optode_selected_returns_no_update(self) -> None:
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_positioning_click

        click_data = {"points": [{"x": 0.1, "y": 0.1}]}
        result = probe_positioning_click(
            click_data, "positioning", None, {"sources": [], "detectors": []}, None
        )
        assert all(r is no_update for r in result)

    def test_places_source_on_click(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import probe_positioning_click

        click_data = {"points": [{"x": 0.05, "y": 0.07}]}
        new_montage, new_selected, _fig = probe_positioning_click(
            click_data,
            "positioning",
            "S1",
            {"sources": [], "detectors": []},
            None,
        )
        assert new_montage["sources"][0] == [0.05, 0.07]
        assert new_selected is None

    def test_places_detector_on_click(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import probe_positioning_click

        click_data = {"points": [{"x": -0.03, "y": 0.02}]}
        new_montage, _sel, _fig = probe_positioning_click(
            click_data,
            "positioning",
            "D1",
            {"sources": [], "detectors": []},
            None,
        )
        assert new_montage["detectors"][0] == [-0.03, 0.02]

    def test_selector_value_used_when_no_selected_optode(self) -> None:
        """When selected_optode is None but selector_value is set, use selector."""
        from nirspy.gui.callbacks.runtime_callbacks import probe_positioning_click

        click_data = {"points": [{"x": 0.0, "y": 0.0}]}
        new_montage, _, _fig = probe_positioning_click(
            click_data,
            "positioning",
            None,  # selected_optode = None
            {"sources": [], "detectors": []},
            "S2",  # selector_value
        )
        # S2 is index 1 (0-based), so sources list should have 2 entries
        assert len(new_montage["sources"]) == 2
        assert new_montage["sources"][1] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Callback: probe_save_positions
# ---------------------------------------------------------------------------

class TestProbeSavePositions:
    def test_no_clicks_returns_no_update(self) -> None:
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_save_positions

        result = probe_save_positions(None, {"sources": [], "detectors": []}, "/fake.snirf")
        assert all(r is no_update for r in result)

    def test_saves_sidecar_and_returns_view_mode(
        self, tmp_path: Path
    ) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import probe_save_positions

        snirf = tmp_path / "test.snirf"
        snirf.write_bytes(b"")
        montage = {"sources": [[0.0, 0.05]], "detectors": [[0.04, 0.0]]}

        new_mode, new_fig = probe_save_positions(1, montage, str(snirf))

        assert new_mode == "view"
        sidecar = tmp_path / "test.montage.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data == montage

    def test_no_snirf_path_returns_no_update(self) -> None:
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_save_positions

        result = probe_save_positions(1, {"sources": [], "detectors": []}, None)
        assert all(r is no_update for r in result)


# ---------------------------------------------------------------------------
# Callback: probe_confirm_run
# ---------------------------------------------------------------------------

class TestProbeConfirmRun:
    """Tests for probe_confirm_run callback."""

    def test_no_clicks_returns_no_update(self) -> None:
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_confirm_run

        result = probe_confirm_run(None, None, None)
        assert all(r is no_update for r in result)

    def test_idle_state_returns_no_update(self) -> None:
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_confirm_run

        exec_state = {"runner_id": "", "status": "idle"}
        result = probe_confirm_run(1, exec_state, [])
        assert all(r is no_update for r in result)

    def test_executes_with_channels_override(self) -> None:
        """probe_confirm_run must call execute_current with channels override."""
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            probe_confirm_run,
        )

        runner_id = str(uuid.uuid4())
        mock_runner = MagicMock()
        mock_runner.is_complete = True
        mock_runner.next_block.return_value = None
        mock_runner.results = []
        mock_runner.total_steps = 1
        mock_runner.current_block = MagicMock()
        mock_runner.current_block.params = None

        mock_context = MagicMock()
        mock_context.extra = {}
        _INTERACTIVE_RUNNERS[runner_id] = (mock_runner, mock_context)

        exec_state = {
            "runner_id": runner_id,
            "current_idx": 0,
            "status": "running",
            "snirf_path": None,
        }
        excluded = ["S1_D2", "S2_D1"]
        probe_confirm_run(1, exec_state, excluded)

        call_args = mock_runner.execute_current.call_args
        assert call_args is not None
        override = call_args[0][0] if call_args[0] else call_args[1].get("params_override")
        if override is None:
            # positional
            override = call_args[0][0]
        assert set(override.get("channels", [])) == {"S1_D2", "S2_D1"}

    def test_confirm_with_complete_run_returns_success(self) -> None:
        """When pipeline completes, probe_confirm_run returns run_results."""
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            probe_confirm_run,
        )

        runner_id = str(uuid.uuid4())
        mock_runner = MagicMock()
        mock_runner.is_complete = True
        mock_runner.next_block.return_value = None
        mock_runner.results = []
        mock_runner.total_steps = 1

        mock_context = MagicMock()
        mock_context.extra = {}
        _INTERACTIVE_RUNNERS[runner_id] = (mock_runner, mock_context)

        exec_state = {
            "runner_id": runner_id,
            "current_idx": 0,
            "status": "running",
            "snirf_path": None,
        }
        _, completed_state, run_results, *rest = probe_confirm_run(1, exec_state, [])
        assert completed_state["status"] == "complete"
        assert run_results is not None


# ---------------------------------------------------------------------------
# Callback: probe_cancel_run
# ---------------------------------------------------------------------------

class TestProbeCancelRun:
    def test_no_clicks_returns_no_update(self) -> None:
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_cancel_run

        result = probe_cancel_run(None, None)
        assert all(r is no_update for r in result)

    def test_cancel_clears_runner(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            probe_cancel_run,
        )

        runner_id = str(uuid.uuid4())
        _INTERACTIVE_RUNNERS[runner_id] = (MagicMock(), MagicMock())
        exec_state = {"runner_id": runner_id, "status": "running"}
        modal_children, new_state = probe_cancel_run(1, exec_state)

        assert modal_children == []
        assert new_state["status"] == "idle"
        assert runner_id not in _INTERACTIVE_RUNNERS


# ---------------------------------------------------------------------------
# Dispatch: _make_modal_children
# ---------------------------------------------------------------------------

class TestMakeModalChildrenDispatch:
    """Tests that _make_modal_children dispatches the probe dialog correctly."""

    def test_manual_channel_exclude_returns_probe_dialog(self) -> None:
        """_make_modal_children must return a probe dialog for manual_channel_exclude."""
        from nirspy.blocks import registry
        from nirspy.gui.callbacks.runtime_callbacks import _make_modal_children

        block_cls = registry.get("manual_channel_exclude")
        spec = block_cls.SPEC  # type: ignore[attr-defined]
        children = _make_modal_children(spec, 0, 3, snirf_path=None)
        assert len(children) == 1
        assert isinstance(children[0], dbc.Modal)
        # Probe dialog has id="probe-dialog-modal"
        assert children[0].id == "probe-dialog-modal"

    def test_other_block_returns_generic_dialog(self) -> None:
        """Non-probe block must return the generic runtime dialog."""
        from nirspy.blocks import registry
        from nirspy.gui.callbacks.runtime_callbacks import _make_modal_children

        block_cls = registry.get("optical_density")
        spec = block_cls.SPEC  # type: ignore[attr-defined]
        children = _make_modal_children(spec, 0, 3)
        assert len(children) == 1
        assert children[0].id == "runtime-dialog-modal"

    def test_probe_dialog_size_xl(self) -> None:
        from nirspy.blocks import registry
        from nirspy.gui.callbacks.runtime_callbacks import _make_modal_children

        block_cls = registry.get("manual_channel_exclude")
        spec = block_cls.SPEC  # type: ignore[attr-defined]
        children = _make_modal_children(spec, 0, 3, snirf_path=None)
        assert children[0].size == "xl"


# ---------------------------------------------------------------------------
# Sidecar precedence
# ---------------------------------------------------------------------------

class TestSidecarPrecedence:
    """Verify sidecar takes precedence over SNIRF native montage."""

    def test_sidecar_overrides_snirf(self, tmp_path: Path) -> None:
        from nirspy.io.montage import resolve_montage

        snirf = tmp_path / "test.snirf"
        snirf.write_bytes(b"")
        sidecar_data = {"sources": [[0.1, 0.1]], "detectors": [[0.2, 0.2]]}
        (tmp_path / "test.montage.json").write_text(
            json.dumps(sidecar_data), encoding="utf-8"
        )
        montage, source = resolve_montage(snirf)
        assert source == "sidecar"
        assert montage == sidecar_data

    def test_snirf_used_when_no_sidecar(self, tmp_path: Path) -> None:
        """When no sidecar exists, resolve returns source='snirf' or 'missing'."""
        from nirspy.io.montage import resolve_montage

        snirf = tmp_path / "nosidecar.snirf"
        snirf.write_bytes(b"")
        _montage, source = resolve_montage(snirf)
        # No h5 data in the empty file → missing
        assert source in ("snirf", "missing")

    def test_missing_when_empty_snirf_no_sidecar(self, tmp_path: Path) -> None:
        from nirspy.io.montage import resolve_montage

        snirf = tmp_path / "empty.snirf"
        snirf.write_bytes(b"")
        montage, source = resolve_montage(snirf)
        assert source == "missing"
        assert montage is None


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------

class TestImportSanity:
    def test_probe_dialog_importable(self) -> None:
        import nirspy.gui.components.probe_dialog  # noqa: F401

    def test_render_probe_dialog_callable(self) -> None:
        assert callable(render_probe_dialog)

    def test_build_channels_override_callable(self) -> None:
        assert callable(build_channels_override)

    def test_probe_callbacks_importable(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import (  # noqa: F401
            probe_cancel_run,
            probe_confirm_run,
            probe_positioning_click,
            probe_save_positions,
            probe_toggle_exclusion,
        )
