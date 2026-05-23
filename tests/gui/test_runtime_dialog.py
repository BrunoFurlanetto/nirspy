"""Tests for T-027 -- Run Interactive button + generic runtime dialog.

Covers:
- render_runtime_dialog produces a Modal with ParamEditor for the correct block
- Header shows correct step counter format
- Three footer buttons are present (Skip, Cancel run, Run with these params)
- Modal uses backdrop="static" and keyboard=False
- start_interactive_run creates a runner and populates state
- advance_run advances current_idx
- cancel_run clears state and closes modal
- skip_block advances without executing and emits warning
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nirspy.blocks import registry
from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _spec_for(block_id: str) -> BlockSpec:
    """Retrieve the BlockSpec for a registered block."""
    block_cls = registry.get(block_id)
    return block_cls.SPEC  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# render_runtime_dialog tests
# ---------------------------------------------------------------------------

class TestRenderRuntimeDialog:
    """Unit tests for the render_runtime_dialog pure render function."""

    def test_returns_modal(self) -> None:
        """render_runtime_dialog must return a dbc.Modal."""
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog
        import dash_bootstrap_components as dbc

        spec = _spec_for("optical_density")
        modal = render_runtime_dialog(spec, 0, 5)
        assert isinstance(modal, dbc.Modal)

    def test_modal_id(self) -> None:
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("optical_density")
        modal = render_runtime_dialog(spec, 0, 5)
        assert modal.id == "runtime-dialog-modal"

    def test_backdrop_static(self) -> None:
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("optical_density")
        modal = render_runtime_dialog(spec, 0, 5)
        assert modal.backdrop == "static"

    def test_keyboard_false(self) -> None:
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("bandpass_filter")
        modal = render_runtime_dialog(spec, 1, 3)
        assert modal.keyboard is False

    def test_size_lg(self) -> None:
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("optical_density")
        modal = render_runtime_dialog(spec, 0, 5)
        assert modal.size == "lg"

    def test_is_open_true(self) -> None:
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("optical_density")
        modal = render_runtime_dialog(spec, 0, 5)
        assert modal.is_open is True

    def test_header_shows_step_counter(self) -> None:
        """Header must contain 'Block N/M: <display_name>'."""
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("optical_density")
        modal_str = str(render_runtime_dialog(spec, 2, 8))
        assert "Block 3/8" in modal_str
        assert spec.display_name in modal_str

    def test_header_step_1(self) -> None:
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("bandpass_filter")
        modal_str = str(render_runtime_dialog(spec, 0, 4))
        assert "Block 1/4" in modal_str

    def test_three_footer_buttons(self) -> None:
        """Footer must contain Skip, Cancel run, Run with these params."""
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("optical_density")
        modal_str = str(render_runtime_dialog(spec, 0, 5))
        assert "runtime-skip-btn" in modal_str
        assert "runtime-cancel-btn" in modal_str
        assert "runtime-advance-btn" in modal_str

    def test_footer_button_labels(self) -> None:
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("optical_density")
        modal_str = str(render_runtime_dialog(spec, 0, 5))
        assert "Skip" in modal_str
        assert "Cancel run" in modal_str
        assert "Run with these params" in modal_str

    def test_param_editor_present_for_block_with_params(self) -> None:
        """ParamEditor content appears in modal body for a block that has params."""
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("bandpass_filter")
        modal_str = str(render_runtime_dialog(spec, 0, 3))
        # ParamEditor renders block_id as h6 heading
        assert "bandpass_filter" in modal_str

    def test_no_params_block_renders_placeholder(self) -> None:
        """Block with no params_class renders 'No parameters' placeholder."""
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("optical_density")  # OpticalDensity has no params
        modal_str = str(render_runtime_dialog(spec, 0, 1))
        # ParamEditor renders 'No parameters' or block_id header
        assert "optical_density" in modal_str or "No parameters" in modal_str

    def test_current_params_pre_fill(self) -> None:
        """custom current_params override seeds form values."""
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog

        spec = _spec_for("bandpass_filter")
        modal_str = str(render_runtime_dialog(spec, 0, 3, {"l_freq": 0.05}))
        # The value 0.05 should appear somewhere in the rendered form
        assert "0.05" in modal_str


# ---------------------------------------------------------------------------
# Layout integration tests
# ---------------------------------------------------------------------------

class TestLayoutIntegration:
    """Verify layout.py wires up the new stores and container."""

    def test_interactive_exec_state_store(self) -> None:
        from nirspy.gui.layouts import create_layout
        layout_str = str(create_layout())
        assert "interactive-exec-state" in layout_str

    def test_runtime_dialog_container(self) -> None:
        from nirspy.gui.layouts import create_layout
        layout_str = str(create_layout())
        assert "runtime-dialog-container" in layout_str

    def test_run_interactive_btn_present(self) -> None:
        from nirspy.gui.layouts import create_layout
        layout_str = str(create_layout())
        assert "run-interactive-btn" in layout_str

    def test_run_interactive_error_alert(self) -> None:
        from nirspy.gui.layouts import create_layout
        layout_str = str(create_layout())
        assert "run-interactive-error" in layout_str

    def test_run_interactive_warning_alert(self) -> None:
        from nirspy.gui.layouts import create_layout
        layout_str = str(create_layout())
        assert "run-interactive-warning" in layout_str


# ---------------------------------------------------------------------------
# Callback: start_interactive_run
# ---------------------------------------------------------------------------

class TestStartInteractiveRun:
    """Unit tests for start_interactive_run callback logic."""

    def test_no_clicks_returns_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.callbacks.runtime_callbacks import start_interactive_run

        result = start_interactive_run(None, None, None)
        assert all(r is no_update for r in result)

    def test_empty_state_returns_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.callbacks.runtime_callbacks import start_interactive_run

        result = start_interactive_run(1, None, None)
        assert all(r is no_update for r in result)

    def test_creates_runner_and_populates_state(self) -> None:
        """start_interactive_run must return state with status='running'."""
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            start_interactive_run,
        )

        pipeline_state = [_make_entry("optical_density")]
        modal_children, exec_state, _err_msg, err_open = start_interactive_run(
            1, pipeline_state, None
        )
        assert exec_state["status"] == "running"
        assert exec_state["runner_id"] != ""
        assert exec_state["current_idx"] == 0
        assert err_open is False

        # Clean up
        _INTERACTIVE_RUNNERS.pop(exec_state["runner_id"], None)

    def test_modal_children_not_empty(self) -> None:
        """Modal children list must be non-empty after start."""
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            start_interactive_run,
        )

        pipeline_state = [_make_entry("optical_density")]
        modal_children, exec_state, _, _ = start_interactive_run(
            1, pipeline_state, None
        )
        assert len(modal_children) > 0

        _INTERACTIVE_RUNNERS.pop(exec_state["runner_id"], None)

    def test_invalid_block_returns_error(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import start_interactive_run

        pipeline_state = [_make_entry("nonexistent_block")]
        _, state, _msg, err_open = start_interactive_run(1, pipeline_state, None)
        assert err_open is True
        assert state["status"] == "idle"

    def test_runner_registered_in_dict(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            start_interactive_run,
        )

        pipeline_state = [_make_entry("optical_density")]
        _, exec_state, _, _ = start_interactive_run(1, pipeline_state, None)
        runner_id = exec_state["runner_id"]
        assert runner_id in _INTERACTIVE_RUNNERS

        _INTERACTIVE_RUNNERS.pop(runner_id, None)


# ---------------------------------------------------------------------------
# Callback: cancel_run
# ---------------------------------------------------------------------------

class TestCancelRun:
    """Unit tests for cancel_run callback."""

    def test_no_clicks_returns_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.callbacks.runtime_callbacks import cancel_run

        result = cancel_run(None, None)
        assert all(r is no_update for r in result)

    def test_cancel_clears_state(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            cancel_run,
        )

        runner_id = str(uuid.uuid4())
        _INTERACTIVE_RUNNERS[runner_id] = (MagicMock(), MagicMock())

        exec_state = {"runner_id": runner_id, "current_idx": 0, "status": "running"}
        modal_children, new_state = cancel_run(1, exec_state)

        assert modal_children == []
        assert new_state["status"] == "idle"
        assert new_state["runner_id"] == ""
        assert runner_id not in _INTERACTIVE_RUNNERS

    def test_cancel_removes_runner_from_dict(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            cancel_run,
        )

        runner_id = str(uuid.uuid4())
        _INTERACTIVE_RUNNERS[runner_id] = (MagicMock(), MagicMock())
        cancel_run(1, {"runner_id": runner_id, "status": "running"})
        assert runner_id not in _INTERACTIVE_RUNNERS


# ---------------------------------------------------------------------------
# Callback: advance_run
# ---------------------------------------------------------------------------

class TestAdvanceRun:
    """Unit tests for advance_run callback."""

    def test_no_clicks_returns_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.callbacks.runtime_callbacks import advance_run

        result = advance_run(None, None)
        assert all(r is no_update for r in result)

    def test_advance_with_more_blocks_updates_idx(self) -> None:
        """After advance, current_idx should be 1 when pipeline has 2 blocks."""
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            advance_run,
        )

        # Build a fully mocked runner so we avoid MNE dependencies
        runner_id = str(uuid.uuid4())
        mock_runner = MagicMock()
        mock_runner.is_complete = False

        # Simulate the second block spec returned by next_block
        next_spec = MagicMock()
        next_spec.display_name = "Beer-Lambert Law"
        next_spec.block_id = "beer_lambert"
        next_spec.params_class = None

        # After execute_current, next_block returns the second spec
        mock_runner.execute_current.return_value = BlockResult(
            data=None, block_id="optical_density"
        )
        mock_runner.next_block.return_value = next_spec
        mock_runner.current_idx = 1
        mock_runner.total_steps = 2
        mock_runner.current_block = MagicMock()
        mock_runner.current_block.params = None

        mock_context = MagicMock()
        mock_context.extra = {}
        _INTERACTIVE_RUNNERS[runner_id] = (mock_runner, mock_context)

        exec_state = {"runner_id": runner_id, "current_idx": 0, "status": "running"}
        modal_children, new_state, *_ = advance_run(1, exec_state)

        assert new_state["current_idx"] == 1
        assert new_state["status"] == "running"

        _INTERACTIVE_RUNNERS.pop(runner_id, None)

    def test_advance_idle_state_returns_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.callbacks.runtime_callbacks import advance_run

        exec_state = {"runner_id": "", "current_idx": -1, "status": "idle"}
        result = advance_run(1, exec_state)
        assert all(r is no_update for r in result)


# ---------------------------------------------------------------------------
# Callback: skip_block
# ---------------------------------------------------------------------------

class TestSkipBlock:
    """Unit tests for skip_block callback."""

    def test_no_clicks_returns_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.callbacks.runtime_callbacks import skip_block

        result = skip_block(None, None)
        assert all(r is no_update for r in result)

    def test_skip_emits_warning(self) -> None:
        """skip_block must return a non-empty warning message."""
        from nirspy.gui.callbacks.runtime_callbacks import (
            _INTERACTIVE_RUNNERS,
            skip_block,
        )

        runner_id = str(uuid.uuid4())
        mock_runner = MagicMock()
        mock_spec = MagicMock()
        mock_spec.display_name = "Optical Density"
        mock_block = MagicMock()
        mock_block.spec = mock_spec
        mock_runner.current_block = mock_block
        mock_runner.is_complete = True
        mock_runner.next_block.return_value = None
        mock_runner.results = []
        mock_runner.total_steps = 1

        mock_context = MagicMock()
        mock_context.extra = {}
        _INTERACTIVE_RUNNERS[runner_id] = (mock_runner, mock_context)

        exec_state = {"runner_id": runner_id, "current_idx": 0, "status": "running"}
        _, _, warning_msg, warning_open = skip_block(1, exec_state)

        assert warning_open is True
        assert "skipped" in warning_msg.lower() or "Optical Density" in warning_msg

    def test_skip_idle_returns_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.callbacks.runtime_callbacks import skip_block

        exec_state = {"runner_id": "", "status": "idle"}
        result = skip_block(1, exec_state)
        assert all(r is no_update for r in result)


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------

class TestImportSanity:
    def test_runtime_dialog_importable(self) -> None:
        import nirspy.gui.components.runtime_dialog  # noqa: F401

    def test_runtime_callbacks_importable(self) -> None:
        import nirspy.gui.callbacks.runtime_callbacks  # noqa: F401

    def test_render_runtime_dialog_callable(self) -> None:
        from nirspy.gui.components.runtime_dialog import render_runtime_dialog
        assert callable(render_runtime_dialog)

    def test_callbacks_callable(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import (
            advance_run,
            cancel_run,
            skip_block,
            start_interactive_run,
        )
        assert callable(start_interactive_run)
        assert callable(advance_run)
        assert callable(cancel_run)
        assert callable(skip_block)

    def test_interactive_runners_dict_exists(self) -> None:
        from nirspy.gui.callbacks.runtime_callbacks import _INTERACTIVE_RUNNERS
        assert isinstance(_INTERACTIVE_RUNNERS, dict)
