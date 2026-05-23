"""Runtime callbacks for interactive step-by-step pipeline execution (T-027).

Manages the lifecycle of a :class:`~nirspy.domain.execution.PipelineRunner`
instance across multiple Dash callback invocations.  Because ``PipelineRunner``
holds live MNE objects it cannot be JSON-serialized into ``dcc.Store``; instead
we use the same module-level dict pattern as ``_VIZ_CACHE`` in
``execution_callbacks.py`` (ADR-016).

State machine
-------------
``interactive-exec-state`` store holds::

    {
        "runner_id": str,        # key into _INTERACTIVE_RUNNERS
        "current_idx": int,      # 0-based index of the block whose dialog is open
        "status": "idle" | "running" | "complete" | "cancelled",
    }

On ``start_interactive_run``  → creates runner, calls start() + next_block(),
    renders first dialog, sets status = "running".
On ``advance_run``            → execute_current(), next_block(). If complete:
    finalises run (populates run-results), closes modal, status = "complete".
    Otherwise renders next dialog.
On ``cancel_run``             → removes runner from dict, closes modal,
    status = "cancelled".
On ``skip_block``             → calls next_block() without execute_current().
    Emits a visible warning because downstream blocks that depend on skipped
    output will fail.

Limitation: skipping a block that produces data consumed by a downstream block
will cause that downstream block to fail.  A warning badge is shown in the UI.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from typing import Any

from dash import Input, Output, State, callback, no_update

from nirspy.domain.block import BlockSpec
from nirspy.domain.exceptions import NirspyError
from nirspy.engine.exceptions import get_user_message
from nirspy.gui.callbacks.execution_callbacks import (
    _VIZ_CACHE,
    _build_pipeline_from_state,
    _is_evoked,
    _is_raw,
)
from nirspy.gui.components.error_display import render_error
from nirspy.gui.components.hrf_runtime_dialog import (
    build_hrf_params_override,
    render_hrf_runtime_dialog,
)
from nirspy.gui.components.probe_dialog import build_channels_override, render_probe_dialog
from nirspy.gui.components.runtime_dialog import render_runtime_dialog
from nirspy.io.montage import save_sidecar_montage

logger = logging.getLogger(__name__)

# Module-level dict holding live PipelineRunner instances.
# Keys are UUIDs stored in the "runner_id" field of interactive-exec-state.
_INTERACTIVE_RUNNERS: dict[str, Any] = {}

_IDLE_STATE: dict[str, Any] = {
    "runner_id": "",
    "current_idx": -1,
    "status": "idle",
}


def _extract_snirf_path(pipeline_state: list[dict[str, Any]]) -> str | None:
    """Return the SNIRF file path from the load_snirf block in *pipeline_state*.

    Returns *None* if the block is absent or has no path configured.
    """
    for entry in pipeline_state:
        if entry.get("block_id") == "load_snirf":
            path = entry.get("params", {}).get("path")
            return str(path) if path is not None else None
    return None


def _make_modal_children(
    spec: BlockSpec,
    current_idx: int,
    total: int,
    current_params: dict[str, Any] | None = None,
    snirf_path: str | None = None,
) -> list[Any]:
    """Return children list for the runtime-dialog-modal container.

    Dispatches a specialised dialog when *spec.block_id* matches a block that
    has a custom interactive dialog:

    - ``block_average``          → hrf_runtime_dialog.render_hrf_runtime_dialog (T-028)
    - ``manual_channel_exclude`` → probe_dialog.render_probe_dialog (T-029)
    - everything else            → runtime_dialog.render_runtime_dialog
    """
    if spec.block_id == "block_average":
        from nirspy.gui.components.condition_windows_editor import (
            read_snirf_condition_names,
        )
        available_conditions = read_snirf_condition_names(snirf_path)
        modal = render_hrf_runtime_dialog(
            block_spec=spec,
            current_idx=current_idx,
            total=total,
            available_conditions=available_conditions,
            current_params=current_params,
        )
    elif spec.block_id == "manual_channel_exclude":
        modal = render_probe_dialog(
            snirf_path=snirf_path,
            current_idx=current_idx,
            total=total,
        )
    else:
        modal = render_runtime_dialog(spec, current_idx, total, current_params)
    return [modal]


def _extract_current_params(
    runner: Any,
) -> dict[str, Any]:
    """Extract current block's default params for pre-filling the dialog."""
    block = runner.current_block
    if block is None:
        return {}
    params = getattr(block, "params", None)
    if params is None or not dataclasses.is_dataclass(params):
        return {}
    return {
        f.name: getattr(params, f.name)
        for f in dataclasses.fields(params)
    }


def _finalise_run(
    runner: Any,
    runner_id: str,
    context_extra: dict[str, Any],
) -> dict[str, Any]:
    """Cache results and return run-results summary (mirrors execution_callbacks)."""
    results = runner.results
    _VIZ_CACHE.clear()
    cache_key = str(uuid.uuid4())
    _VIZ_CACHE[cache_key] = {
        "results": results,
        "timestamp": time.time(),
    }
    # Remove runner now that it's complete
    _INTERACTIVE_RUNNERS.pop(runner_id, None)

    sci_values = context_extra.get("sci_values")
    has_evoked = any(_is_evoked(r.data) for r in results)
    has_raw = any(_is_raw(r.data) for r in results)

    return {
        "cache_key": cache_key,
        "blocks_executed": len(results),
        "total_blocks": runner.total_steps,
        "has_raw": has_raw,
        "has_evoked": has_evoked,
        "has_sci": sci_values is not None,
    }


# ---------------------------------------------------------------------------
# Callback 1 — start_interactive_run
# ---------------------------------------------------------------------------

@callback(
    Output("runtime-dialog-container", "children"),
    Output("interactive-exec-state", "data"),
    Output("run-interactive-error", "children"),
    Output("run-interactive-error", "is_open"),
    Output("hrf-runtime-state", "data"),
    Input("run-interactive-btn", "n_clicks"),
    State("pipeline-state", "data"),
    State("input-file-path", "data"),
    prevent_initial_call=True,
)
def start_interactive_run(
    n_clicks: int | None,
    pipeline_state: list[dict[str, Any]] | None,
    input_file_path: str | None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Create a PipelineRunner and open the first block's dialog.

    Also resets ``hrf-runtime-state`` so previous run's group configuration
    does not bleed into the new run.
    """
    _hrf_reset: dict[str, Any] = {"groups": [], "available_conditions": None}
    if not n_clicks or not pipeline_state:
        return no_update, no_update, no_update, no_update, no_update

    # Override LoadSnirf path if a file was uploaded (mirrors execution_callbacks)
    if input_file_path:
        for entry in pipeline_state:
            if entry["block_id"] == "load_snirf":
                entry.setdefault("params", {})["path"] = input_file_path
                break

    try:
        pipeline = _build_pipeline_from_state(pipeline_state)
    except (KeyError, NirspyError) as exc:
        logger.exception("Interactive run: failed to build pipeline")
        msg = get_user_message(exc) if isinstance(exc, NirspyError) else str(exc)
        return [], _IDLE_STATE, render_error(msg), True, _hrf_reset

    from nirspy.domain.execution import ExecutionContext, PipelineRunner

    context = ExecutionContext()
    runner = PipelineRunner(pipeline, context)
    runner.start()

    spec = runner.next_block()
    if spec is None:
        # Empty pipeline — nothing to run
        return [], _IDLE_STATE, "Pipeline has no enabled blocks.", True, _hrf_reset

    runner_id = str(uuid.uuid4())
    _INTERACTIVE_RUNNERS[runner_id] = (runner, context)

    snirf_path = _extract_snirf_path(pipeline_state)
    state: dict[str, Any] = {
        "runner_id": runner_id,
        "current_idx": runner.current_idx,
        "status": "running",
        "snirf_path": snirf_path,
    }

    # Pre-populate available_conditions in hrf-runtime-state so the add-group
    # callback can serve the correct dropdown options (T-028).
    from nirspy.gui.components.condition_windows_editor import (
        read_snirf_condition_names,
    )
    available_conditions = read_snirf_condition_names(snirf_path)
    hrf_state: dict[str, Any] = {
        "groups": [],
        "available_conditions": available_conditions,
    }

    current_params = _extract_current_params(runner)
    modal_children = _make_modal_children(
        spec, runner.current_idx, runner.total_steps, current_params, snirf_path
    )
    return modal_children, state, "", False, hrf_state


# ---------------------------------------------------------------------------
# Callback 2 — advance_run ("Run with these params")
# ---------------------------------------------------------------------------

@callback(
    Output("runtime-dialog-container", "children", allow_duplicate=True),
    Output("interactive-exec-state", "data", allow_duplicate=True),
    Output("run-results", "data", allow_duplicate=True),
    Output("run-interactive-error", "children", allow_duplicate=True),
    Output("run-interactive-error", "is_open", allow_duplicate=True),
    Output("run-success", "children", allow_duplicate=True),
    Output("run-success", "is_open", allow_duplicate=True),
    Input("runtime-advance-btn", "n_clicks"),
    State("interactive-exec-state", "data"),
    State("hrf-runtime-state", "data"),
    prevent_initial_call=True,
)
def advance_run(
    n_clicks: int | None,
    exec_state: dict[str, Any] | None,
    hrf_state: dict[str, Any] | None,
) -> tuple[Any, ...]:
    """Execute the current block then advance to the next, or finalise the run.

    When the current block is ``block_average`` and the HRF specialized dialog
    was shown, reads ``hrf-runtime-state`` to build a ``params_override`` with
    ``per_condition_groups`` (T-028).  For all other blocks, no override is
    applied (params come from the pipeline builder state).
    """
    no_op = (
        no_update, no_update, no_update,
        no_update, no_update, no_update, no_update,
    )
    if not n_clicks or not exec_state:
        return no_op

    runner_id: str = exec_state.get("runner_id", "")
    status: str = exec_state.get("status", "idle")
    if status != "running" or not runner_id:
        return no_op

    entry = _INTERACTIVE_RUNNERS.get(runner_id)
    if entry is None:
        return no_op
    runner, context = entry

    # Determine params_override: HRF groups dialog provides per_condition_groups
    # for block_average; all other blocks execute without override.
    current_block = runner.current_block
    current_block_id: str = (
        current_block.spec.block_id if current_block is not None else ""
    )
    params_override: dict[str, Any] | None = None
    if current_block_id == "block_average":
        params_override = build_hrf_params_override(hrf_state)
        # Strip internal marker not recognised by BlockAverageParams
        if params_override and "_hrf_mode" in params_override:
            params_override = {
                k: v for k, v in params_override.items() if k != "_hrf_mode"
            }

    try:
        runner.execute_current(params_override)
    except NirspyError as exc:
        logger.exception("Interactive run: block execution failed")
        msg = get_user_message(exc)
        return (
            no_update, no_update, no_update,
            render_error(msg), True,
            no_update, no_update,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Interactive run: unexpected error")
        return (
            no_update, no_update, no_update,
            render_error("Unexpected error. Check the log for details."),
            True, no_update, no_update,
        )

    # Advance to next block
    spec = runner.next_block()
    if spec is None or runner.is_complete:
        # Pipeline complete — finalise
        context_extra: dict[str, Any] = context.extra
        run_results = _finalise_run(runner, runner_id, context_extra)
        completed_state: dict[str, Any] = {
            "runner_id": "",
            "current_idx": -1,
            "status": "complete",
        }
        success_msg = (
            f"Interactive run complete "
            f"({run_results['blocks_executed']}/{run_results['total_blocks']} blocks)."
        )
        return (
            [],  # close modal
            completed_state,
            run_results,
            "", False,
            success_msg, True,
        )

    # More blocks — render next dialog
    snirf_path_adv: str | None = exec_state.get("snirf_path")
    updated_state: dict[str, Any] = {
        "runner_id": runner_id,
        "current_idx": runner.current_idx,
        "status": "running",
        "snirf_path": snirf_path_adv,
    }
    current_params = _extract_current_params(runner)
    modal_children = _make_modal_children(
        spec, runner.current_idx, runner.total_steps, current_params, snirf_path_adv
    )
    return (
        modal_children,
        updated_state,
        no_update,
        "", False,
        "", False,
    )


# ---------------------------------------------------------------------------
# Callback 3 — cancel_run
# ---------------------------------------------------------------------------

@callback(
    Output("runtime-dialog-container", "children", allow_duplicate=True),
    Output("interactive-exec-state", "data", allow_duplicate=True),
    Input("runtime-cancel-btn", "n_clicks"),
    State("interactive-exec-state", "data"),
    prevent_initial_call=True,
)
def cancel_run(
    n_clicks: int | None,
    exec_state: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Discard the runner and close the dialog."""
    if not n_clicks or not exec_state:
        return no_update, no_update

    runner_id: str = exec_state.get("runner_id", "")
    _INTERACTIVE_RUNNERS.pop(runner_id, None)

    return [], _IDLE_STATE


# ---------------------------------------------------------------------------
# Callback 4 — skip_block
# ---------------------------------------------------------------------------

@callback(
    Output("runtime-dialog-container", "children", allow_duplicate=True),
    Output("interactive-exec-state", "data", allow_duplicate=True),
    Output("run-interactive-warning", "children", allow_duplicate=True),
    Output("run-interactive-warning", "is_open", allow_duplicate=True),
    Input("runtime-skip-btn", "n_clicks"),
    State("interactive-exec-state", "data"),
    prevent_initial_call=True,
)
def skip_block(
    n_clicks: int | None,
    exec_state: dict[str, Any] | None,
) -> tuple[Any, Any, Any, Any]:
    """Skip the current block without executing it.

    .. warning::
        Skipping a block that produces output consumed by a downstream block
        will cause that downstream block to fail when it runs.  The user is
        warned in the UI.
    """
    no_op = (no_update, no_update, no_update, no_update)
    if not n_clicks or not exec_state:
        return no_op

    runner_id: str = exec_state.get("runner_id", "")
    status: str = exec_state.get("status", "idle")
    if status != "running" or not runner_id:
        return no_op

    entry = _INTERACTIVE_RUNNERS.get(runner_id)
    if entry is None:
        return no_op
    runner, _context = entry

    skipped_block = runner.current_block
    skipped_name = (
        skipped_block.spec.display_name if skipped_block is not None else "unknown"
    )

    # Mark current block as skipped without executing — advance the pointer
    # by calling next_block (which moves current_idx without execute).
    # We need to manually advance current_idx because execute_current was
    # NOT called, so _block_ready is still True.  We call next_block() which
    # will consume the ready state and advance.
    spec = runner.next_block()

    warning_msg = (
        f"Block '{skipped_name}' was skipped.  "
        "Downstream blocks that depend on its output may fail."
    )

    if spec is None or runner.is_complete:
        # All remaining blocks were already executed (or nothing after skip)
        # Finalise with partial results
        context_extra: dict[str, Any] = _context.extra
        _finalise_run(runner, runner_id, context_extra)
        completed_state: dict[str, Any] = {
            "runner_id": "",
            "current_idx": -1,
            "status": "complete",
        }
        return [], completed_state, warning_msg, True

    snirf_path_skip_raw = exec_state.get("snirf_path") if exec_state else None
    snirf_path_skip: str | None = (
        str(snirf_path_skip_raw) if snirf_path_skip_raw is not None else None
    )
    updated_state: dict[str, Any] = {
        "runner_id": runner_id,
        "current_idx": runner.current_idx,
        "status": "running",
        "snirf_path": snirf_path_skip,
    }
    current_params = _extract_current_params(runner)
    modal_children = _make_modal_children(
        spec, runner.current_idx, runner.total_steps, current_params, snirf_path_skip
    )
    return modal_children, updated_state, warning_msg, True


# ---------------------------------------------------------------------------
# Callback 5 — probe_confirm_run  (ManualChannelExclude dialog)
# ---------------------------------------------------------------------------

@callback(
    Output("runtime-dialog-container", "children", allow_duplicate=True),
    Output("interactive-exec-state", "data", allow_duplicate=True),
    Output("run-results", "data", allow_duplicate=True),
    Output("run-interactive-error", "children", allow_duplicate=True),
    Output("run-interactive-error", "is_open", allow_duplicate=True),
    Output("run-success", "children", allow_duplicate=True),
    Output("run-success", "is_open", allow_duplicate=True),
    Input("probe-confirm-btn", "n_clicks"),
    State("interactive-exec-state", "data"),
    State("probe-excluded-store", "data"),
    prevent_initial_call=True,
)
def probe_confirm_run(
    n_clicks: int | None,
    exec_state: dict[str, Any] | None,
    excluded_channels: list[str] | None,
) -> tuple[Any, ...]:
    """Execute ManualChannelExclude with the user's exclusion choices.

    Reads the excluded channel prefixes from ``probe-excluded-store``,
    builds a ``params_override``, calls ``execute_current(params_override)``
    on the runner, then advances to the next block (or finalises the run).
    """
    no_op = (
        no_update, no_update, no_update,
        no_update, no_update, no_update, no_update,
    )
    if not n_clicks or not exec_state:
        return no_op

    runner_id: str = exec_state.get("runner_id", "")
    status: str = exec_state.get("status", "idle")
    if status != "running" or not runner_id:
        return no_op

    entry = _INTERACTIVE_RUNNERS.get(runner_id)
    if entry is None:
        return no_op
    runner, context = entry

    params_override = build_channels_override(excluded_channels or [])

    try:
        runner.execute_current(params_override)
    except NirspyError as exc:
        logger.exception("Interactive run (probe): block execution failed")
        msg = get_user_message(exc)
        return (
            no_update, no_update, no_update,
            render_error(msg), True,
            no_update, no_update,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Interactive run (probe): unexpected error")
        return (
            no_update, no_update, no_update,
            render_error("Unexpected error. Check the log for details."),
            True, no_update, no_update,
        )

    spec = runner.next_block()
    if spec is None or runner.is_complete:
        context_extra: dict[str, Any] = context.extra
        run_results = _finalise_run(runner, runner_id, context_extra)
        completed_state: dict[str, Any] = {
            "runner_id": "",
            "current_idx": -1,
            "status": "complete",
        }
        success_msg = (
            f"Interactive run complete "
            f"({run_results['blocks_executed']}/{run_results['total_blocks']} blocks)."
        )
        return (
            [],
            completed_state,
            run_results,
            "", False,
            success_msg, True,
        )

    snirf_path_confirm: str | None = exec_state.get("snirf_path")
    updated_state: dict[str, Any] = {
        "runner_id": runner_id,
        "current_idx": runner.current_idx,
        "status": "running",
        "snirf_path": snirf_path_confirm,
    }
    current_params = _extract_current_params(runner)
    modal_children = _make_modal_children(
        spec, runner.current_idx, runner.total_steps, current_params, snirf_path_confirm
    )
    return (
        modal_children,
        updated_state,
        no_update,
        "", False,
        "", False,
    )


# ---------------------------------------------------------------------------
# Callback 6 — probe_cancel_run  (ManualChannelExclude dialog cancel)
# ---------------------------------------------------------------------------

@callback(
    Output("runtime-dialog-container", "children", allow_duplicate=True),
    Output("interactive-exec-state", "data", allow_duplicate=True),
    Input("probe-cancel-btn", "n_clicks"),
    State("interactive-exec-state", "data"),
    prevent_initial_call=True,
)
def probe_cancel_run(
    n_clicks: int | None,
    exec_state: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Discard the runner and close the probe dialog."""
    if not n_clicks or not exec_state:
        return no_update, no_update

    runner_id: str = exec_state.get("runner_id", "")
    _INTERACTIVE_RUNNERS.pop(runner_id, None)

    return [], _IDLE_STATE


# ---------------------------------------------------------------------------
# Callback 7 — probe_toggle_exclusion  (click-to-exclude on probe graph)
# ---------------------------------------------------------------------------

@callback(
    Output("probe-excluded-store", "data", allow_duplicate=True),
    Output("probe-status-badge", "children", allow_duplicate=True),
    Output("probe-dialog-graph", "figure", allow_duplicate=True),
    Input("probe-dialog-graph", "clickData"),
    State("probe-excluded-store", "data"),
    State("probe-mode-store", "data"),
    State("probe-snirf-path-store", "data"),
    prevent_initial_call=True,
)
def probe_toggle_exclusion(
    click_data: dict[str, Any] | None,
    excluded_channels: list[str] | None,
    mode: str | None,
    snirf_path: str | None,
) -> tuple[Any, Any, Any]:
    """Toggle channel exclusion on click in view+exclude mode."""
    from nirspy.gui.components.probe_dialog import (
        _build_probe_figure,
        _build_status_badge,
    )

    if not click_data or mode != "view":
        return no_update, no_update, no_update

    points = click_data.get("points", [])
    if not points:
        return no_update, no_update, no_update

    point = points[0]
    customdata = point.get("customdata")
    if not customdata or not str(customdata).startswith(("S", "s")):
        # Only toggle channel midpoints (S<n>_D<m> prefixes), not optodes
        return no_update, no_update, no_update

    channel_label = str(customdata)
    # Only toggle S<n>_D<m> pairs (contains underscore), not plain S<n> or D<n>
    if "_" not in channel_label:
        return no_update, no_update, no_update

    excluded_set = set(excluded_channels or [])
    if channel_label in excluded_set:
        excluded_set.discard(channel_label)
    else:
        excluded_set.add(channel_label)

    new_excluded = list(excluded_set)

    # Re-build figure
    montage = None
    if snirf_path:
        from nirspy.io.montage import resolve_montage as _resolve
        montage, _ = _resolve(snirf_path)

    if montage is None:
        return new_excluded, no_update, no_update

    new_fig = _build_probe_figure(montage, excluded_set)
    new_badge = _build_status_badge(excluded_set)

    return new_excluded, new_badge.children, new_fig


# ---------------------------------------------------------------------------
# Callback 8 — probe_positioning_click  (2-click placement in positioning mode)
# ---------------------------------------------------------------------------

@callback(
    Output("probe-positioned-montage-store", "data", allow_duplicate=True),
    Output("probe-selected-optode-store", "data", allow_duplicate=True),
    Output("probe-dialog-graph", "figure", allow_duplicate=True),
    Input("probe-dialog-graph", "clickData"),
    State("probe-mode-store", "data"),
    State("probe-selected-optode-store", "data"),
    State("probe-positioned-montage-store", "data"),
    State("probe-optode-selector", "value"),
    prevent_initial_call=True,
)
def probe_positioning_click(
    click_data: dict[str, Any] | None,
    mode: str | None,
    selected_optode: str | None,
    positioned_montage: dict[str, Any] | None,
    selector_value: str | None,
) -> tuple[Any, Any, Any]:
    """Handle 2-click placement in positioning mode.

    First-click phase: if ``selected_optode`` is None, check ``selector_value``
    instead — user picked from the dropdown.  Second click on the graph
    places the optode at that coordinate.
    """
    from nirspy.gui.components.probe_dialog import _build_positioning_figure

    if not click_data or mode != "positioning":
        return no_update, no_update, no_update

    # Determine which optode to place
    optode_to_place = selected_optode or selector_value
    if not optode_to_place:
        return no_update, no_update, no_update

    points = click_data.get("points", [])
    if not points:
        return no_update, no_update, no_update

    point = points[0]
    x = point.get("x")
    y = point.get("y")
    if x is None or y is None:
        return no_update, no_update, no_update

    # Place the optode
    montage = dict(positioned_montage or {"sources": [], "detectors": []})
    sources: list[list[float]] = list(montage.get("sources", []))
    detectors: list[list[float]] = list(montage.get("detectors", []))

    label_upper = optode_to_place.upper()
    if label_upper.startswith("S"):
        idx_str = label_upper[1:]
        try:
            idx = int(idx_str) - 1
        except ValueError:
            return no_update, no_update, no_update
        while len(sources) <= idx:
            sources.append([0.0, 0.0])
        sources[idx] = [float(x), float(y)]
    elif label_upper.startswith("D"):
        idx_str = label_upper[1:]
        try:
            idx = int(idx_str) - 1
        except ValueError:
            return no_update, no_update, no_update
        while len(detectors) <= idx:
            detectors.append([0.0, 0.0])
        detectors[idx] = [float(x), float(y)]
    else:
        return no_update, no_update, no_update

    new_montage: dict[str, Any] = {"sources": sources, "detectors": detectors}
    new_fig = _build_positioning_figure(new_montage, selected_optode=None)

    return new_montage, None, new_fig


# ---------------------------------------------------------------------------
# Callback 9 — probe_save_positions  (save sidecar JSON)
# ---------------------------------------------------------------------------

@callback(
    Output("probe-mode-store", "data", allow_duplicate=True),
    Output("probe-dialog-graph", "figure", allow_duplicate=True),
    Input("probe-save-positions-btn", "n_clicks"),
    State("probe-positioned-montage-store", "data"),
    State("probe-snirf-path-store", "data"),
    prevent_initial_call=True,
)
def probe_save_positions(
    n_clicks: int | None,
    positioned_montage: dict[str, Any] | None,
    snirf_path: str | None,
) -> tuple[Any, Any]:
    """Save the in-progress positions to the sidecar JSON file."""
    from nirspy.gui.components.probe_dialog import _build_probe_figure

    if not n_clicks or not positioned_montage or not snirf_path:
        return no_update, no_update

    try:
        save_sidecar_montage(snirf_path, positioned_montage)
    except OSError:
        logger.exception("probe_save_positions: failed to write sidecar")
        return no_update, no_update

    # Switch to view+exclude mode now that positions are saved
    new_fig = _build_probe_figure(positioned_montage, set())
    new_fig.update_layout(
        title="Probe Layout — positions saved. Click channel to toggle exclusion."
    )
    return "view", new_fig
