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

T-028 additions
---------------
- ``start_interactive_run`` now resets ``hrf-runtime-state`` and populates
  ``available_conditions`` from the SNIRF file so the HRF groups dialog can
  auto-populate its condition Dropdowns.
- ``_make_modal_children`` dispatches ``render_hrf_runtime_dialog`` when the
  current block's ``block_id`` is ``"block_average"``.
- ``advance_run`` reads ``hrf-runtime-state`` and, if the current block is
  ``block_average``, builds a ``params_override`` with ``per_condition_groups``
  from the HRF dialog before calling ``execute_current``.
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
from nirspy.gui.components.runtime_dialog import render_runtime_dialog

logger = logging.getLogger(__name__)

# Module-level dict holding live PipelineRunner instances.
# Keys are UUIDs stored in the "runner_id" field of interactive-exec-state.
_INTERACTIVE_RUNNERS: dict[str, Any] = {}

_IDLE_STATE: dict[str, Any] = {
    "runner_id": "",
    "current_idx": -1,
    "status": "idle",
}


def _extract_snirf_path(
    pipeline_state: list[dict[str, Any]],
) -> str | None:
    """Return the SNIRF file path from the load_snirf block in *pipeline_state*."""
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

    - ``block_average`` → ``render_hrf_runtime_dialog`` (T-028)
    - everything else   → ``render_runtime_dialog``
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

    Also resets ``hrf-runtime-state`` so a previous run's group configuration
    does not bleed into the new run.  Pre-populates ``available_conditions``
    from the SNIRF file for the HRF groups dialog (T-028).
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

    # Pre-populate available_conditions in hrf-runtime-state (T-028)
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

    When the current block is ``block_average`` the HRF specialized dialog
    (T-028) may have collected ``per_condition_groups`` in ``hrf-runtime-state``.
    If so, a transient ``params_override`` is built and passed to
    ``execute_current`` so the HRF groups run without modifying the pipeline
    builder state.
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
        raw_override = build_hrf_params_override(hrf_state)
        if raw_override is not None:
            # Strip internal marker not recognised by BlockAverageParams
            params_override = {
                k: v for k, v in raw_override.items() if k != "_hrf_mode"
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
    snirf_path_adv: str | None = exec_state.get("snirf_path")
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

    snirf_path_skip: str | None = exec_state.get("snirf_path")
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
