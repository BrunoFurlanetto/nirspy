"""Execution callbacks -- run pipeline in background, store results.

Uses a module-level cache (``_VIZ_CACHE``) to hold MNE objects that
cannot be serialized to JSON.  The ``dcc.Store("run-results")`` only
holds a JSON-safe summary with a cache key so that viz callbacks can
retrieve the actual data.

Design note (ADR-016): background callbacks use diskcache as the
Dash callback manager backend.  The MNE objects themselves stay in
the in-process ``_VIZ_CACHE`` dict -- diskcache is used only by Dash
internally for callback orchestration.
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from dash import Input, Output, State, callback, clientside_callback, html, no_update

from nirspy.blocks import registry
from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.exceptions import NirspyError
from nirspy.engine.exceptions import get_user_message
from nirspy.gui.components.error_display import render_error

logger = logging.getLogger(__name__)

# Module-level cache for MNE objects -- single-user local app.
_VIZ_CACHE: dict[str, Any] = {}

clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return [false, '', false, 'info'];
        return [
            true,
            'Running pipeline… This may take a few minutes for GLM analysis.',
            true,
            'info'
        ];
    }
    """,
    Output("run-button", "disabled", allow_duplicate=True),
    Output("run-status", "children", allow_duplicate=True),
    Output("run-status", "is_open", allow_duplicate=True),
    Output("run-status", "color", allow_duplicate=True),
    Input("run-button", "n_clicks"),
    prevent_initial_call=True,
)


def _build_pipeline_from_state(
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Reconstruct a Pipeline object from GUI pipeline-state."""
    from nirspy.domain.pipeline import Pipeline

    steps: list[Any] = []
    for entry in pipeline_state:
        block_id: str = entry["block_id"]
        enabled: bool = entry.get("enabled", True)
        params_dict: dict[str, Any] = entry.get("params", {})

        block_cls = registry.get(block_id)
        spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]

        # Instantiate block with params. We deliberately *do not* swallow
        # TypeError/ValueError here -- silently falling back to a default
        # block instance was hiding bad user input (e.g. an unknown param
        # name or a malformed per_condition_windows dict), so the pipeline
        # ran with stale defaults and the user wondered why their changes
        # had no effect.
        if (
            spec.params_class is not None
            and dataclasses.is_dataclass(spec.params_class)
            and params_dict
        ):
            # Strip GUI-only meta-fields (prefixed with "_") before passing
            # to the frozen params dataclass which rejects unknown kwargs.
            params_dict = {
                k: v for k, v in params_dict.items() if not k.startswith("_")
            }

            # T-041: epochs_extraction stores per-condition groups in
            # "per_condition_groups" (a GUI-side dict keyed by label) because
            # it reuses the block_average group callbacks.  Convert to the
            # "groups" list[ConditionGroup] expected by EpochsExtractionParams.
            if block_id == "epochs_extraction":
                pcg = params_dict.pop("per_condition_groups", None)
                if pcg and isinstance(pcg, dict):
                    import contextlib

                    from nirspy.blocks.analysis import ConditionGroup

                    groups_list = []
                    for _lbl, val in pcg.items():
                        if isinstance(val, dict):
                            with contextlib.suppress(TypeError, ValueError):
                                groups_list.append(ConditionGroup(**val))
                        elif hasattr(val, "condition_names"):
                            groups_list.append(val)
                    if groups_list:
                        params_dict["groups"] = groups_list

            try:
                params_instance = spec.params_class(**params_dict)
            except (TypeError, ValueError) as exc:
                raise NirspyError(
                    f"Invalid parameters for '{block_id}': {exc}"
                ) from exc
            block_instance = block_cls(  # type: ignore[call-arg]
                params_instance
            )
        else:
            block_instance = block_cls()

        # Override enabled flag on spec (always — toggling back to True must propagate)
        object.__setattr__(block_instance.spec, "enabled", enabled)

        steps.append(block_instance)

    return Pipeline(
        name="gui-pipeline",
        description="Pipeline assembled via GUI",
        steps=steps,
    )


@callback(
    Output("run-results", "data"),
    Output("run-progress", "value"),
    Output("run-progress", "max"),
    Output("run-progress", "style"),
    Output("run-error", "children"),
    Output("run-error", "is_open"),
    Output("run-success", "children"),
    Output("run-success", "is_open"),
    Output("run-button", "disabled"),
    Output("run-status", "children"),
    Output("run-status", "is_open"),
    Output("run-status", "color"),
    Input("run-button", "n_clicks"),
    State("pipeline-state", "data"),
    State("input-file-path", "data"),
    State("global-conditions-store", "data"),
    prevent_initial_call=True,
)
def run_pipeline_callback(
    n_clicks: int | None,
    pipeline_state: list[dict[str, Any]] | None,
    input_file_path: str | None,
    global_conditions_store_data: dict[str, Any] | None,
) -> tuple[Any, ...]:
    """Execute the pipeline and store results for visualization.

    This is a synchronous callback.  A true background callback
    with set_progress requires the DiskcacheManager to be wired
    into the app.  For the initial implementation we run
    synchronously -- the progress bar shows completion after
    execution finishes.
    """
    if not n_clicks or not pipeline_state:
        return (
            no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update,
        )

    # Override LoadSnirf path if user uploaded a file
    if input_file_path:
        for entry in pipeline_state:
            if entry["block_id"] == "load_snirf":
                entry.setdefault("params", {})["path"] = (
                    input_file_path
                )
                break

    # Build pipeline
    try:
        pipeline = _build_pipeline_from_state(pipeline_state)
        if global_conditions_store_data:
            from nirspy.domain.conditions import global_conditions_from_dict

            pipeline.global_conditions = global_conditions_from_dict(
                global_conditions_store_data
            )
    except (KeyError, ValueError, NirspyError) as exc:
        logger.exception("Failed to build pipeline")
        block_id = _extract_block_id(pipeline_state, exc)
        if isinstance(exc, NirspyError):
            msg = get_user_message(exc)
        else:
            msg = f"Failed to build pipeline: {exc}"
        if block_id:
            msg = f"[{block_id}] {msg}"
        return (
            no_update, 0, 100, {"display": "none"},
            render_error(msg), True,
            "", False,
            False, "", False, "danger",
        )

    # Execution
    from nirspy.domain.execution import (
        ExecutionContext,
        run_pipeline_sync,
    )

    enabled_count = sum(
        1 for s in pipeline.steps if s.spec.enabled
    )
    progress_value = 0

    def _progress(
        block_id: str, step: int, total: int
    ) -> None:
        nonlocal progress_value
        progress_value = step

    context = ExecutionContext(progress=_progress)

    try:
        results: list[BlockResult] = run_pipeline_sync(
            pipeline, context
        )
    except NirspyError as exc:
        logger.exception("Pipeline execution failed")
        block_id = getattr(exc, "block_id", None) or ""
        msg = get_user_message(exc)
        if block_id:
            msg = f"[{block_id}] {msg}"
        return (
            no_update, 0, 100, {"display": "none"},
            render_error(msg), True,
            "", False,
            False, "", False, "danger",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during pipeline execution")
        return (
            no_update, 0, 100, {"display": "none"},
            render_error(
                "An unexpected error occurred. "
                "Please check the log file for details."
            ),
            True,
            "", False,
            False, "", False, "danger",
        )

    # Cache results for viz callbacks. Clear previous entries so we don't
    # leak MNE objects (and so the user can never accidentally see stale
    # results from a previous run).
    _VIZ_CACHE.clear()
    cache_key = str(uuid.uuid4())
    _VIZ_CACHE[cache_key] = {
        "results": results,
        "timestamp": time.time(),
    }

    # Extract metadata summary for JSON store
    sci_values = context.extra.get("sci_values")
    has_evoked = any(_is_evoked(r.data) for r in results)
    has_raw = any(_is_raw(r.data) for r in results)

    summary: dict[str, Any] = {
        "cache_key": cache_key,
        "blocks_executed": len(results),
        "total_blocks": enabled_count,
        "has_raw": has_raw,
        "has_evoked": has_evoked,
        "has_sci": sci_values is not None,
    }

    return (
        summary,
        enabled_count,
        enabled_count,
        {"display": "block"},
        "",
        False,
        f"Pipeline executed successfully "
        f"({len(results)}/{enabled_count} blocks).",
        True,
        False, "", False, "info",
    )



def _extract_block_id(
    pipeline_state: list[dict[str, Any]] | None,
    exc: Exception,
) -> str:
    """Try to extract a block_id from the exception context."""
    block_id = getattr(exc, "block_id", None)
    if block_id:
        return str(block_id)
    if pipeline_state:
        msg = str(exc).lower()
        for entry in pipeline_state:
            if entry["block_id"].lower() in msg:
                return str(entry["block_id"])
    return ""


def _is_raw(data: Any) -> bool:
    """Check if data is an MNE Raw object."""
    try:
        import mne

        return isinstance(data, mne.io.BaseRaw)
    except ImportError:
        return False


def _is_evoked(data: Any) -> bool:
    """Check if data is an MNE Evoked or dict of Evoked."""
    try:
        import mne

        if isinstance(data, mne.Evoked):
            return True
        if isinstance(data, dict):
            return any(
                isinstance(v, mne.Evoked)
                for v in data.values()
            )
        return False
    except ImportError:
        return False


@callback(
    Output("input-file-path", "data"),
    Output("input-file-label", "children"),
    Output("pipeline-state", "data", allow_duplicate=True),
    Output("condition-config-state", "data", allow_duplicate=True),
    Output("condition-modal-open-trigger", "data", allow_duplicate=True),
    Input("upload-input-file", "filename"),
    State("upload-input-file", "contents"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def store_input_file(
    filename: str | None,
    contents: str | None,
    pipeline_state: list[dict[str, Any]] | None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Persist the uploaded SNIRF and propagate its path into LoadSnirf.

    The file is decoded from base64, written to a temp location and the
    resulting path is stored in three places:
      1. ``input-file-path`` store — read by run_pipeline_callback as a
         last-resort override.
      2. The label below the upload button (visual confirmation).
      3. Every ``load_snirf`` step in ``pipeline-state`` — so downstream
         callbacks (e.g. per-condition windows reading conditions from
         the SNIRF) see the path immediately, before any pipeline run.

    Additionally (T-042i): if the SNIRF file has annotations, the
    condition-config modal is populated and opened automatically so the
    user can configure global conditions before running the pipeline.
    """
    if not filename or not contents:
        return no_update, no_update, no_update, no_update, no_update

    content_string = (
        contents.split(",", 1)[1]
        if "," in contents
        else contents
    )
    raw_bytes = base64.b64decode(content_string)

    tmp_dir = Path(tempfile.gettempdir()) / "nirspy"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # SEC-INFO-01: sanitize filename to prevent path traversal
    safe_name = os.path.basename(filename)
    tmp_path = tmp_dir / safe_name
    tmp_path.write_bytes(raw_bytes)

    label = html.Small(
        f"Selected: {filename}",
        className="text-success",
    )

    updated_state: Any = no_update
    if pipeline_state:
        new_state = list(pipeline_state)
        changed = False
        for entry in new_state:
            if entry.get("block_id") == "load_snirf":
                params = dict(entry.get("params", {}))
                params["path"] = str(tmp_path)
                entry["params"] = params
                changed = True
        if changed:
            updated_state = new_state

    # T-042i: populate condition-config-state if SNIRF has annotations.
    # The open trigger is written to a *separate* store so that _populate_modal
    # no longer shares an output with _sync_condition_inputs, eliminating the
    # Dash callback serialisation that caused the keystroke race condition.
    condition_config_state: Any = no_update
    open_trigger: Any = no_update
    try:
        import mne  # noqa: I001
        from nirspy.gui.components.condition_config_modal import (
            build_conditions_from_annotations,
        )

        raw_snirf = mne.io.read_raw_snirf(str(tmp_path), preload=False, verbose=False)
        annotations = [
            {
                "description": str(ann["description"]),
                "onset": float(ann["onset"]),
                "duration": float(ann["duration"]),
            }
            for ann in raw_snirf.annotations
            if not str(ann["description"]).startswith("BAD")
            and str(ann["description"]) not in ("", "boundary")
        ]
        if annotations:
            conditions = build_conditions_from_annotations(annotations)
            condition_config_state = {
                "conditions": conditions,
                "groups": [],
            }
            # Fire the dedicated open trigger — _populate_modal listens on this
            # store rather than condition-config-state to avoid serialisation
            # with _sync_condition_inputs.
            open_trigger = {"ts": filename}
    except Exception:  # noqa: BLE001
        logger.debug(
            "Could not read annotations from SNIRF for condition config modal.",
            exc_info=True,
        )

    return str(tmp_path), label, updated_state, condition_config_state, open_trigger
