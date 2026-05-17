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
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from dash import Input, Output, State, callback, html, no_update

from nirspy.blocks import registry
from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.exceptions import NirspyError

# Module-level cache for MNE objects -- single-user local app.
_VIZ_CACHE: dict[str, Any] = {}


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

        # Instantiate block with params
        if (
            spec.params_class is not None
            and dataclasses.is_dataclass(spec.params_class)
            and params_dict
        ):
            try:
                block_instance = block_cls(  # type: ignore[call-arg]
                    spec.params_class(**params_dict)
                )
            except (TypeError, ValueError):
                block_instance = block_cls()
        else:
            block_instance = block_cls()

        # Override enabled flag on spec if needed
        if not enabled:
            object.__setattr__(
                block_instance.spec, "enabled", False
            )

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
    Input("run-button", "n_clicks"),
    State("pipeline-state", "data"),
    State("input-file-path", "data"),
    prevent_initial_call=True,
)
def run_pipeline_callback(
    n_clicks: int | None,
    pipeline_state: list[dict[str, Any]] | None,
    input_file_path: str | None,
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
    except (KeyError, NirspyError) as exc:
        return (
            no_update, 0, 100, {"display": "none"},
            f"Failed to build pipeline: {exc}", True,
            "", False,
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
        return (
            no_update, 0, 100, {"display": "none"},
            f"Pipeline execution failed: {exc}", True,
            "", False,
        )
    except Exception:  # noqa: BLE001
        return (
            no_update, 0, 100, {"display": "none"},
            "An unexpected error occurred during execution.",
            True,
            "", False,
        )

    # Cache results for viz callbacks
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
    )


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
    Input("upload-input-file", "filename"),
    State("upload-input-file", "contents"),
    prevent_initial_call=True,
)
def store_input_file(
    filename: str | None,
    contents: str | None,
) -> tuple[Any, Any]:
    """Store the uploaded SNIRF file to a temp path.

    Dash dcc.Upload delivers file contents as a base64 string.
    We decode and write to a temporary file so the pipeline can
    read it via the normal file-based path.
    """
    if not filename or not contents:
        return no_update, no_update

    content_string = (
        contents.split(",", 1)[1]
        if "," in contents
        else contents
    )
    raw_bytes = base64.b64decode(content_string)

    # Write to temp file
    tmp_dir = Path(tempfile.gettempdir()) / "nirspy"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / filename
    tmp_path.write_bytes(raw_bytes)

    label = html.Small(
        f"Selected: {filename}",
        className="text-success",
    )
    return str(tmp_path), label
