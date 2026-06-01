"""Pipeline callbacks -- add, remove, reorder, select, toggle enable.

All mutation goes through ``dcc.Store("pipeline-state")`` and
``dcc.Store("selected-block")``.  Callbacks are stateless; the store
is the single source of truth.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

from dash import ALL, Input, Output, State, callback, ctx, no_update

from nirspy.blocks import registry
from nirspy.domain.block import BlockSpec
from nirspy.gui.components.param_editor import render_param_editor
from nirspy.gui.components.pipeline_view import render_pipeline_view

# Re-export for app.py side-effect import
REGISTERED: bool = True


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input({"type": "catalog-item", "block_id": ALL}, "n_clicks"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def add_block(n_clicks: list[int | None], pipeline_state: list[dict[str, Any]]) -> Any:
    """Append a block to the pipeline when a catalog item is clicked."""
    if not ctx.triggered_id or all(c is None for c in n_clicks):
        return no_update

    block_id: str = ctx.triggered_id["block_id"]

    block_cls = registry.get(block_id)
    spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]
    default_params: dict[str, Any] = {}
    if spec.params_class is not None and dataclasses.is_dataclass(spec.params_class):
        try:
            default_obj = spec.params_class()
            default_params = dataclasses.asdict(default_obj)
        except TypeError:
            for f in dataclasses.fields(spec.params_class):
                if f.default is not dataclasses.MISSING:
                    default_params[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    default_params[f.name] = f.default_factory()
                else:
                    default_params[f.name] = None

    new_entry: dict[str, Any] = {
        "block_id": block_id,
        "instance_id": str(uuid.uuid4()),
        "params": default_params,
        "enabled": True,
    }

    state = list(pipeline_state) if pipeline_state else []
    state.append(new_entry)
    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Output("selected-block", "data", allow_duplicate=True),
    Input({"type": "btn-remove", "instance_id": ALL}, "n_clicks"),
    State("pipeline-state", "data"),
    State("selected-block", "data"),
    prevent_initial_call=True,
)
def remove_block(
    n_clicks: list[int | None],
    pipeline_state: list[dict[str, Any]],
    selected: str | None,
) -> tuple[Any, Any]:
    """Remove a block from the pipeline."""
    if not ctx.triggered_id or all(c is None for c in n_clicks):
        return no_update, no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    state = [e for e in pipeline_state if e["instance_id"] != instance_id]
    new_selected = None if selected == instance_id else selected
    return state, new_selected


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input({"type": "btn-up", "instance_id": ALL}, "n_clicks"),
    Input({"type": "btn-down", "instance_id": ALL}, "n_clicks"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def reorder_blocks(
    up_clicks: list[int | None],
    down_clicks: list[int | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Move a block up or down in the pipeline."""
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    direction: str = ctx.triggered_id["type"]

    state = list(pipeline_state)
    idx = next((i for i, e in enumerate(state) if e["instance_id"] == instance_id), None)
    if idx is None:
        return no_update

    if direction == "btn-up" and idx > 0:
        state[idx], state[idx - 1] = state[idx - 1], state[idx]
    elif direction == "btn-down" and idx < len(state) - 1:
        state[idx], state[idx + 1] = state[idx + 1], state[idx]
    else:
        return no_update

    return state


@callback(
    Output("selected-block", "data", allow_duplicate=True),
    Input({"type": "block-card", "instance_id": ALL}, "n_clicks"),
    State("selected-block", "data"),
    prevent_initial_call=True,
)
def select_block(
    n_clicks: list[int | None],
    current_selected: str | None,
) -> Any:
    """Set the selected block when a card is clicked."""
    if not ctx.triggered_id:
        return no_update
    # Identify the n_clicks for the specific card that triggered. If it is
    # zero/None we are on the initial render fire — don't change selection.
    triggered_index = next(
        (
            i
            for i, inp in enumerate(ctx.inputs_list[0])
            if inp.get("id") == ctx.triggered_id
        ),
        None,
    )
    if triggered_index is None:
        return no_update
    n = n_clicks[triggered_index]
    if not n:
        return no_update
    instance_id: str = ctx.triggered_id["instance_id"]
    if instance_id == current_selected:
        return None
    return instance_id


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input({"type": "switch-enable", "instance_id": ALL}, "value"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def toggle_enable(
    values: list[bool],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Update enabled state of a block."""
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    state = list(pipeline_state)

    triggered_value = ctx.triggered[0]["value"] if ctx.triggered else None
    if triggered_value is not None:
        for entry in state:
            if entry["instance_id"] == instance_id:
                entry["enabled"] = bool(triggered_value)
                break

    return state


@callback(
    Output("pipeline-view", "children"),
    Input("pipeline-state", "data"),
    Input("selected-block", "data"),
)
def render_pipeline(
    pipeline_state: list[dict[str, Any]] | None,
    selected: str | None,
) -> Any:
    """Re-render the pipeline view when state changes."""
    state = pipeline_state or []
    return render_pipeline_view(state, selected)


@callback(
    Output("param-editor", "children"),
    Input("selected-block", "data"),
    Input("pipeline-state", "data"),
)
def render_params(
    selected: str | None,
    pipeline_state: list[dict[str, Any]] | None,
) -> Any:
    """Re-render the parameter editor when the selection or state changes."""
    if not selected or not pipeline_state:
        return render_param_editor(
            block_id=None,
            instance_id=None,
            params_class=None,
            current_values={},
        )

    entry = next((e for e in pipeline_state if e["instance_id"] == selected), None)
    if entry is None:
        return render_param_editor(
            block_id=None,
            instance_id=None,
            params_class=None,
            current_values={},
        )

    block_id: str = entry["block_id"]
    block_cls = registry.get(block_id)
    spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]

    # For blocks with per-condition widget (block_average, epochs_extraction),
    # harvest available condition names + SNIRF path from any LoadSnirf step in
    # the pipeline so the editor dropdown and timeline (T-030) are pre-populated
    # without requiring a pipeline run first.
    _BLOCKS_WITH_CONDITION_WIDGET = ("block_average", "epochs_extraction")
    available_conditions: list[str] | None = None
    snirf_path: str | None = None
    if block_id in _BLOCKS_WITH_CONDITION_WIDGET:
        from nirspy.gui.components.condition_windows_editor import (
            read_snirf_condition_names,
        )

        for step in pipeline_state:
            if step.get("block_id") == "load_snirf":
                raw_path = step.get("params", {}).get("path")
                if raw_path:
                    snirf_path = str(raw_path)
                    names = read_snirf_condition_names(snirf_path)
                    if names:
                        available_conditions = names
                    break

    return render_param_editor(
        block_id=block_id,
        instance_id=entry["instance_id"],
        params_class=spec.params_class,
        current_values=entry.get("params", {}),
        available_conditions=available_conditions,
        snirf_path=snirf_path,
    )
