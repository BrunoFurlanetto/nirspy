"""Param callbacks -- update params in pipeline-state when form fields change."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from dash import ALL, Input, Output, State, callback, ctx, no_update


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input({"type": "param-input", "instance_id": ALL, "field": ALL}, "value"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def update_param(
    values: list[Any],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Update a single param value in the pipeline state."""
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    field_name: str = ctx.triggered_id["field"]
    new_value = ctx.triggered[0]["value"] if ctx.triggered else None

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] == instance_id:
            params = dict(entry.get("params", {}))

            # Attempt to parse JSON for complex types (dict, list)
            if isinstance(new_value, str) and new_value.strip().startswith(("{", "[")):
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    new_value = json.loads(new_value)

            # An empty input means "fall back to the dataclass default".
            # Dropping the key from params lets `block_cls(**params)` honour
            # the default — keeping the key with None would crash blocks
            # whose field type is non-Optional float/int.
            if new_value == "" or new_value is None:
                params.pop(field_name, None)
            else:
                params[field_name] = new_value
            entry["params"] = params
            break

    return state


# ---------------------------------------------------------------------------
# Per-condition windows callbacks (T-012)
# ---------------------------------------------------------------------------


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {"type": "cond-window-row", "instance_id": ALL,
         "condition": ALL, "field": ALL},
        "value",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def update_condition_window(
    values: list[Any],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Update a single field within per_condition_windows."""
    if not ctx.triggered_id:
        return no_update

    tid = ctx.triggered_id
    instance_id: str = tid["instance_id"]
    condition: str = tid["condition"]
    field_name: str = tid["field"]
    new_value = ctx.triggered[0]["value"] if ctx.triggered else None

    if field_name == "condition_name":
        # Renaming a condition is not handled here (complex UX)
        return no_update

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] == instance_id:
            params = dict(entry.get("params", {}))
            pcw = dict(params.get("per_condition_windows", {}))
            cond_dict = dict(pcw.get(condition, {}))
            if new_value is not None and new_value != "":
                cond_dict[field_name] = float(new_value)
            pcw[condition] = cond_dict
            params["per_condition_windows"] = pcw
            entry["params"] = params
            break

    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {"type": "cond-window-add", "instance_id": ALL},
        "n_clicks",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def add_condition_window(
    n_clicks: list[int | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Add a new empty condition row to per_condition_windows."""
    if not ctx.triggered_id or not any(n_clicks):
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] == instance_id:
            params = dict(entry.get("params", {}))
            pcw = dict(params.get("per_condition_windows", {}))
            # Generate a placeholder name
            idx = len(pcw) + 1
            name = f"condition_{idx}"
            while name in pcw:
                idx += 1
                name = f"condition_{idx}"
            pcw[name] = {
                "tmin": -2.0,
                "tmax": 18.0,
                "baseline_tmin": -2.0,
                "baseline_tmax": 0.0,
            }
            params["per_condition_windows"] = pcw
            entry["params"] = params
            break

    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {"type": "cond-window-remove", "instance_id": ALL,
         "condition": ALL},
        "n_clicks",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def remove_condition_window(
    n_clicks: list[int | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Remove a condition row from per_condition_windows."""
    if not ctx.triggered_id or not any(n_clicks):
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    condition: str = ctx.triggered_id["condition"]
    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] == instance_id:
            params = dict(entry.get("params", {}))
            pcw = dict(params.get("per_condition_windows", {}))
            pcw.pop(condition, None)
            params["per_condition_windows"] = pcw
            entry["params"] = params
            break

    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {"type": "cond-window-switch", "instance_id": ALL},
        "value",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def toggle_condition_windows(
    values: list[bool],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Toggle per-condition windows on/off.

    When turning ON we auto-populate one row per condition discovered in the
    upstream SNIRF file (read via ``read_snirf_condition_names``). The user
    cannot type a free-form condition name — they only edit the temporal
    windows. When turning OFF the dict is cleared and globals apply.
    """
    from nirspy.gui.components.condition_windows_editor import (
        read_snirf_condition_names,
    )

    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    is_enabled = ctx.triggered[0]["value"] if ctx.triggered else False

    state = list(pipeline_state)

    # Discover conditions from the LoadSnirf step (path is known)
    discovered: list[str] | None = None
    for step in state:
        if step.get("block_id") == "load_snirf":
            snirf_path = step.get("params", {}).get("path")
            discovered = read_snirf_condition_names(snirf_path)
            if discovered:
                break

    for entry in state:
        if entry["instance_id"] != instance_id:
            continue
        params = dict(entry.get("params", {}))
        if not is_enabled:
            params["per_condition_windows"] = {}
        else:
            existing = params.get("per_condition_windows") or {}
            defaults = {
                "tmin": float(params.get("tmin", -2.0) or -2.0),
                "tmax": float(params.get("tmax", 18.0) or 18.0),
                "baseline_tmin": float(
                    params.get("baseline_tmin", -2.0) or -2.0
                ),
                "baseline_tmax": float(
                    params.get("baseline_tmax", 0.0) or 0.0
                ),
            }
            if discovered:
                # One row per condition; preserve any existing edits
                new_pcw: dict[str, Any] = {}
                for cond in discovered:
                    new_pcw[cond] = (
                        existing[cond] if cond in existing else dict(defaults)
                    )
                params["per_condition_windows"] = new_pcw
            elif not existing:
                # No SNIRF visible yet — keep the previous fallback of one
                # placeholder row so the user can at least see the editor.
                params["per_condition_windows"] = {"condition_1": dict(defaults)}
        entry["params"] = params
        break

    return state
