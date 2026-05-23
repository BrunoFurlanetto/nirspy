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
#
# Conditions are sourced exclusively from the upstream LoadSnirf file.
# There is no add / remove / rename — the row set is fixed by the SNIRF.
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
    """Update a single tmin/tmax/baseline field within per_condition_windows."""
    if not ctx.triggered_id:
        return no_update

    tid = ctx.triggered_id
    instance_id: str = tid["instance_id"]
    condition: str = tid["condition"]
    field_name: str = tid["field"]
    new_value = ctx.triggered[0]["value"] if ctx.triggered else None

    if field_name not in {"tmin", "tmax", "baseline_tmin", "baseline_tmax"}:
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

    When turning ON we read the condition names from the upstream
    LoadSnirf file and create one row per condition seeded with the
    block's global tmin/tmax/baseline values. When turning OFF the dict
    is cleared and globals apply.

    If no readable SNIRF is reachable, the switch is disabled in the
    UI and this callback can only ever be fired with ``False`` — but we
    still guard against an unexpected ON by writing ``{}`` (no rows).
    """
    from nirspy.gui.components.condition_windows_editor import (
        read_snirf_condition_names,
    )

    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    is_enabled = ctx.triggered[0]["value"] if ctx.triggered else False

    state = list(pipeline_state)

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
        if not is_enabled or not discovered:
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
            params["per_condition_windows"] = {
                cond: (existing[cond] if cond in existing else dict(defaults))
                for cond in discovered
            }
        entry["params"] = params
        break

    return state


# ---------------------------------------------------------------------------
# Per-condition groups callbacks (T-025)
#
# Groups are user-defined: each group has a label, a set of condition
# names drawn from the SNIRF, and temporal parameters.
#
# Storage format in pipeline-state::
#
#   params["per_condition_groups"] = {
#       "<label>": {
#           "label": "<label>",
#           "condition_names": [...],
#           "tmin": float, "tmax": float,
#           "baseline_tmin": float, "baseline_tmax": float,
#       },
#       ...
#   }
#
# The dict is keyed by label for fast lookup; the "label" key inside
# each value is redundant but makes ConditionGroup(**val) round-trip
# work in __post_init__.
# ---------------------------------------------------------------------------


def _get_groups_list(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per_condition_groups as an ordered list of plain dicts."""
    raw = params.get("per_condition_groups") or {}
    result: list[dict[str, Any]] = []
    for lbl, val in raw.items():
        if hasattr(val, "condition_names"):
            result.append(
                {
                    "label": lbl,
                    "condition_names": list(val.condition_names),
                    "event_indices": list(val.event_indices),
                    "tmin": val.tmin,
                    "tmax": val.tmax,
                    "baseline_tmin": val.baseline_tmin,
                    "baseline_tmax": val.baseline_tmax,
                }
            )
        elif isinstance(val, dict):
            entry = dict(val)
            entry.setdefault("label", lbl)
            entry.setdefault("event_indices", [])
            result.append(entry)
    return result


def _groups_list_to_dict(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert an ordered list of group dicts back to the keyed dict format."""
    out: dict[str, Any] = {}
    for g in groups:
        lbl = g.get("label") or ""
        if not lbl:
            lbl = f"Group {len(out) + 1}"
        out[lbl] = {
            "label": lbl,
            "condition_names": list(g.get("condition_names") or []),
            "event_indices": list(g.get("event_indices") or []),
            "tmin": float(g.get("tmin") or -2.0),
            "tmax": float(g.get("tmax") or 18.0),
            "baseline_tmin": float(g.get("baseline_tmin") or -2.0),
            "baseline_tmax": float(g.get("baseline_tmax") or 0.0),
        }
    return out


@callback(
    Output({"type": "cg-windows-panel", "instance_id": ALL}, "style"),
    Output({"type": "cg-groups-panel", "instance_id": ALL}, "style"),
    Input({"type": "cg-mode-radio", "instance_id": ALL}, "value"),
    prevent_initial_call=False,
)
def toggle_hrf_mode_visibility(
    values: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Show/hide the windows/groups editor panels based on the radio value."""
    windows_styles = [
        {"display": "block" if v == "windows" else "none"} for v in values
    ]
    groups_styles = [
        {"display": "block" if v == "groups" else "none"} for v in values
    ]
    return windows_styles, groups_styles


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input({"type": "cg-mode-radio", "instance_id": ALL}, "value"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def toggle_hrf_mode(
    values: list[str],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Switch between per-condition-windows and per-condition-groups modes.

    When switching to "groups": clears per_condition_windows so the mutual
    exclusion invariant in BlockAverageParams.__post_init__ is satisfied.
    When switching to "windows": clears per_condition_groups.
    """
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    new_mode = ctx.triggered[0]["value"] if ctx.triggered else "windows"

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] != instance_id:
            continue
        params = dict(entry.get("params", {}))
        params["_hrf_mode"] = new_mode
        if new_mode == "groups":
            params["per_condition_windows"] = {}
        else:
            params["per_condition_groups"] = {}
        entry["params"] = params
        break

    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input({"type": "cg-add", "instance_id": ALL}, "n_clicks"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def add_condition_group(
    n_clicks: list[int | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Append an empty group to per_condition_groups."""
    if not ctx.triggered_id:
        return no_update
    if not any(n for n in n_clicks if n):
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] != instance_id:
            continue
        params = dict(entry.get("params", {}))
        groups = _get_groups_list(params)
        # Generate a unique default label
        existing_labels = {g.get("label", "") for g in groups}
        idx = len(groups) + 1
        while f"Group {idx}" in existing_labels:
            idx += 1
        groups.append(
            {
                "label": f"Group {idx}",
                "condition_names": [],
                "tmin": float(params.get("tmin") or -2.0),
                "tmax": float(params.get("tmax") or 18.0),
                "baseline_tmin": float(params.get("baseline_tmin") or -2.0),
                "baseline_tmax": float(params.get("baseline_tmax") or 0.0),
            }
        )
        params["per_condition_groups"] = _groups_list_to_dict(groups)
        params["per_condition_windows"] = {}
        params["_hrf_mode"] = "groups"
        # Auto-select newly added group as active for the timeline (UX)
        params[f"_active_group_{instance_id}"] = f"Group {idx}"
        entry["params"] = params
        break

    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {"type": "cg-remove", "instance_id": ALL, "group_idx": ALL},
        "n_clicks",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def remove_condition_group(
    n_clicks: list[int | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Remove a group card by its index."""
    if not ctx.triggered_id:
        return no_update
    if not any(n for n in n_clicks if n):
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    group_idx: int = int(ctx.triggered_id["group_idx"])

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] != instance_id:
            continue
        params = dict(entry.get("params", {}))
        groups = _get_groups_list(params)
        if 0 <= group_idx < len(groups):
            groups.pop(group_idx)
        params["per_condition_groups"] = _groups_list_to_dict(groups)
        entry["params"] = params
        break

    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {"type": "cg-label", "instance_id": ALL, "group_idx": ALL},
        "value",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def update_group_label(
    values: list[str | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Update the label of a condition group."""
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    group_idx: int = int(ctx.triggered_id["group_idx"])
    new_label = ctx.triggered[0]["value"] if ctx.triggered else None
    if not new_label:
        return no_update

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] != instance_id:
            continue
        params = dict(entry.get("params", {}))
        groups = _get_groups_list(params)
        if 0 <= group_idx < len(groups):
            groups[group_idx] = dict(groups[group_idx])
            groups[group_idx]["label"] = new_label
        params["per_condition_groups"] = _groups_list_to_dict(groups)
        entry["params"] = params
        break

    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {"type": "cg-conditions", "instance_id": ALL, "group_idx": ALL},
        "value",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def update_group_conditions(
    values: list[list[str] | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Update the condition_names list for a group."""
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    group_idx: int = int(ctx.triggered_id["group_idx"])
    new_conditions = ctx.triggered[0]["value"] if ctx.triggered else None
    if new_conditions is None:
        new_conditions = []

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] != instance_id:
            continue
        params = dict(entry.get("params", {}))
        groups = _get_groups_list(params)
        if 0 <= group_idx < len(groups):
            groups[group_idx] = dict(groups[group_idx])
            groups[group_idx]["condition_names"] = list(new_conditions)
            # Mutual exclusion (D8): setting condition_names clears event_indices
            if new_conditions:
                groups[group_idx]["event_indices"] = []
        params["per_condition_groups"] = _groups_list_to_dict(groups)
        entry["params"] = params
        break

    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {
            "type": "cg-time",
            "instance_id": ALL,
            "group_idx": ALL,
            "field": ALL,
        },
        "value",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def update_group_time(
    values: list[float | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Update a tmin/tmax/baseline field for a condition group."""
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    group_idx: int = int(ctx.triggered_id["group_idx"])
    field_name: str = ctx.triggered_id["field"]
    new_value = ctx.triggered[0]["value"] if ctx.triggered else None

    if field_name not in {"tmin", "tmax", "baseline_tmin", "baseline_tmax"}:
        return no_update
    if new_value is None or new_value == "":
        return no_update

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] != instance_id:
            continue
        params = dict(entry.get("params", {}))
        groups = _get_groups_list(params)
        if 0 <= group_idx < len(groups):
            groups[group_idx] = dict(groups[group_idx])
            groups[group_idx][field_name] = float(new_value)
        params["per_condition_groups"] = _groups_list_to_dict(groups)
        entry["params"] = params
        break

    return state


# ---------------------------------------------------------------------------
# Timeline selection callbacks (T-030)
#
# Two callbacks:
# 1. toggle_event_in_group  -- click on timeline marker -> add/remove index
# 2. update_active_group    -- radio selector changes active group in state
#
# Active group label is persisted in pipeline-state as a UI-only key
# ``_active_group_<instance_id>`` so rerenders pick it up without a
# separate dcc.Store.
# ---------------------------------------------------------------------------


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {"type": "condition-timeline-graph", "instance_id": ALL},
        "clickData",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def toggle_event_in_group(
    click_data_list: list[dict[str, Any] | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Toggle a single event occurrence into/out of the active group.

    Reads the ``customdata[0]`` (event_index) from the clicked marker.
    Looks up the active group label from pipeline-state
    (``_active_group_<instance_id>``). Adds the index when not present;
    removes it when already there.

    When event_indices becomes non-empty for a group, condition_names is
    cleared for that group to maintain D8 mutual exclusion.
    """
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    click_data = ctx.triggered[0]["value"] if ctx.triggered else None
    if click_data is None:
        return no_update

    # Extract event_index from customdata
    try:
        points = click_data.get("points", [])
        if not points:
            return no_update
        event_index: int = int(points[0]["customdata"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return no_update

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] != instance_id:
            continue
        params = dict(entry.get("params", {}))
        active_label: str | None = params.get(f"_active_group_{instance_id}")
        if not active_label or active_label == "__none__":
            return no_update

        groups = _get_groups_list(params)
        for g in groups:
            if g.get("label") != active_label:
                continue
            g = dict(g)
            indices: list[int] = list(g.get("event_indices") or [])
            if event_index in indices:
                indices.remove(event_index)
            else:
                indices.append(event_index)
                indices.sort()
            g["event_indices"] = indices
            # Mutual exclusion (D8): clear condition_names when indices non-empty
            if indices:
                g["condition_names"] = []
            # Replace group in list
            for i, existing in enumerate(groups):
                if existing.get("label") == active_label:
                    groups[i] = g
                    break
            break

        params["per_condition_groups"] = _groups_list_to_dict(groups)
        entry["params"] = params
        break

    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input(
        {"type": "condition-timeline-active-group", "instance_id": ALL},
        "value",
    ),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def update_active_group(
    values: list[str | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Persist the active group label selection into pipeline-state.

    Stores the selection under ``_active_group_<instance_id>`` so that
    the timeline render function can read it on the next render cycle.
    The special value ``"__none__"`` means no group is selected.
    """
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    new_label = ctx.triggered[0]["value"] if ctx.triggered else None

    state = list(pipeline_state)
    for entry in state:
        if entry["instance_id"] != instance_id:
            continue
        params = dict(entry.get("params", {}))
        params[f"_active_group_{instance_id}"] = new_label or "__none__"
        entry["params"] = params
        break

    return state
