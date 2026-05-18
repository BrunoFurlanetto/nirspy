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
